use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig,
    },
    tensor::{backend::Backend, Device, Distribution, Tensor},
};
use std::f64::consts::PI;

use super::Activation;

#[derive(Module, Debug)]
struct PositionEmbeddingRandom<B: Backend> {
    positional_encoding_gaussian_matrix: Tensor<B, 2>,
}

impl<B: Backend> PositionEmbeddingRandom<B> {
    fn pe_encoding(&self, coords: Tensor<B, 3>) -> Tensor<B, 3> {
        // assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        let coords = coords.mul_scalar(2.0).sub_scalar(1.0);
        let coords = matmul_for_difference_dims_3d_with_2d(
            coords,
            self.positional_encoding_gaussian_matrix.clone(),
        );
        let coords = coords * (2. * PI);
        // outputs d_1 x ... x d_n x C shape
        Tensor::cat(
            vec![coords.clone().sin(), coords.clone().cos()],
            coords.dims().len() - 1,
        )
    }

    fn forward(&self, size: (usize, usize)) -> Tensor<B, 3> {
        let (h, w) = size;
        let device = self.positional_encoding_gaussian_matrix.device();
        let grid = Tensor::ones([h, w], &device);
        let y_embed = cumsum2d(grid.clone(), 0)
            .sub_scalar(0.5)
            .div_scalar(h as f64);
        let x_embed = cumsum2d(grid.clone(), 1)
            .sub_scalar(0.5)
            .div_scalar(w as f64);

        let stack_tensor = Tensor::<B, 2>::stack::<3>(vec![x_embed, y_embed], 3 - 1);
        self.pe_encoding(stack_tensor).permute([2, 0, 1])
    }

    fn forward_with_coords(
        &self,
        coords_input: Tensor<B, 3>,
        image_size: (usize, usize),
    ) -> Tensor<B, 3> {
        let coords = coords_input.clone();
        let coords_dims = coords.dims();
        let coords = coords.clone().slice_assign(
            [0..coords_dims[0], 0..coords_dims[1], 0..1],
            coords
                .clone()
                .slice([0..coords_dims[0], 0..coords_dims[1], 0..1])
                .div_scalar(image_size.0 as f64),
        );
        let coords = coords.clone().slice_assign(
            [0..coords_dims[0], 0..coords_dims[1], 1..2],
            coords
                .clone()
                .slice([0..coords_dims[0], 0..coords_dims[1], 1..2])
                .div_scalar(image_size.1 as f64),
        );
        self.pe_encoding(coords)
    }
}

fn cumsum2d<B: Backend>(x: Tensor<B, 2>, dim: usize) -> Tensor<B, 2> {
    // Returns the cumulative sum of elements of input in the dimension dim.
    let dims = x.dims();
    let mut output_tensor = x.zeros_like();

    let dim_for_cumsum = if dim == 0 { dims[1] } else { dims[0] };

    let mut cumsum_value = Tensor::zeros([dim_for_cumsum], &x.device());

    for i in 0..dims[dim] {
        let slice_indices = match dim {
            0 => [i..i + 1, 0..dims[1]],
            1 => [0..dims[0], i..i + 1],
            _ => panic!("Invalid dimension"),
        };
        // Slice the input tensor along the specified dimension
        let slice_tensor = x.clone().slice(slice_indices.clone());

        // unsqueeze cumsum_value to [1, 1, 1, 1]
        let slice_tensor = if dim == 0 {
            slice_tensor.squeeze::<1>(0)
        } else {
            slice_tensor.squeeze::<1>(1)
        };
        cumsum_value = cumsum_value.clone() + slice_tensor;

        let cumsum2d_value = if dim == 0 {
            cumsum_value.clone().unsqueeze_dim(0)
        } else {
            cumsum_value.clone().unsqueeze_dim(1)
        };

        output_tensor = output_tensor.slice_assign(slice_indices, cumsum2d_value);
    }

    output_tensor
}

fn matmul_for_difference_dims_3d_with_2d<B: Backend>(
    x: Tensor<B, 3>,
    y: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let x_dims = x.dims();
    let y_dims = y.dims();

    let reshaped_tensor_3d = x.reshape([-1, x_dims[2] as i32]);
    let result = reshaped_tensor_3d.matmul(y);
    result.reshape([x_dims[0], x_dims[1], y_dims[1]])
}

#[derive(Config, Debug)]
struct PositionEmbeddingRandomConfig {
    #[config(default = "64")]
    num_post_feats: usize,
    scale: Option<f64>,
}

impl PositionEmbeddingRandomConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> PositionEmbeddingRandom<B> {
        let scale = self.scale.unwrap_or(1.0);
        let scale = if scale <= 0.0 { 1.0 } else { scale };
        let positional_encoding_gaussian_matrix = Tensor::<B, 2>::random(
            [2, self.num_post_feats],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let positional_encoding_gaussian_matrix =
            positional_encoding_gaussian_matrix.mul_scalar(scale);
        PositionEmbeddingRandom {
            positional_encoding_gaussian_matrix,
        }
    }
}

#[derive(Module, Debug)]
pub struct PromptEncoder<B: Backend> {
    embed_dim: usize,
    input_image_size: (usize, usize),
    image_embedding_size: (usize, usize),
    pe_layer: PositionEmbeddingRandom<B>,
    point_embedding: Vec<Embedding<B>>,
    not_a_point_embed: Embedding<B>,
    mask_downscalling_conv1: Conv2d<B>,
    mask_downscalling_ln1: LayerNorm<B>,
    mask_downscalling_conv2: Conv2d<B>,
    mask_downscalling_ln2: LayerNorm<B>,
    mask_downscalling_conv3: Conv2d<B>,
    activation: Activation,
    no_mask_embed: Embedding<B>,
}

impl<B: Backend> PromptEncoder<B> {
    pub fn forward(
        &self,
        // point coords BxNx2, point labels BxN
        points: Option<(Tensor<B, 3>, Tensor<B, 2>)>,
        // boxes Bx4
        boxes: Option<Tensor<B, 2>>,
        // masks Bx1xHxW
        masks: Option<Tensor<B, 4>>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let bs = self.get_batch_size(points.clone(), boxes.clone(), masks.clone());
        let sparse_embeddings = Tensor::empty([bs, 0, self.embed_dim], &self.get_device());
        let sparse_embeddings = if let Some((coords, labels)) = points {
            let point_embeddings = self.embed_points(coords, labels, boxes.is_none());
            Tensor::cat(vec![sparse_embeddings, point_embeddings], 1)
        } else {
            sparse_embeddings
        };

        let sparse_embeddings = if let Some(boxes) = boxes {
            let box_embeddings = self.embed_boxes(boxes);
            Tensor::cat(vec![sparse_embeddings, box_embeddings], 1)
        } else {
            sparse_embeddings
        };

        let dense_embeddings = if let Some(masks) = masks {
            self.embed_masks(masks)
        } else {
            self.no_mask_embed
                .weight
                .val()
                .reshape([1, -1, 1, 1])
                .expand([
                    bs as i32,
                    -1,
                    self.image_embedding_size.0 as i32,
                    self.image_embedding_size.1 as i32,
                ])
        };

        (sparse_embeddings, dense_embeddings)
    }

    pub fn get_dense_pe(&self) -> Tensor<B, 4> {
        // 1x(embed_dim)x(embedding_h)x(embedding_w)
        self.pe_layer.forward(self.image_embedding_size).unsqueeze()
    }

    fn embed_points(&self, points: Tensor<B, 3>, labels: Tensor<B, 2>, pad: bool) -> Tensor<B, 3> {
        let points = points.add_scalar(0.5);
        let (points, labels) = if pad {
            let padding_point = Tensor::zeros([points.dims()[0], 1, 2], &points.device());
            let padding_label =
                Tensor::ones([labels.dims()[0], 1], &labels.device()).mul_scalar(-1.0);
            let points = Tensor::cat(vec![points, padding_point], 1);
            let labels = Tensor::cat(vec![labels, padding_label], 1);
            (points, labels)
        } else {
            (points, labels)
        };

        let point_embedding = self
            .pe_layer
            .forward_with_coords(points, self.input_image_size);
        let labels_bool = labels.clone().equal_elem(-1).unsqueeze_dim::<3>(2);
        let point_embedding = point_embedding.mask_fill(labels_bool.clone(), 0);

        let temp_tensor =
            add_tensor_3d_and_2d(point_embedding.clone(), self.not_a_point_embed.weight.val());
        let point_embedding = point_embedding.mask_where(labels_bool, temp_tensor);

        let labels_bool = labels.clone().equal_elem(0).unsqueeze_dim::<3>(2);
        let temp_tensor = add_tensor_3d_and_2d(
            point_embedding.clone(),
            self.point_embedding[0].weight.val(),
        );
        let point_embedding = point_embedding.mask_where(labels_bool, temp_tensor);

        let labels_bool = labels.equal_elem(1).unsqueeze_dim::<3>(2);
        let temp_tensor = add_tensor_3d_and_2d(
            point_embedding.clone(),
            self.point_embedding[1].weight.val(),
        );
        let point_embedding = point_embedding.mask_where(labels_bool, temp_tensor);

        point_embedding
    }

    fn embed_boxes(&self, boxes: Tensor<B, 2>) -> Tensor<B, 3> {
        let boxes = boxes.add_scalar(0.5);
        let coords = boxes.reshape([-1, 2, 2]);
        let corner_embedding = self
            .pe_layer
            .forward_with_coords(coords, self.input_image_size);
        let corner_embedding_dims = corner_embedding.dims();
        let corner_embedding = corner_embedding.clone().slice_assign(
            [
                0..corner_embedding_dims[0],
                0..1,
                0..corner_embedding_dims[2],
            ],
            corner_embedding
                .slice([
                    0..corner_embedding_dims[0],
                    0..1,
                    0..corner_embedding_dims[2],
                ])
                .add(self.point_embedding[2].weight.val().unsqueeze_dim::<3>(1)),
        );
        let corner_embedding = corner_embedding.clone().slice_assign(
            [
                0..corner_embedding_dims[0],
                1..2,
                0..corner_embedding_dims[2],
            ],
            corner_embedding
                .slice([
                    0..corner_embedding_dims[0],
                    1..2,
                    0..corner_embedding_dims[2],
                ])
                .add(self.point_embedding[3].weight.val().unsqueeze_dim::<3>(1)),
        );

        corner_embedding
    }

    fn embed_masks(&self, masks: Tensor<B, 4>) -> Tensor<B, 4> {
        // doing downscalling
        let masks = self.mask_downscalling_conv1.forward(masks);
        let masks = self.mask_downscalling_ln1.forward(masks);
        let masks = self.activation.forward(masks);
        let masks = self.mask_downscalling_conv2.forward(masks);
        let masks = self.mask_downscalling_ln2.forward(masks);
        let masks = self.activation.forward(masks);
        let masks = self.mask_downscalling_conv3.forward(masks);
        masks
    }

    fn get_batch_size(
        &self,
        points: Option<(Tensor<B, 3>, Tensor<B, 2>)>,
        boxes: Option<Tensor<B, 2>>,
        masks: Option<Tensor<B, 4>>,
    ) -> usize {
        let batch_size = if let Some((points, _)) = points {
            points.dims()[0]
        } else if let Some(boxes) = boxes {
            boxes.dims()[0]
        } else if let Some(masks) = masks {
            masks.dims()[0]
        } else {
            1
        };
        batch_size
    }

    fn get_device(&self) -> B::Device {
        self.point_embedding[0].weight.device()
    }
}

fn add_tensor_3d_and_2d<B: Backend>(x: Tensor<B, 3>, y: Tensor<B, 2>) -> Tensor<B, 3> {
    let x_dims = x.dims();
    let y = y.unsqueeze_dim::<3>(1);
    let y = y.repeat(1, x_dims[1]);
    x + y
}

#[derive(Config, Debug)]
pub struct PromptEncoderConfig {
    embed_dim: usize,
    image_embedding_size: (usize, usize),
    input_image_size: (usize, usize),
    mask_in_channels: usize,
}

impl PromptEncoderConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>, activation: Activation) -> PromptEncoder<B> {
        let pe_layer = PositionEmbeddingRandomConfig::new()
            .with_num_post_feats(self.embed_dim / 2)
            .init(device);

        let num_point_embeddings = 4;
        let point_embedding = (0..num_point_embeddings)
            .map(|_| EmbeddingConfig::new(1, self.embed_dim).init(device))
            .collect::<Vec<_>>();

        let not_a_point_embed = EmbeddingConfig::new(1, self.embed_dim).init(device);

        let mask_downscalling_conv1 = Conv2dConfig::new([1, self.mask_in_channels / 4], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let mask_downscalling_ln1 = LayerNormConfig::new(self.mask_in_channels / 4).init(device);
        let mask_downscalling_conv2 =
            Conv2dConfig::new([self.mask_in_channels / 4, self.mask_in_channels], [2, 2])
                .with_stride([2, 2])
                .init(device);
        let mask_downscalling_ln2 = LayerNormConfig::new(self.mask_in_channels).init(device);
        let mask_downscalling_conv3 =
            Conv2dConfig::new([self.mask_in_channels, self.embed_dim], [2, 2])
                .with_stride([2, 2])
                .init(device);

        let no_mask_embed = EmbeddingConfig::new(1, self.embed_dim).init(device);

        PromptEncoder {
            embed_dim: self.embed_dim,
            input_image_size: self.input_image_size,
            image_embedding_size: self.image_embedding_size,
            pe_layer,
            point_embedding,
            not_a_point_embed,
            mask_downscalling_conv1,
            mask_downscalling_ln1,
            mask_downscalling_conv2,
            mask_downscalling_ln2,
            mask_downscalling_conv3,
            activation,
            no_mask_embed,
        }
    }
}
