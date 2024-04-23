use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{ConvTranspose2d, ConvTranspose2dConfig},
        Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{
        activation::{relu, sigmoid},
        backend::Backend,
        Device, Tensor,
    },
};

use super::{
    transformer::{TwoWayTransformer, TwoWayTransformerConfig},
    Activation,
};

#[derive(Module, Debug)]
struct MLP<B: Backend> {
    layers: Vec<Linear<B>>,
    num_layers: usize,
    sigmoid_output: bool,
}

impl<B: Backend> MLP<B> {
    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < self.num_layers - 1 {
                x = relu(x);
            }
        }
        let x = if self.sigmoid_output { sigmoid(x) } else { x };
        x
    }
}

#[derive(Config, Debug)]
struct MLPConfig {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    num_layers: usize,
    #[config(default = false)]
    sigmoid_output: bool,
}

impl MLPConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> MLP<B> {
        let initializer = Initializer::KaimingUniform {
            gain: 1.0 / libm::sqrt(5.0),
            fan_out_only: false,
        };
        // let mut layers = Vec::with_capacity(self.num_layers);
        // for i in 0..self.num_layers {
        //     let in_dim = if i == 0 { self.input_dim } else { self.hidden_dim };
        //     let out_dim = if i == self.num_layers - 1 {
        //         self.output_dim
        //     } else {
        //         self.hidden_dim
        //     };
        //     let layer = LinearConfig::new(in_dim, out_dim)
        //         .with_initializer(initializer)
        //         .init(device);
        //     layers.push(layer);
        // }
        let layers = (0..self.num_layers)
            .map(|i| {
                let in_dim = if i == 0 {
                    self.input_dim
                } else {
                    self.hidden_dim
                };
                let out_dim = if i == self.num_layers - 1 {
                    self.output_dim
                } else {
                    self.hidden_dim
                };
                LinearConfig::new(in_dim, out_dim)
                    .with_initializer(initializer.clone())
                    .init(device)
            })
            .collect();

        MLP {
            layers,
            num_layers: self.num_layers,
            sigmoid_output: self.sigmoid_output,
        }
    }
}

#[derive(Module, Debug)]
pub struct MaskDecoder<B: Backend> {
    transformer: TwoWayTransformer<B>,
    iou_token: Embedding<B>,
    num_mask_tokens: usize,
    mask_tokens: Embedding<B>,
    output_upscaling_conv1: ConvTranspose2d<B>,
    output_upscaling_ln: LayerNorm<B>,
    output_upscaling_conv2: ConvTranspose2d<B>,
    output_hypernetworks_mlps: Vec<MLP<B>>,
    iou_prediction_head: MLP<B>,
    activation: Activation,
}

impl<B: Backend> MaskDecoder<B> {
    pub fn forward(
        &self,
        image_embeddings: Tensor<B, 4>,
        image_pe: Tensor<B, 4>,
        sparse_prompt_embeddings: Tensor<B, 3>,
        dense_prompt_embeddings: Tensor<B, 4>,
        multimask_output: bool,
    ) -> (Tensor<B, 4>, Tensor<B, 3>) {
        let (masks, iou_pred) = self.predict_mask(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
        );

        let masks_dims = masks.dims();
        let iou_pred_dims = iou_pred.dims();
        let (masks, iou_pred) = if multimask_output {
            let masks = masks.slice([0..masks_dims[0], 0..masks_dims[1], 1..masks_dims[2]]);
            let iou_pred = iou_pred.slice([
                0..iou_pred_dims[0],
                0..iou_pred_dims[1],
                1..iou_pred_dims[2],
            ]);
            (masks, iou_pred)
        } else {
            let masks = masks.slice([0..masks_dims[0], 0..masks_dims[1], 0..1]);
            let iou_pred = iou_pred.slice([0..iou_pred_dims[0], 0..iou_pred_dims[1], 0..1]);
            (masks, iou_pred)
        };

        (masks, iou_pred)
    }

    fn predict_mask(
        &self,
        image_embeddings: Tensor<B, 4>,
        image_pe: Tensor<B, 4>,
        sparse_prompt_embeddings: Tensor<B, 3>,
        dense_prompt_embeddings: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 3>) {
        // concatenate output token
        let output_tokens = Tensor::cat(
            vec![self.iou_token.weight.val(), self.mask_tokens.weight.val()],
            0,
        );
        let output_tokens = output_tokens.unsqueeze::<3>().expand([
            sparse_prompt_embeddings.dims()[0] as i32,
            -1,
            -1,
        ]);
        let tokens = Tensor::cat(vec![output_tokens, sparse_prompt_embeddings], 1);

        // expand per-image data in batch direction to be per-mask
        let src = repeat_interleave(image_embeddings, tokens.dims()[0], 0);
        let src = src + dense_prompt_embeddings;
        let pos_src = repeat_interleave(image_pe, tokens.dims()[0], 0);
        let [b, c, h, w] = src.dims();

        // transformer
        let (hs, src) = self.transformer.forward(src, pos_src, tokens);
        let iou_token_out = hs.clone().slice([0..hs.clone().dims()[0], 0..1]);
        let mask_tokens_out = hs
            .clone()
            .slice([0..hs.dims()[0], 1..(1 + self.num_mask_tokens)]);

        // upscale mask embeddings and predict mask using the masks tokens
        let src = src.swap_dims(1, 2).reshape([b, c, h, w]);
        let upscaled_embedding = self.output_upscaling_conv1.forward(src);
        let upscaled_embedding = self.output_upscaling_ln.forward(upscaled_embedding);
        let upscaled_embedding = self.activation.forward(upscaled_embedding);
        let upscaled_embedding = self.output_upscaling_conv2.forward(upscaled_embedding);
        let upscaled_embedding = self.activation.forward(upscaled_embedding);

        let hyper_in_list = (0..self.num_mask_tokens)
            .map(|i| {
                self.output_hypernetworks_mlps[i].forward(
                    mask_tokens_out
                        .clone()
                        .slice([0..mask_tokens_out.dims()[0], i..i + 1]),
                )
            })
            .collect::<Vec<_>>();

        let hyper_in = Tensor::stack(hyper_in_list, 1);
        let [b, c, h, w] = upscaled_embedding.dims();
        let masks = hyper_in
            .matmul(upscaled_embedding.reshape([b, c, h * w]))
            .reshape([b as i32, -1, h as i32, w as i32]);

        // generate mask quality prediction
        let iou_pred = self.iou_prediction_head.forward(iou_token_out);
        (masks, iou_pred)
    }
}

fn repeat_interleave<B: Backend, const N: usize>(
    tensor: Tensor<B, N>,
    repeats: usize,
    axis: usize,
) -> Tensor<B, N> {
    let mut new_shape = tensor.dims().to_vec();
    new_shape[axis] *= repeats;
    let mut new_tensor = Tensor::<B, N>::zeros(new_shape, &tensor.device());
    let mut index = 0;
    for i in 0..tensor.dims()[axis] {
        for _ in 0..repeats {
            let slice = tensor.clone().slice([i..i + 1]);
            new_tensor = new_tensor.slice_assign([index..index + 1], slice);
            index += 1;
        }
    }
    new_tensor
}

#[derive(Config, Debug)]
pub struct MaskDecoderConfig {
    transformer: TwoWayTransformerConfig,
    transformer_dim: usize,
    #[config(default = 3)]
    num_multimask_outputs: usize,
    #[config(default = 3)]
    iou_head_depth: usize,
    #[config(default = 256)]
    iou_head_hidden_dim: usize,
}

impl MaskDecoderConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>, activation: Activation) -> MaskDecoder<B> {
        let iou_token = EmbeddingConfig::new(1, self.transformer_dim).init(device);
        let num_mask_tokens = self.num_multimask_outputs + 1;
        let mask_tokens = EmbeddingConfig::new(num_mask_tokens, self.transformer_dim).init(device);

        let output_upscaling_conv1 =
            ConvTranspose2dConfig::new([self.transformer_dim, self.transformer_dim / 4], [2, 2])
                .with_stride([2, 2])
                .with_initializer(Initializer::KaimingUniform {
                    gain: 1.0 / libm::sqrt(5.0),
                    fan_out_only: false,
                })
                .init(device);

        let output_upscaling_ln = LayerNormConfig::new(self.transformer_dim / 4).init(device);

        let output_upscaling_conv2 = ConvTranspose2dConfig::new(
            [self.transformer_dim / 4, self.transformer_dim / 8],
            [2, 2],
        )
        .with_stride([2, 2])
        .with_initializer(Initializer::KaimingUniform {
            gain: 1.0 / libm::sqrt(5.0),
            fan_out_only: false,
        })
        .init(device);

        let output_hypernetworks_mlps = (0..num_mask_tokens)
            .map(|_| {
                MLPConfig::new(
                    self.transformer_dim,
                    self.transformer_dim,
                    self.transformer_dim / 8,
                    3,
                )
                .init(device)
            })
            .collect();

        let iou_prediction_head = MLPConfig::new(
            self.transformer_dim,
            self.iou_head_hidden_dim,
            num_mask_tokens,
            self.iou_head_depth,
        )
        .init(device);

        MaskDecoder {
            transformer: self.transformer.init(device, activation),
            iou_token,
            num_mask_tokens,
            mask_tokens,
            output_upscaling_conv1,
            output_upscaling_ln,
            output_upscaling_conv2,
            output_hypernetworks_mlps,
            iou_prediction_head,
            activation,
        }
    }
}
