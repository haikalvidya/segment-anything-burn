use crate::model::MLPBlock;
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d,
    },
    tensor::{activation::softmax, backend::Backend, Device, Float, Tensor},
};

use super::{Activation, MLPBlockConfig};

#[derive(Module, Debug)]
struct Attention<B: Backend> {
    num_heads: usize,
    scale: f64,
    qkv: Linear<B>,
    proj: Linear<B>,
    rel_pos_hw: Option<(Tensor<B, 2>, Tensor<B, 2>)>,
}

impl<B: Backend> Attention<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let b = x.dims()[0] as i32;
        let h = x.dims()[1] as i32;
        let w = x.dims()[2] as i32;

        let qkv = self
            .qkv
            .forward(x)
            .reshape([b, h * w, 3, self.num_heads as i32, -1])
            .permute([2, 0, 3, 1, 4])
            .reshape([3, b * self.num_heads as i32, h * w, -1]);

        let qkv_dims = qkv.dims();
        // TODO check if the delete/squeeze the first dim
        // delete first dimension on each tensor
        let q = qkv
            .clone()
            .slice([0..1, 0..qkv_dims[1], 0..qkv_dims[2], 0..qkv_dims[3]])
            .squeeze::<3>(0);
        let k = qkv
            .clone()
            .slice([1..2, 0..qkv_dims[1], 0..qkv_dims[2], 0..qkv_dims[3]])
            .squeeze::<3>(0);
        let v = qkv
            .slice([2..3, 0..qkv_dims[1], 0..qkv_dims[2], 0..qkv_dims[3]])
            .squeeze::<3>(0);

        let attn = q
            .clone()
            .mul_scalar(self.scale)
            .matmul(k.swap_dims(4 - 2, 4 - 1));

        let attn = match &self.rel_pos_hw {
            Some((rel_pos_h, rel_pos_w)) => self.add_decomposed_rel_pos(
                attn,
                q,
                rel_pos_h.clone(),
                rel_pos_w.clone(),
                (h as usize, w as usize),
                (h as usize, w as usize),
            ),
            None => attn,
        };

        let attn = softmax(attn, 4 - 1);
        let attn = attn.matmul(v);

        let attn = attn
            .reshape([b, self.num_heads as i32, h, w, -1])
            .permute([0, 2, 3, 1, 4])
            .reshape([b, h, w, -1]);

        self.proj.forward(attn)
    }

    fn add_decomposed_rel_pos(
        &self,
        attn: Tensor<B, 3>,
        q: Tensor<B, 3>,
        rel_pos_h: Tensor<B, 2>,
        rel_pos_w: Tensor<B, 2>,
        (q_h, q_w): (usize, usize),
        (k_h, k_w): (usize, usize),
    ) -> Tensor<B, 3> {
        let r_h = get_rel_pos(q_h, k_h, rel_pos_h.clone());
        let r_w = get_rel_pos(q_w, k_w, rel_pos_w.clone());
        let [b, _, dim] = q.dims();

        let r_q = q.reshape([b, q_h, q_w, dim]);
        // einsum operation of "bhwc,hkc->bhwk"
        let rel_h = self.einsum_bhwc_hkc(&r_q, &r_h);
        // einsum operation of "bhwc,wkc->bhwk"
        let rel_w = self.einsum_bhwc_wkc(&r_q, &r_w);

        // rel_h[:, :, :, :, None]
        let rel_h = rel_h.unsqueeze_dim::<5>(4);
        // rel_w[:, :, :, None, :]
        let rel_w = rel_w.unsqueeze_dim::<5>(3);

        let attn = (attn.reshape([b, q_h, q_w, k_h, k_w]) + rel_h + rel_w).reshape([
            b,
            q_h * q_w,
            k_h * k_w,
        ]);

        attn
    }

    // einsum of bhwc,hkc->bhwk
    fn einsum_bhwc_hkc(&self, bhwc: &Tensor<B, 4>, hkc: &Tensor<B, 3>) -> Tensor<B, 4> {
        let batch_size = bhwc.dims()[0];
        let height = bhwc.dims()[1];
        let weight = bhwc.dims()[2];
        let kernel_size = hkc.dims()[1];

        let device = bhwc.device();

        let mut result = Tensor::<B, 4>::zeros([batch_size, height, weight, kernel_size], &device);

        // einsum "bhwc,hkc->bhwk"
        for b in 0..batch_size {
            for h in 0..height {
                for w in 0..weight {
                    for k in 0..kernel_size {
                        let tensor1 = bhwc.clone().slice([b..b + 1, h..h + 1, w..w + 1, 0..4]);
                        let tensor2 = hkc.clone().slice([h..h + 1, k..k + 1, 0..4]);
                        // squeeze tensor1 and tensor2 need to be one dimension
                        // tensor1 and tensor2 dimension is [1, 1, 1, 4] and [1, 1, 4]
                        let tensor1 = tensor1.squeeze::<3>(0).squeeze::<2>(0).squeeze::<1>(0);
                        let tensor2 = tensor2.squeeze::<2>(0).squeeze::<1>(0);
                        let tensor_sum = tensor1.mul(tensor2).sum();
                        // unsqueeze tensor_sum to [1, 1, 1, 1]
                        let tensor_sum = tensor_sum.unsqueeze::<4>();
                        result = result
                            .slice_assign([b..b + 1, h..h + 1, w..w + 1, k..k + 1], tensor_sum);
                    }
                }
            }
        }

        result
    }

    // einsum of bhwc,wkc->bhwk
    fn einsum_bhwc_wkc(&self, bhwc: &Tensor<B, 4>, wkc: &Tensor<B, 3>) -> Tensor<B, 4> {
        let batch_size = bhwc.dims()[0];
        let height = bhwc.dims()[1];
        let weight = bhwc.dims()[2];
        let kernel_size = wkc.dims()[1];

        let device = bhwc.device();

        let mut result = Tensor::<B, 4>::zeros([batch_size, height, weight, kernel_size], &device);

        // einsum "bhwc,wkc->bhwk"
        for b in 0..batch_size {
            for h in 0..height {
                for w in 0..weight {
                    for k in 0..kernel_size {
                        let tensor1 = bhwc.clone().slice([b..b + 1, h..h + 1, w..w + 1, 0..4]);
                        let tensor2 = wkc.clone().slice([w..w + 1, k..k + 1, 0..4]);
                        // squeeze tensor1 and tensor2 need to be one dimension
                        // tensor1 and tensor2 dimension is [1, 1, 1, 4] and [1, 1, 4]
                        let tensor1 = tensor1.squeeze::<3>(0).squeeze::<2>(0).squeeze::<1>(0);
                        let tensor2 = tensor2.squeeze::<2>(0).squeeze::<1>(0);
                        let tensor_sum = tensor1.mul(tensor2).sum();
                        // unsqueeze tensor_sum to [1, 1, 1, 1]
                        let tensor_sum = tensor_sum.unsqueeze::<4>();
                        result = result
                            .slice_assign([b..b + 1, h..h + 1, w..w + 1, k..k + 1], tensor_sum);
                    }
                }
            }
        }

        result
    }
}

#[derive(Config, Debug)]
struct AttentionConfig {
    dim: usize,
    #[config(default = "8")]
    num_heads: usize,
    #[config(default = "true")]
    qkv_bias: bool,
    #[config(default = "false")]
    use_rel_pos: bool,
    #[config(default = "true")]
    rel_pos_zero_init: bool,
    input_size: Option<[usize; 2]>,
}

impl AttentionConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> Attention<B> {
        let initializer = Initializer::KaimingUniform {
            gain: 1.0 / libm::sqrt(5.0),
            fan_out_only: false,
        };
        let head_dim = self.dim / self.num_heads;
        let qkv = LinearConfig::new(self.dim, 3 * self.dim)
            .with_bias(self.qkv_bias)
            .with_initializer(initializer.clone())
            .init(device);
        let proj = LinearConfig::new(self.dim, self.dim)
            .with_initializer(initializer)
            .init(device);
        let rel_pos_hw = if self.use_rel_pos {
            let size = self.input_size.unwrap_or_else(|| {
                panic!("Input size must be provide if relative positional encoding is true")
            });
            let rel_pos_h = Tensor::zeros([2 * size[0] - 1, head_dim], device);
            let rel_pos_w = Tensor::zeros([2 * size[1] - 1, head_dim], device);
            Some((rel_pos_h, rel_pos_w))
        } else {
            None
        };
        Attention {
            num_heads: self.num_heads,
            scale: 1. / (head_dim as f64).sqrt(),
            qkv,
            proj,
            rel_pos_hw,
        }
    }
}

fn get_rel_pos<B: Backend>(q_size: usize, k_size: usize, rel_pos: Tensor<B, 2>) -> Tensor<B, 3> {
    let device = rel_pos.device();
    let max_rel_dist = 2 * usize::max(q_size, k_size) - 1;
    // Interpolate rel pos if needed.
    let rel_pos_resized = if rel_pos.dims()[0] != max_rel_dist {
        // Interpolate rel pos
        // TODO: create interpolate for linear and 3D
        // interpolate(
        //     rel_pos
        //         .reshape([1, rel_pos.dims()[0] as i32, -1])
        //         .permute([0, 2, 1]),
        //     max_rel_dist,
        //     InterpolateOptions {
        //         mode: InterpolateMode::Bilinear,
        //     },
        // ).reshape(-1, max_rel_dist).permute([1, 0])
        todo!("Interpolate linear 3D")
    } else {
        rel_pos
    };

    // Scale the coords with short length if shapes for q and k are different.
    let q_coords: Tensor<B, 2, Float> = Tensor::arange(0..(q_size as i64), &device)
        .reshape([q_size, 1])
        .float();
    let k_coords: Tensor<B, 2, Float> = Tensor::arange(0..(k_size as i64), &device)
        .reshape([1, k_size])
        .float();

    let q_coords = q_coords * f64::max(1f64, k_size as f64 / q_size as f64);
    let k_coords = k_coords * f64::max(1f64, q_size as f64 / k_size as f64);

    let relative_coords = q_coords.sub(k_coords);
    let relative_coords = relative_coords
        .add_scalar((k_size - 1) as f64 * f64::max(1f64, q_size as f64 / k_size as f64));

    let [d1, d2] = rel_pos_resized.dims();
    rel_pos_resized
        .clone()
        .select(0, relative_coords.reshape([-1]).int())
        .reshape([d1 as i32, d2 as i32, -1])
}

#[derive(Module, Debug)]
struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    attn: Attention<B>,
    norm2: LayerNorm<B>,
    mlp: MLPBlock<B>,
    window_size: usize,
}

impl<B: Backend<FloatElem = f32>> Block<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let shorcut = x.clone();
        let x = self.norm1.forward(x);
        let (h, w) = (x.dims()[1], x.dims()[2]);
        let (x, pad_hw) = if self.window_size > 0 {
            window_partition(x, self.window_size)
        } else {
            (x, (0, 0))
        };
        let x = self.attn.forward(x);
        let x = if self.window_size > 0 {
            window_unpartition(x, self.window_size, pad_hw, (h as usize, w as usize))
        } else {
            x
        };

        let x = shorcut + x;
        let x = x.clone() + self.mlp.forward(self.norm2.forward(x));
        x
    }
}

fn window_partition<B: Backend<FloatElem = f32>>(
    x: Tensor<B, 4>,
    window_size: usize,
) -> (Tensor<B, 4>, (usize, usize)) {
    let [b, h, w, c] = x.dims();

    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;

    let x = if pad_h > 0 || pad_w > 0 {
        x.pad((0, 0, pad_h, pad_w), 0.0)
    } else {
        x
    };
    let (hp, wp) = (h + pad_h, w + pad_w);

    let x = x.reshape([
        b,
        hp / window_size,
        window_size,
        wp / window_size,
        window_size,
        c,
    ]);
    let windows = x.permute([0, 1, 3, 2, 4, 5]).reshape([
        -1,
        window_size as i32,
        window_size as i32,
        c as i32,
    ]);
    (windows, (hp, wp))
}

fn window_unpartition<B: Backend>(
    windows: Tensor<B, 4>,
    window_size: usize,
    (h_p, w_p): (usize, usize),
    (h, w): (usize, usize),
) -> Tensor<B, 4> {
    let b = windows.dims()[0] as i32 / (h_p * w_p / window_size / window_size) as i32;
    let x = windows.reshape([
        b as i32,
        (h_p / window_size) as i32,
        (w_p / window_size) as i32,
        window_size as i32,
        window_size as i32,
        -1,
    ]);
    let x = x
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([b, h_p as i32, w_p as i32, -1]);

    if h_p > h || w_p > w {
        // let [_, _, _, c] = x.dims();
        // x.slice([0..b as usize, 0..h, 0..w, 0..c]);
        // or we can use narrow
        x.narrow(1, 0, h).narrow(2, 0, w)
    } else {
        x
    }
}

#[derive(Config, Debug)]
struct BlockConfig {
    dim: usize,
    num_heads: usize,
    #[config(default = 4.0)]
    mlp_ratio: f64,
    #[config(default = "true")]
    qkv_bias: bool,
    #[config(default = "false")]
    use_rel_pos: bool,
    #[config(default = "true")]
    rel_pos_zero_init: bool,
    #[config(default = "0")]
    window_size: usize,
    input_size: Option<[usize; 2]>,
}

impl BlockConfig {
    fn init<B: Backend>(&self, device: &Device<B>, activation: Activation) -> Block<B> {
        let input_size_attn = if self.window_size == 0 {
            let size = self
                .input_size
                .unwrap_or_else(|| panic!("Input size must be provide if window size is 0"));
            Some([size[0], size[1]])
        } else {
            Some([self.window_size, self.window_size])
        };
        let attn = AttentionConfig::new(self.dim)
            .with_num_heads(self.num_heads)
            .with_qkv_bias(self.qkv_bias)
            .with_use_rel_pos(self.use_rel_pos)
            .with_rel_pos_zero_init(self.rel_pos_zero_init)
            .with_input_size(input_size_attn)
            .init(device);
        let mlp_dim = (self.dim as f64 * self.mlp_ratio as f64).round() as usize;
        let mlp = MLPBlockConfig::new(self.dim, mlp_dim).init(device, activation);
        Block {
            norm1: LayerNormConfig::new(self.dim).init(device),
            attn,
            norm2: LayerNormConfig::new(self.dim).init(device),
            mlp,
            window_size: self.window_size,
        }
    }
}

#[derive(Module, Debug)]
struct PatchEmbed<B: Backend> {
    proj: Conv2d<B>,
}

impl<B: Backend> PatchEmbed<B> {
    fn forward(self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.proj.forward(x);
        // B C H W -> B H W C
        output.permute([0, 2, 3, 1])
    }
}

#[derive(Config, Debug)]
struct PatchEmbedConfig {
    #[config(default = "[16, 16]")]
    kernel_size: [usize; 2],
    #[config(default = "[16, 16]")]
    stride: [usize; 2],
    #[config(default = "[0, 0]")]
    padding: [usize; 2],
    #[config(default = "3")]
    in_channels: usize,
    #[config(default = "768")]
    embed_dim: usize,
}

impl PatchEmbedConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> PatchEmbed<B> {
        let initializer = Initializer::KaimingUniform {
            gain: 1.0 / libm::sqrt(5.0),
            fan_out_only: false,
        };
        PatchEmbed {
            proj: Conv2dConfig::new([self.in_channels, self.embed_dim], self.kernel_size)
                .with_stride(self.stride)
                .with_padding(PaddingConfig2d::Explicit(self.padding[0], self.padding[1]))
                .with_initializer(initializer)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct ImageEncoderVit<B: Backend> {
    patch_embed: PatchEmbed<B>,
    pos_embed: Option<Tensor<B, 4>>,
    blocks: Vec<Block<B>>,
    neck_conv1: Conv2d<B>,
    neck_ln1: LayerNorm<B>,
    neck_conv2: Conv2d<B>,
    neck_ln2: LayerNorm<B>,
}

impl<B: Backend<FloatElem = f32>> ImageEncoderVit<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.patch_embed.clone().forward(x);
        let mut x = if self.pos_embed.is_some() {
            x + self.pos_embed.clone().unwrap()
        } else {
            x
        };

        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        let x = x.permute([0, 3, 1, 2]);
        let x = self.neck_conv1.forward(x);
        let x = self.neck_ln1.forward(x);
        let x = self.neck_conv2.forward(x);
        self.neck_ln2.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ImageEncoderVitConfig {
    global_attn_indexes: Vec<usize>,
    #[config(default = "1024")]
    img_size: usize,
    #[config(default = "16")]
    patch_size: usize,
    #[config(default = "3")]
    in_channels: usize,
    #[config(default = "256")]
    out_channels: usize,
    #[config(default = "768")]
    embed_dim: usize,
    #[config(default = "12")]
    depth: usize,
    #[config(default = "12")]
    num_heads: usize,
    #[config(default = "4.0")]
    mlp_ratio: f64,
    #[config(default = "true")]
    qkv_bias: bool,
    #[config(default = "true")]
    use_abs_pos: bool,
    #[config(default = "false")]
    use_rel_pos: bool,
    #[config(default = "true")]
    rel_post_zero_init: bool,
    #[config(default = "0")]
    window_size: usize,
}

impl ImageEncoderVitConfig {
    pub fn init<B: Backend>(
        &self,
        device: &Device<B>,
        activation: Activation,
    ) -> ImageEncoderVit<B> {
        let pos_embed = if self.use_abs_pos {
            let pos_embed = Tensor::zeros(
                [
                    1,
                    self.img_size / self.patch_size,
                    self.img_size / self.patch_size,
                    self.embed_dim,
                ],
                device,
            );
            Some(pos_embed)
        } else {
            None
        };

        let patch_embed = PatchEmbedConfig::new()
            .with_kernel_size([self.patch_size, self.patch_size])
            .with_stride([self.patch_size, self.patch_size])
            .with_in_channels(self.in_channels)
            .with_embed_dim(self.embed_dim)
            .init(device);

        let blocks = (0..self.depth)
            .map(|idx| {
                let windows_size = if self.global_attn_indexes.contains(&idx) {
                    0
                } else {
                    self.window_size
                };
                BlockConfig::new(self.embed_dim, self.num_heads)
                    .with_mlp_ratio(self.mlp_ratio)
                    .with_qkv_bias(self.qkv_bias)
                    .with_use_rel_pos(self.use_rel_pos)
                    .with_rel_pos_zero_init(self.rel_post_zero_init)
                    .with_window_size(windows_size)
                    .with_input_size(Some([
                        self.img_size / self.patch_size,
                        self.img_size / self.patch_size,
                    ]))
                    .init(device, activation)
            })
            .collect();

        let neck_conv1 = Conv2dConfig::new([self.embed_dim, self.out_channels], [1, 1])
            .with_bias(false)
            .with_initializer(Initializer::KaimingUniform {
                gain: 1.0 / libm::sqrt(5.0),
                fan_out_only: false,
            })
            .init(device);

        let neck_ln1 = LayerNormConfig::new(self.out_channels).init(device);

        let neck_conv2 = Conv2dConfig::new([self.out_channels, self.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(Initializer::KaimingUniform {
                gain: 1.0 / libm::sqrt(5.0),
                fan_out_only: false,
            })
            .init(device);

        let neck_ln2 = LayerNormConfig::new(self.out_channels).init(device);

        ImageEncoderVit {
            patch_embed,
            pos_embed,
            blocks,
            neck_conv1,
            neck_ln1,
            neck_conv2,
            neck_ln2,
        }
    }
}
