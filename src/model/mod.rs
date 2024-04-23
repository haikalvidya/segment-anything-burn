use burn::{
    module::Module,
    nn::{Initializer, Linear, LinearConfig},
    tensor::{activation, backend::Backend, Device, Tensor},
};

pub mod image_encoder;
pub mod mask_decoder;
pub mod prompt_encoder;
pub mod transformer;

#[derive(Module, Clone, Copy, Debug)]
pub enum Activation {
    Relu,
    Gelu,
}

impl Activation {
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Activation::Relu => activation::relu(input),
            Activation::Gelu => activation::gelu(input),
        }
    }
}

#[derive(Module, Debug)]
pub struct MLPBlock<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
    activation: Activation,
}

impl<B: Backend> MLPBlock<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.activation.forward(self.lin1.forward(input));
        self.lin2.forward(x)
    }
}

struct MLPBlockConfig {
    lin1: LinearConfig,
    lin2: LinearConfig,
}

impl MLPBlockConfig {
    fn new(embedding_dim: usize, mlp_dim: usize) -> Self {
        Self {
            lin1: LinearConfig::new(embedding_dim, mlp_dim).with_bias(true),
            lin2: LinearConfig::new(mlp_dim, embedding_dim).with_bias(true),
        }
    }

    fn init<B: Backend>(&self, device: &Device<B>, activation: Activation) -> MLPBlock<B> {
        let initializer = Initializer::KaimingUniform {
            gain: 1.0 / libm::sqrt(5.0),
            fan_out_only: false,
        };
        MLPBlock {
            lin1: self
                .lin1
                .clone()
                .with_initializer(initializer.clone())
                .init(device),
            lin2: self
                .lin2
                .clone()
                .with_initializer(initializer.clone())
                .init(device),
            activation,
        }
    }
}

// fn interpolate_linear_3D<B: Backend>(
//     x: &Tensor<B, 4>,
//     size: [usize; 2],
//     device: &Device<B>,
// ) -> Tensor<B, 4> {
//     let [h, w] = size;
//     let [_, c, _] = x.size();
//     let x = x.view([x.size()[0], x.size()[1], x.size()[2] * x.size()[3]]);
//     let x = x.interpolate([h, w], "bilinear", false);
//     x.view([x.size()[0], c, h, w])
// }

// #[derive(Module, Debug)]
// pub struct LayerNorm2d<B: Backend> {
//     weight: Tensor<B, 1>,
//     bias: Tensor<B, 1>,
//     eps: f64,
// }

// impl<B: Backend> LayerNorm2d<B> {
//     pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
//         let u = x.mean_dim(1);
//         let s = (x - u).powi_scalar(2).mean_dim(1);
//         let x = self.weight * x + self.bias;
//         x
//     }
// }
