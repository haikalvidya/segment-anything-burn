use burn::{
    config::Config,
    module::Module,
    nn::{Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Device, Tensor},
};

use super::{Activation, MLPBlock, MLPBlockConfig};

#[derive(Module, Debug)]
struct Attention<B: Backend> {
    num_heads: usize,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
}

impl<B: Backend> Attention<B> {
    fn separate_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, n, c] = x.dims();
        let x = x.reshape([b, n, self.num_heads, c / self.num_heads]);
        x.swap_dims(1, 2) // b x n_heads x n x chan_per_head
    }

    fn recombine_head(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, n_heads, n_tokens, chan_per_head] = x.dims();
        let x = x.swap_dims(1, 2); // b x n x n_heads x chan_per_head
        x.reshape([b, n_tokens, n_heads * chan_per_head]) // b x n x c
    }

    fn forward(&self, q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.q_proj.forward(q);
        let k = self.k_proj.forward(k);
        let v = self.v_proj.forward(v);

        // separate into heads
        let q = self.separate_heads(q);
        let k = self.separate_heads(k);
        let v = self.separate_heads(v);

        let [_, _, _, chan_per_head] = q.dims();
        let attn = q.matmul(k.swap_dims(4 - 1, 4 - 2));
        let attn = attn.div_scalar(chan_per_head as i32);
        let attn = softmax(attn, 4 - 1);

        // output
        let out = attn.matmul(v);
        let out = self.recombine_head(out);
        self.out_proj.forward(out)
    }
}

#[derive(Config, Debug)]
struct AttentionConfig {
    embedding_dim: usize,
    num_heads: usize,
    #[config(default = "1")]
    downsample_rate: usize,
}

impl AttentionConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> Attention<B> {
        let internal_dim = self.embedding_dim / self.downsample_rate;
        if internal_dim % self.num_heads == 0 {
            panic!("Internal dimension must be divisible by the number of heads");
        }
        let initializer = Initializer::KaimingUniform {
            gain: 1.0 / libm::sqrt(5.0),
            fan_out_only: false,
        };

        let q_proj = LinearConfig::new(self.embedding_dim, internal_dim)
            .with_initializer(initializer.clone())
            .init(device);
        let k_proj = LinearConfig::new(self.embedding_dim, internal_dim)
            .with_initializer(initializer.clone())
            .init(device);
        let v_proj = LinearConfig::new(self.embedding_dim, internal_dim)
            .with_initializer(initializer.clone())
            .init(device);
        let out_proj = LinearConfig::new(internal_dim, self.embedding_dim)
            .with_initializer(initializer)
            .init(device);

        Attention {
            num_heads: self.num_heads,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        }
    }
}

#[derive(Module, Debug)]
struct TwoWayAttentionBlock<B: Backend> {
    self_attention: Attention<B>,
    norm1: LayerNorm<B>,
    cross_attn_token_to_image: Attention<B>,
    norm2: LayerNorm<B>,
    mlp: MLPBlock<B>,
    norm3: LayerNorm<B>,
    norm4: LayerNorm<B>,
    cross_attn_image_to_token: Attention<B>,
    skip_first_layer_pe: bool,
}

impl<B: Backend> TwoWayAttentionBlock<B> {
    fn forward(
        &self,
        queries: Tensor<B, 3>,
        keys: Tensor<B, 3>,
        query_pe: Tensor<B, 3>,
        key_pe: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let queries = if self.skip_first_layer_pe {
            self.self_attention
                .forward(queries.clone(), queries.clone(), queries)
        } else {
            let q = queries.clone() + query_pe.clone();
            let attn_out = self.self_attention.forward(q.clone(), q, queries.clone());
            queries + attn_out
        };
        let queries = self.norm1.forward(queries);

        // cross attention block, tokens attending to image
        let q = queries.clone() + query_pe.clone();
        let k = keys.clone() + key_pe.clone();
        let attn_out = self.cross_attn_token_to_image.forward(q, k, keys.clone());
        let queries = queries + attn_out;
        let queries = self.norm2.forward(queries);

        // MLP Block
        let mlp_out = self.mlp.forward(queries.clone());
        let queries = queries + mlp_out;
        let queries = self.norm3.forward(queries);

        // corss attention block, image attending to tokens
        let q = queries.clone() + query_pe;
        let k = keys.clone() + key_pe;
        let attn_out = self
            .cross_attn_image_to_token
            .forward(k, q, queries.clone());
        let keys = keys + attn_out;
        let keys = self.norm4.forward(keys);

        (queries, keys)
    }
}

#[derive(Config, Debug)]
struct TwoWayAttentionBlockConfig {
    embedding_dim: usize,
    num_heads: usize,
    #[config(default = "2048")]
    mlp_dim: usize,
    #[config(default = "2")]
    attention_downsample_rate: usize,
    #[config(default = "false")]
    skip_first_layer_pe: bool,
}

impl TwoWayAttentionBlockConfig {
    fn init<B: Backend>(
        &self,
        device: &Device<B>,
        activation: Activation,
    ) -> TwoWayAttentionBlock<B> {
        let self_attention = AttentionConfig::new(self.embedding_dim, self.num_heads).init(device);

        let norm1 = LayerNormConfig::new(self.embedding_dim).init(device);

        let cross_attn_token_to_image = AttentionConfig::new(self.embedding_dim, self.num_heads)
            .with_downsample_rate(self.attention_downsample_rate)
            .init(device);
        let norm2 = LayerNormConfig::new(self.embedding_dim).init(device);

        let mlp = MLPBlockConfig::new(self.embedding_dim, self.mlp_dim).init(device, activation);
        let norm3 = LayerNormConfig::new(self.embedding_dim).init(device);

        let norm4 = LayerNormConfig::new(self.embedding_dim).init(device);
        let cross_attn_image_to_token = AttentionConfig::new(self.embedding_dim, self.num_heads)
            .with_downsample_rate(self.attention_downsample_rate)
            .init(device);

        TwoWayAttentionBlock {
            self_attention,
            norm1,
            cross_attn_token_to_image,
            norm2,
            mlp,
            norm3,
            norm4,
            cross_attn_image_to_token,
            skip_first_layer_pe: self.skip_first_layer_pe,
        }
    }
}

#[derive(Module, Debug)]
pub struct TwoWayTransformer<B: Backend> {
    layers: Vec<TwoWayAttentionBlock<B>>,
    final_attn_token_to_image: Attention<B>,
    norm_final_attention: LayerNorm<B>,
}

impl<B: Backend> TwoWayTransformer<B> {
    pub fn forward(
        &self,
        image_embeddings: Tensor<B, 4>,
        image_pe: Tensor<B, 4>,
        point_embeddings: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let image_embeddings = image_embeddings.flatten(2, 3).permute([0, 2, 1]);
        let image_pe = image_pe.flatten(2, 3).permute([0, 2, 1]);

        let queries = point_embeddings.clone();
        let keys = image_embeddings;

        let (queries, keys) = self
            .layers
            .iter()
            .fold((queries, keys), |(queries, keys), layer| {
                layer.forward(queries, keys, point_embeddings.clone(), image_pe.clone())
            });

        let q = queries.clone() + point_embeddings;
        let k = keys.clone() + image_pe;
        let attn_out = self.final_attn_token_to_image.forward(q, k, keys.clone());
        let queries = queries + attn_out;
        let queries = self.norm_final_attention.forward(queries);

        (queries, keys)
    }
}

#[derive(Config, Debug)]
pub struct TwoWayTransformerConfig {
    depth: usize,
    embedding_dim: usize,
    num_heads: usize,
    mlp_dim: usize,
    #[config(default = "2")]
    attention_downsample_rate: usize,
}

impl TwoWayTransformerConfig {
    pub fn init<B: Backend>(
        &self,
        device: &Device<B>,
        activation: Activation,
    ) -> TwoWayTransformer<B> {
        let layers = (0..self.depth)
            .map(|idx| {
                TwoWayAttentionBlockConfig::new(self.embedding_dim, self.num_heads)
                    .with_mlp_dim(self.mlp_dim)
                    .with_attention_downsample_rate(self.attention_downsample_rate)
                    .with_skip_first_layer_pe(idx == 0)
                    .init(device, activation)
            })
            .collect();

        let final_attn_token_to_image = AttentionConfig::new(self.embedding_dim, self.num_heads)
            .with_downsample_rate(self.attention_downsample_rate)
            .init(device);
        let norm_final_attention = LayerNormConfig::new(self.embedding_dim).init(device);

        TwoWayTransformer {
            layers,
            final_attn_token_to_image,
            norm_final_attention,
        }
    }
}
