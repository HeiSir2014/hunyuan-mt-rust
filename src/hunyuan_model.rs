//! Hunyuan Dense model for quantized GGUF
//!
//! This module handles Hunyuan-specific GGUF tensor names and metadata.

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, IndexOp, Tensor, DType, D};
use candle_core::{Module, Result as CandleResult};
use candle_nn::Embedding;
use std::collections::HashMap;

const MAX_SEQ_LEN: usize = 4096;

/// Manual RmsNorm implementation compatible with Metal backend
/// Uses basic tensor operations (sqr, sum_keepdim, sqrt, broadcast_div, broadcast_mul)
/// that are supported by Metal, instead of the specialized rms_norm kernel.
#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x_dtype = x.dtype();
        // Use F32 for numerical stability
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
    }
}

/// Manual softmax implementation compatible with Metal backend
/// softmax(x) = exp(x - max) / sum(exp(x - max))
pub fn softmax_last_dim(xs: &Tensor) -> CandleResult<Tensor> {
    let last_dim = xs.dims().len() - 1;
    // Subtract max for numerical stability
    let max = xs.max_keepdim(last_dim)?;
    let diff = xs.broadcast_sub(&max)?;
    // exp(x - max)
    let num = diff.exp()?;
    // sum(exp(x - max))
    let den = num.sum_keepdim(last_dim)?;
    // exp(x - max) / sum
    num.broadcast_div(&den)
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: candle_core::quantized::QMatMul,
    feed_forward_w2: candle_core::quantized::QMatMul,
    feed_forward_w3: candle_core::quantized::QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        // Hunyuan uses SiLU activation
        let silu_w1 = candle_nn::ops::silu(&w1)?;
        self.feed_forward_w2.forward(&(silu_w1 * w3)?)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: candle_core::quantized::QMatMul,
    attention_wk: candle_core::quantized::QMatMul,
    attention_wv: candle_core::quantized::QMatMul,
    attention_wo: candle_core::quantized::QMatMul,
    attention_norm: RmsNorm,
    attn_q_norm: RmsNorm,
    attn_k_norm: RmsNorm,
    mlp: Mlp,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl LayerWeights {
    /// Manual RoPE implementation compatible with Metal backend
    /// Uses basic tensor operations instead of the specialized rotary-emb kernel.
    /// Implements the NeoX/contiguous style (not interleaved/GPT-J style).
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let (_b_sz, _n_head, seq_len, n_embd) = x.dims4()?;

        // cos/sin are [MAX_SEQ, head_dim/2], need to duplicate for full head_dim
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;

        // Duplicate cos/sin for full head dimension: [seq, head_dim/2] -> [seq, head_dim]
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?;
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;

        // Reshape for broadcasting: [seq, n_embd] -> [1, 1, seq, n_embd]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // rotate_half: split last dim in half, negate second half, swap
        let half = n_embd / 2;
        let xs1 = x.narrow(D::Minus1, 0, half)?;
        let xs2 = x.narrow(D::Minus1, half, half)?;
        let x_rotated = Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)?;

        // Apply rotation: x * cos + rotate_half(x) * sin
        x.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?
    }

    /// Repeat KV heads to match Q heads for GQA (Grouped Query Attention)
    /// K/V: [b, n_kv_head, seq, head_dim] -> [b, n_head, seq, head_dim]
    fn repeat_kv(&self, x: &Tensor) -> CandleResult<Tensor> {
        let n_rep = self.n_head / self.n_kv_head;
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        // [b, n_kv, seq, dim] -> [b, n_kv, 1, seq, dim] -> [b, n_kv, n_rep, seq, dim]
        let x = x.unsqueeze(2)?
            .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
        Ok(x)
    }

    /// Manual scaled dot-product attention for CPU backend
    /// softmax(Q @ K^T / sqrt(d)) @ V
    fn manual_attn(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, is_causal: bool) -> CandleResult<Tensor> {
        // Repeat K and V for GQA
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // Q @ K^T: [b, n_head, q_seq, head_dim] @ [b, n_head, head_dim, kv_seq] -> [b, n_head, q_seq, kv_seq]
        let attn_scores = q.contiguous()?.matmul(&k.contiguous()?.transpose(D::Minus2, D::Minus1)?)?;

        // Scale
        let attn_scores = (attn_scores * (scale as f64))?;

        // Apply causal mask if needed
        let attn_scores = if is_causal {
            let (_, _, q_len, kv_len) = attn_scores.dims4()?;
            // Create causal mask: positions where j > i should be -inf
            let mask: Vec<f32> = (0..q_len)
                .flat_map(|i| {
                    (0..kv_len).map(move |j| {
                        // For causal attention, mask out future positions
                        // j is the key position, i is the query position
                        // For prompt processing (q_len == kv_len), mask j > i
                        // This assumes q positions are at the end of kv sequence
                        let q_pos = kv_len - q_len + i;
                        if j > q_pos { f32::NEG_INFINITY } else { 0.0 }
                    })
                })
                .collect();
            let mask = Tensor::from_slice(&mask, (q_len, kv_len), attn_scores.device())?;
            attn_scores.broadcast_add(&mask)?
        } else {
            attn_scores
        };

        // Softmax over last dimension (kv_seq)
        let attn_probs = softmax_last_dim(&attn_scores)?;

        // @ V: [b, n_head, q_seq, kv_seq] @ [b, n_head, kv_seq, head_dim] -> [b, n_head, q_seq, head_dim]
        attn_probs.matmul(&v.contiguous()?)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        _mask: Option<&Tensor>,
        index_pos: usize,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        // Apply RoPE first
        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // Apply QK Norm AFTER RoPE (per-head normalization)
        let q = self.attn_q_norm.forward(&q.contiguous()?)?;
        let k = self.attn_k_norm.forward(&k.contiguous()?)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };

        self.kv_cache = Some((k.clone(), v.clone()));

        let scale = 1.0 / (self.head_dim as f64).sqrt() as f32;
        let is_causal = seq_len > 1;

        // Use optimized SDPA for GPU, manual attention for CPU
        let attn = match x.device() {
            Device::Cpu => {
                // Manual attention for CPU
                self.manual_attn(&q, &k, &v, scale, is_causal)?
            }
            _ => {
                // SDPA for GPU (Metal/CUDA)
                // SDPA handles GQA/MQA automatically when q_heads % kv_heads == 0
                candle_nn::ops::sdpa(&q.contiguous()?, &k.contiguous()?, &v.contiguous()?, None, is_causal, scale, 1.0)?
            }
        };

        let attn = attn.transpose(1, 2)?.reshape((b_sz, seq_len, n_embd))?;
        self.attention_wo.forward(&attn)
    }
}

fn precomput_freqs_cis(head_dim: usize, freq_base: f32, device: &Device) -> CandleResult<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

pub struct HunyuanModel {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: candle_core::quantized::QMatMul,
    masks: HashMap<usize, Tensor>,
}

impl HunyuanModel {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Helper to get metadata
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v.clone()),
        };

        // Extract parameters from Hunyuan metadata
        // Note: to_u32 and to_f32 return candle_core::Result, need to convert
        let head_count = match md_get("llama.attention.head_count") {
            Ok(v) => v.to_u32().context("head_count")? as usize,
            Err(e) => match md_get("hunyuan-dense.attention.head_count") {
                Ok(v) => v.to_u32().context("head_count")? as usize,
                Err(_) => return Err(e),
            },
        };

        let head_count_kv = match md_get("llama.attention.head_count_kv") {
            Ok(v) => v.to_u32().context("head_count_kv")? as usize,
            Err(_) => match md_get("hunyuan-dense.attention.head_count_kv") {
                Ok(v) => v.to_u32().context("head_count_kv")? as usize,
                Err(e) => return Err(e),
            },
        };

        let block_count = match md_get("llama.block_count") {
            Ok(v) => v.to_u32().context("block_count")? as usize,
            Err(_) => match md_get("hunyuan-dense.block_count") {
                Ok(v) => v.to_u32().context("block_count")? as usize,
                Err(e) => return Err(e),
            },
        };

        let embedding_length = match md_get("llama.embedding_length") {
            Ok(v) => v.to_u32().context("embedding_length")? as usize,
            Err(_) => match md_get("hunyuan-dense.embedding_length") {
                Ok(v) => v.to_u32().context("embedding_length")? as usize,
                Err(e) => return Err(e),
            },
        };

        let rope_dim = match md_get("llama.rope.dimension_count") {
            Ok(v) => v.to_u32().context("rope_dim")? as usize,
            Err(_) => match md_get("hunyuan-dense.rope.dimension_count") {
                Ok(v) => v.to_u32().context("rope_dim")? as usize,
                Err(_) => {
                    // For Hunyuan, rope_dim = head_dim = embedding_length / head_count
                    embedding_length / head_count
                }
            },
        };

        // Hunyuan's rms_norm_eps
        let rms_norm_eps = match md_get("llama.attention.layer_norm_rms_epsilon") {
            Ok(v) => v.to_f32().context("rms_norm_eps")? as f64,
            Err(_) => match md_get("hunyuan-dense.attention.layer_norm_rms_epsilon") {
                Ok(v) => v.to_f32().context("rms_norm_eps")? as f64,
                Err(_) => 1e-5,
            },
        };

        let rope_freq_base = match md_get("llama.rope.freq_base") {
            Ok(v) => v.to_f32().context("rope_freq_base")?,
            Err(_) => match md_get("hunyuan-dense.rope.freq_base") {
                Ok(v) => v.to_f32().context("rope_freq_base")?,
                Err(_) => 10000.0,
            },
        };

        // Precompute rotary embeddings
        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;

        // Load tensors - try both Llama and Hunyuan naming conventions
        // GGUF embedding is already [vocab_size, embed_dim], no transpose needed
        let tok_embeddings_weight = match ct.tensor(reader, "token_embd.weight", device) {
            Ok(t) => t.dequantize(device)?,
            Err(_) => ct.tensor(reader, "model.embed_tokens.weight", device)?.dequantize(device)?,
        };

        let norm = match ct.tensor(reader, "output_norm.weight", device) {
            Ok(t) => t,
            Err(_) => ct.tensor(reader, "model.norm.weight", device)?,
        };

        // For tied weights (like Hunyuan), the output layer uses the same weights as token embeddings
        // We need to reload the token embeddings for the output layer
        let output_weight = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => t,
            Err(_) => {
                // Tied weights - reload token embeddings for output
                match ct.tensor(reader, "token_embd.weight", device) {
                    Ok(t) => t,
                    Err(_) => ct.tensor(reader, "model.embed_tokens.weight", device)?,
                }
            }
        };

        let mut layers = Vec::with_capacity(block_count);

        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");

            // Try to load with Hunyuan naming first, then fall back to Llama
            let attention_wq = match ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device) {
                Ok(t) => t,
                Err(_) => {
                    let llama_prefix = format!("model.layers.{layer_idx}");
                    ct.tensor(reader, &format!("{llama_prefix}.attn.q.weight"), device)?
                }
            };

            let attention_wk = match ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device) {
                Ok(t) => t,
                Err(_) => ct.tensor(reader, &format!("model.layers.{layer_idx}.attn.k.weight"), device)?,
            };

            let attention_wv = match ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device) {
                Ok(t) => t,
                Err(_) => ct.tensor(reader, &format!("model.layers.{layer_idx}.attn.v.weight"), device)?,
            };

            let attention_wo = match ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device) {
                Ok(t) => t,
                Err(_) => ct.tensor(reader, &format!("model.layers.{layer_idx}.attn.o.weight"), device)?,
            };

            let attention_norm = match ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device) {
                Ok(t) => t,
                Err(_) => ct.tensor(reader, &format!("model.layers.{layer_idx}.attn_ln.weight"), device)?,
            };

            let feed_forward_w1 = match ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device) {
                Ok(t) => t,
                Err(_) => ct.tensor(reader, &format!("model.layers.{layer_idx}.mlp.gate.weight"), device)?,
            };

            let feed_forward_w2 = match ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device) {
                Ok(t) => t,
                Err(_) => ct.tensor(reader, &format!("model.layers.{layer_idx}.mlp.down.weight"), device)?,
            };

            let feed_forward_w3 = match ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device) {
                Ok(t) => t,
                Err(_) => ct.tensor(reader, &format!("model.layers.{layer_idx}.mlp.up.weight"), device)?,
            };

            let ffn_norm = match ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device) {
                Ok(t) => t,
                Err(_) => ct.tensor(reader, &format!("model.layers.{layer_idx}.ffn_ln.weight"), device)?,
            };

            // Load QK Norm weights
            let attn_q_norm = ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?;
            let attn_k_norm = ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?;

            layers.push(LayerWeights {
                attention_wq: candle_core::quantized::QMatMul::from_qtensor(attention_wq)?,
                attention_wk: candle_core::quantized::QMatMul::from_qtensor(attention_wk)?,
                attention_wv: candle_core::quantized::QMatMul::from_qtensor(attention_wv)?,
                attention_wo: candle_core::quantized::QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::new(attention_norm.dequantize(device)?, rms_norm_eps),
                attn_q_norm: RmsNorm::new(attn_q_norm.dequantize(device)?, rms_norm_eps),
                attn_k_norm: RmsNorm::new(attn_k_norm.dequantize(device)?, rms_norm_eps),
                mlp: Mlp {
                    feed_forward_w1: candle_core::quantized::QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: candle_core::quantized::QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: candle_core::quantized::QMatMul::from_qtensor(feed_forward_w3)?,
                },
                ffn_norm: RmsNorm::new(ffn_norm.dequantize(device)?, rms_norm_eps),
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.clone(),
                sin: sin.clone(),
                kv_cache: None,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings_weight, embedding_length),
            layers,
            norm: RmsNorm::new(norm.dequantize(device)?, rms_norm_eps),
            output: candle_core::quantized::QMatMul::from_qtensor(output_weight)?,
            masks: HashMap::new(),
        })
    }

    fn mask(&mut self, t: usize, device: &Device) -> CandleResult<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            return Ok(mask.clone());
        }
        let mask: Vec<_> = (0..t).flat_map(|i| (0..t).map(move |j| u8::from(j > i))).collect();
        let mask = Tensor::from_slice(&mask, (t, t), device)?;
        self.masks.insert(t, mask.clone());
        Ok(mask)
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 { None } else { Some(self.mask(seq_len, x.device())?) };

        let mut layer_in = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;

            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            layer_in = (x + residual)?;
        }

        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        self.output.forward(&x)
    }
}
