# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hunyuan MT is a machine translation inference project for Tencent's HY-MT1.5 models (1.8B and 7B parameters). It combines Python tools for model conversion with a Rust high-performance inference engine using Candle framework for quantized GGUF model inference.

**Supported languages:** 33 languages including Chinese, English, French, Japanese, Korean, Arabic, Russian, and more.

## Build and Run Commands

### Rust Inference Engine

```bash
# Build (from rust-infer directory)
cd rust-infer && cargo build --release

# Run inference
cd rust-infer && cargo run --release
```

### Python Scripts

```bash
# Test model with HuggingFace Transformers
python scripts/test_model.py

# Convert model to GGUF format
python scripts/convert_hunyuan_gguf.py input.gguf output.gguf

# Patch GGUF tensor names (binary-level)
python scripts/patch_gguf.py
```

## Architecture

### Two-Layer Design

1. **Python Layer** - Model loading, conversion, and reference testing using HuggingFace Transformers
2. **Rust Layer** (`rust-infer/`) - High-performance quantized inference using Candle framework

### Rust Inference Engine Structure

- `rust-infer/src/main.rs` - Entry point: device selection, model loading, tokenization, inference loop
- `rust-infer/src/hunyuan_model.rs` - Custom HunyuanModel implementation with GGUF loading

### Key Rust Components

- **HunyuanModel** - Main model struct with embeddings, transformer layers, and output projection
- **LayerWeights** - Attention (Q/K/V/O), RoPE, MLP, RMSNorm, and KV cache
- **Mlp** - Gated feed-forward network with SiLU activation

### Model Loading Flow

```
GGUF File → gguf_file::Content → HunyuanModel::from_gguf() → Forward Pass
```

### Tensor Name Mapping

Hunyuan uses non-standard tensor names. The code handles mapping between Hunyuan and Llama conventions:
- `blk.{i}` → `model.layers.{i}`
- `attn_*` → `attn.*`
- `ffn_*` → `mlp.*`

## Platform-Specific Notes

- **macOS**: Tries Metal GPU first, falls back to CPU if RmsNorm fails
- **Linux/Other**: Tries CUDA first, falls back to CPU

## Model Files

- `models/HY-MT1.5-1.8B/` - Original HuggingFace model (safetensors, tokenizer, config)
- `models/HY-MT1.5-1.8B-GGUF/` - Quantized GGUF versions (Q8_0, Q4_K_M, Q6_K)

## Inference Parameters

Recommended settings from model documentation:
```json
{
  "top_k": 20,
  "top_p": 0.6,
  "repetition_penalty": 1.05,
  "temperature": 0.7,
  "seed": random
}
```

## Prompt Templates

### Chinese ⇔ Other Languages
```
将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：

{source_text}
```

### Non-Chinese Translation
```
Translate the following segment into {target_language}, without additional explanation.

{source_text}
```

## Candle 依赖说明

本项目使用 `third_party/candle` 中的 candle 0.9.2-alpha.2 版本（未做修改），而非 crates.io 上的 0.9.1 版本。

**原因**: crates.io 的 0.9.1 版本缺少完整的 Metal SDPA (Scaled Dot-Product Attention) 支持，而 0.9.2-alpha.2 包含优化的 Metal SDPA kernel。

**Cargo.toml 配置**:
```toml
[target.'cfg(target_os = "macos")'.dependencies]
candle-core = { path = "third_party/candle/candle-core", features = ["metal"] }
candle-nn = { path = "third_party/candle/candle-nn", features = ["metal"] }  # 必须启用 metal feature

[target.'cfg(not(target_os = "macos"))'.dependencies]
candle-core = { path = "third_party/candle/candle-core", features = ["cuda"] }
candle-nn = { path = "third_party/candle/candle-nn", features = ["cuda"] }
```

**关键**: `candle-nn` 必须显式启用 `metal` feature，否则 SDPA 的 Metal 实现不会被编译。

## Hunyuan Dense 架构特点

Hunyuan Dense 与标准 Llama 架构的主要差异：

| 特性 | Llama | Hunyuan Dense |
|------|-------|---------------|
| RoPE 风格 | interleaved (`rope_i`) | contiguous/NeoX (`rope`) |
| QK Norm | 无 | 有 (RoPE 后应用 RMSNorm) |
| GQA 比例 | 通常 8:1 | 4:1 (16 Q heads, 4 KV heads) |
| rope_freq_base | 10000 | 11158840 |
| 词表大小 | 32000 | 120064 |

### QK Norm 顺序 (关键)

Hunyuan 的 QK Norm 必须在 RoPE **之后** 应用：

```rust
// 1. 先应用 RoPE
let q = self.apply_rotary_emb(&q, index_pos)?;
let k = self.apply_rotary_emb(&k, index_pos)?;
// 2. 再应用 QK Norm
let q = self.attn_q_norm.forward(&q)?;
let k = self.attn_k_norm.forward(&k)?;
```

参考: llama.cpp 的 `hunyuan-dense.cpp` 实现

## Metal 兼容性实现

由于 Candle 的部分算子缺少 Metal kernel，本项目使用基础张量操作手动实现：

### 1. RmsNorm (手动实现)

```rust
// 避免使用 candle_nn::RmsNorm，它会调用不存在的 rms-norm Metal kernel
let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
x_normed.broadcast_mul(&self.weight)
```

### 2. RoPE (手动实现, NeoX 风格)

```rust
// 避免使用 candle_nn::rotary_emb::rope，它会调用不存在的 rotary-emb Metal kernel
let half = n_embd / 2;
let xs1 = x.narrow(D::Minus1, 0, half)?;
let xs2 = x.narrow(D::Minus1, half, half)?;
let x_rotated = Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)?;
x.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?
```

### 3. Softmax (手动实现)

```rust
// 避免使用 candle_nn::ops::softmax_last_dim
let max = xs.max_keepdim(last_dim)?;
let diff = xs.broadcast_sub(&max)?;
let num = diff.exp()?;
let den = num.sum_keepdim(last_dim)?;
num.broadcast_div(&den)
```

### 4. SDPA (使用 candle_nn::ops::sdpa)

SDPA 有完整的 Metal 实现，可直接使用，但需确保：
- `candle-nn` 启用了 `metal` feature
- 输入 tensor 是 contiguous 的

```rust
candle_nn::ops::sdpa(&q.contiguous()?, &k.contiguous()?, &v.contiguous()?, None, is_causal, scale, 1.0)?
```

## 性能数据

| 平台 | 后端 | 速度 |
|------|------|------|
| macOS M 系列 | Metal GPU (SDPA) | ~31 tok/s |
| macOS M 系列 | CPU | ~14 tok/s |

## Known Limitations

1. Hunyuan MT uses a custom architecture (HunYuanDenseV1ForCausalLM), not standard Llama
2. Some Candle ops lack Metal kernels - manual implementations required (RmsNorm, RoPE, Softmax)
3. Tensor name remapping required between Hunyuan and standard GGUF formats
4. Must use candle 0.9.2-alpha.2 from third_party for Metal SDPA support
