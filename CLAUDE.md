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

- **macOS**: Uses Metal GPU acceleration (feature flag in Cargo.toml), but currently falls back to CPU due to missing RMSNorm kernel
- **Linux/Other**: Uses CUDA when available

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

## Known Limitations

1. Hunyuan MT uses a custom architecture (HunYuanDenseV1ForCausalLM), not standard Llama
2. Candle's Metal backend lacks RMSNorm kernel - CPU fallback used on macOS
3. Tensor name remapping required between Hunyuan and standard GGUF formats
