<!-- markdownlint-disable -->

# Hunyuan MT Rust Inference

使用 Candle 框架实现的 Hunyuan Dense 1.8B GGUF 推理引擎。

## 快速开始

```bash
# 构建并运行
cargo run --release
```

## 模型文件

需要以下文件：
- `models/HY-MT1.5-1.8B-GGUF/HY-MT1.5-1.8B-Q8_0.gguf` - 量化模型
- `models/HY-MT1.5-1.8B/tokenizer.json` - Tokenizer

### 下载模型

```bash
# 1. 下载 GGUF 量化模型
git lfs install
git clone https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF models/HY-MT1.5-1.8B-GGUF

# 2. 下载 tokenizer.json（从完整模型中获取）
git clone https://huggingface.co/tencent/HY-MT1.5-1.8B models/HY-MT1.5-1.8B
```

或使用 huggingface-hub Python 包：

```python
from huggingface_hub import hf_hub_download

# 下载 GGUF 模型
hf_hub_download(
    repo_id="tencent/HY-MT1.5-1.8B-GGUF",
    filename="HY-MT1.5-1.8B-Q8_0.gguf",
    local_dir="models/HY-MT1.5-1.8B-GGUF"
)

# 下载 tokenizer.json
hf_hub_download(
    repo_id="tencent/HY-MT1.5-1.8B",
    filename="tokenizer.json",
    local_dir="models/HY-MT1.5-1.8B"
)
```

可用的量化版本：
| 文件 | 大小 | 精度 |
|------|------|------|
| `HY-MT1.5-1.8B-Q8_0.gguf` | 1.8GB | 最高 |
| `HY-MT1.5-1.8B-Q6_K.gguf` | 1.4GB | 中等 |
| `HY-MT1.5-1.8B-Q4_K_M.gguf` | 1.1GB | 最小 |

## 架构说明

本项目实现了自定义的 `HunyuanModel`，而非使用 candle 的 `quantized_llama`，原因是 Hunyuan Dense 有以下特殊架构：

### Hunyuan Dense vs quantized_llama 差异

| 特性 | quantized_llama | Hunyuan Dense |
|------|-------|---------------|
| RoPE 风格 | interleaved (`rope_i`) | contiguous (`rope`) |
| QK Norm | 无 | 有 (RoPE 后应用) |
| GQA 比例 | 通常 8:1 | 4:1 (16 Q heads, 4 KV heads) |
| rope_freq_base | 10000 | 11158840 |

### 关键实现

**1. QK Norm (Hunyuan 特有)**

Hunyuan 在 RoPE 之后对 Q 和 K 应用 RMSNorm：

```rust
// RoPE first
let q = self.apply_rotary_emb(&q, index_pos)?;
let k = self.apply_rotary_emb(&k, index_pos)?;

// QK Norm after RoPE
let q = self.attn_q_norm.forward(&q.contiguous()?)?;
let k = self.attn_k_norm.forward(&k.contiguous()?)?;
```

**2. RoPE 风格**

```rust
// Hunyuan 使用 contiguous/NeoX 风格
candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)

// 而非 quantized_llama 的 interleaved 风格
// candle_nn::rotary_emb::rope_i(...)
```

**3. GQA (Grouped Query Attention)**

```rust
// 16 Q heads, 4 KV heads -> 重复 K/V 4 次
let n_rep = self.n_head / self.n_kv_head;  // 16 / 4 = 4
let k = k.unsqueeze(2)?
    .expand((b, n_kv, n_rep, s, d))?
    .reshape((b, n_kv * n_rep, s, d))?;
```

**4. Causal Mask**

```rust
// 使用算术运算 (candle 的 where_cond 不支持 F32)
let neg_inf = Tensor::new(-1e9f32, device)?;
let masked = mask.broadcast_mul(&neg_inf)?;
(scores + masked)?
```

## Chat Template

```
<｜hy_begin▁of▁sentence｜><｜hy_User｜>{content}<｜hy_Assistant｜>
```

EOS Token ID: `120020`

## 推理参数

推荐设置（来自模型文档）：
- `temperature`: 0.7
- `top_p`: 0.6

## 代码结构

```
src/
├── main.rs           # 入口，推理循环
└── hunyuan_model.rs  # Hunyuan Dense 模型实现
    ├── Mlp           # FFN 层 (SwiGLU)
    ├── LayerWeights  # 单层权重 + 注意力计算
    └── HunyuanModel  # 完整模型
        ├── from_gguf()  # 加载 GGUF 权重
        └── forward()    # 前向传播
```

## 性能

macOS M 系列芯片 CPU 推理：~14 tok/s

## 平台支持

- **macOS**: 优先使用 Metal GPU，若失败则回退到 CPU
- **Linux/Windows**: 优先使用 CUDA，若失败则回退到 CPU

> 注意：Metal 后端在处理量化模型的 RmsNorm 时可能存在问题。

## 依赖

- `candle-core` - 张量计算
- `candle-nn` - 神经网络层 (RmsNorm, Embedding, RoPE)
- `candle-transformers` - LogitsProcessor
- `tokenizers` - HuggingFace tokenizer
