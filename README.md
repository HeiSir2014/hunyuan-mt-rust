<!-- markdownlint-disable -->

# Hunyuan MT Rust Inference

使用 Candle 框架实现的 Hunyuan Dense 1.8B GGUF 推理引擎。

## 快速开始

```bash
# 克隆项目（需要带上 submodule）
git clone --recurse-submodules git@github.com:HeiSir2014/hunyuan-mt-rust.git
cd hunyuan-mt-rust

# 构建并运行
cargo run --release
```

## 模型文件

需要以下文件：
- `models/HY-MT1.5-1.8B-GGUF/HY-MT1.5-1.8B-Q8_0.gguf` - 量化模型
- `models/HY-MT1.5-1.8B/tokenizer.json` - Tokenizer

### 下载模型

```bash
# 1. 克隆项目（需要带上 submodule）
git clone --recurse-submodules git@github.com:HeiSir2014/hunyuan-mt-rust.git
cd hunyuan-mt-rust

# 2. 下载 GGUF 量化模型
git lfs install
git clone https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF models/HY-MT1.5-1.8B-GGUF

# 3. 下载 tokenizer.json（从完整模型中获取）
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

| 平台 | 后端 | 速度 |
|------|------|------|
| macOS M 系列 | Metal GPU (SDPA) | ~31 tok/s |
| macOS M 系列 | CPU | ~14 tok/s |

## 平台支持

- **macOS**: 使用 Metal GPU 加速
- **Linux/Windows**: 使用 CUDA 加速

## 实现过程中遇到的问题

在实现 Metal GPU 推理时遇到了多个问题，以下是问题总结和解决方案：

### 1. Metal 不支持专用算子

**问题**: Candle 的某些算子没有 Metal 实现，直接使用会报错：
- `no metal implementation for rms-norm`
- `no metal implementation for rotary-emb`
- `no metal implementation for softmax-last-dim`

**解决方案**: 使用基础张量操作手动实现这些算子：

```rust
// 手动实现 RmsNorm
fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
    x_normed.broadcast_mul(&self.weight)
}

// 手动实现 RoPE (NeoX/contiguous 风格)
fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
    let half = n_embd / 2;
    let xs1 = x.narrow(D::Minus1, 0, half)?;
    let xs2 = x.narrow(D::Minus1, half, half)?;
    let x_rotated = Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)?;
    x.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?
}

// 手动实现 softmax
pub fn softmax_last_dim(xs: &Tensor) -> CandleResult<Tensor> {
    let max = xs.max_keepdim(last_dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(last_dim)?;
    num.broadcast_div(&den)
}
```

### 2. SDPA Metal 支持未启用

**问题**: 使用 `candle_nn::ops::sdpa` 时报错 `no metal implementation for metal-sdpa`，即使 Metal SDPA kernel 文件存在。

**原因**: `candle-nn` 的 `metal_fwd` 函数使用 `#[cfg(feature = "metal")]` 条件编译，需要显式启用 `metal` feature。

**解决方案**: 在 `Cargo.toml` 中为 `candle-nn` 启用 metal feature：

```toml
[target.'cfg(target_os = "macos")'.dependencies]
candle-core = { path = "third_party/candle/candle-core", features = ["metal"] }
candle-nn = { path = "third_party/candle/candle-nn", features = ["metal"] }  # 关键！
```

### 3. LogitsProcessor 中的 softmax 问题

**问题**: `candle_transformers::generation::LogitsProcessor` 内部使用 `candle_nn::ops::softmax_last_dim`，在 Metal 上会失败。

**解决方案**: 自定义 `LogitsProcessor`，使用手动实现的 `softmax_last_dim`。

### 4. RoPE cos/sin 维度问题

**问题**: 预计算的 `cos/sin` 形状为 `[seq, head_dim/2]`，但 RoPE 计算需要 `[seq, head_dim]`。

**解决方案**: 拼接 cos/sin 以匹配完整的 head_dim：

```rust
let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?;  // [seq, head_dim/2] -> [seq, head_dim]
let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;
```

### 5. QK Norm 顺序问题

**问题**: QK Norm 应该在 RoPE 之后应用，而非之前。

**正确顺序**:
```rust
// 1. RoPE
let q = self.apply_rotary_emb(&q, index_pos)?;
let k = self.apply_rotary_emb(&k, index_pos)?;
// 2. QK Norm (在 RoPE 之后)
let q = self.attn_q_norm.forward(&q)?;
let k = self.attn_k_norm.forward(&k)?;
```

### 6. GQA 实现

**问题**: Hunyuan 使用 16 个 Q heads 和 4 个 KV heads (4:1 GQA)。

**解决方案**: 使用 SDPA 自动处理 GQA，无需手动扩展 K/V：

```rust
// SDPA 自动处理 GQA，只需确保 q_heads % kv_heads == 0
candle_nn::ops::sdpa(&q, &k, &v, None, is_causal, scale, 1.0)?
```

### 7. Candle 版本问题

**问题**: crates.io 上的 candle 0.9.1 没有完整的 Metal SDPA 支持。

**解决方案**: 使用 third_party 中的 candle 0.9.2-alpha.2，该版本包含完整的 Metal SDPA 实现。

## 依赖

- `candle-core` - 张量计算 (需启用 metal/cuda feature)
- `candle-nn` - 神经网络层 (需启用 metal/cuda feature 以支持 SDPA)
- `candle-transformers` - 模型组件
- `tokenizers` - HuggingFace tokenizer
