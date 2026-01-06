use anyhow::{Context, Result};
use candle_core::{quantized::gguf_file, Device, Tensor, DType};
use rand::{Rng, SeedableRng};
use std::io::Write;
use std::path::Path;
use tokenizers::Tokenizer;
use crate::hunyuan_model::{HunyuanModel, softmax_last_dim};

mod hunyuan_model;

/// Select the best available device at runtime
fn select_device() -> Device {
    // Try Metal on macOS
    #[cfg(target_os = "macos")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                println!("✓ Metal GPU detected");
                return device;
            }
            Err(_) => println!("✗ Metal GPU not available"),
        }
    }

    // Try CUDA on non-macOS (only if compiled with cuda feature)
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    {
        match Device::cuda_if_available(0) {
            Ok(device) if !matches!(device, Device::Cpu) => {
                println!("✓ CUDA GPU detected");
                return device;
            }
            _ => println!("✗ CUDA GPU not available"),
        }
    }

    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    {
        println!("ℹ CUDA support not compiled (use --features cuda to enable)");
    }

    println!("→ Using CPU");
    Device::Cpu
}

/// Custom LogitsProcessor compatible with Metal backend
/// Uses our manual softmax implementation instead of candle_nn::ops::softmax_last_dim
struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    temperature: Option<f64>,
    top_p: Option<f64>,
}

impl LogitsProcessor {
    fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let rng = if seed != 0 {
            let bytes = seed.to_le_bytes();
            let mut seed_array = [0u8; 32];
            seed_array[..8].copy_from_slice(&bytes);
            rand::rngs::StdRng::from_seed(seed_array)
        } else {
            rand::rngs::StdRng::from_entropy()
        };
        Self { rng, temperature, top_p }
    }

    /// Sample next token from logits
    fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits_f32 = logits.to_dtype(DType::F32)?;

        // Apply temperature
        let logits = match self.temperature {
            Some(t) if t > 0.0 => (&logits_f32 / t)?,
            _ => logits_f32,
        };

        // Get logits as Vec<f32>
        let logits_vec: Vec<f32> = logits.to_vec1()?;

        // Apply top-p filtering
        let logits_filtered = match self.top_p {
            Some(p) if p < 1.0 => self.top_p_filter(&logits_vec, p as f32),
            _ => logits_vec,
        };

        // Convert back to tensor for softmax
        let logits_tensor = Tensor::new(&*logits_filtered, logits.device())?;

        // Apply our Metal-compatible softmax
        let probs = softmax_last_dim(&logits_tensor)?;

        // Get probabilities as Vec<f32>
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // Sample from distribution
        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs_vec.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                return Ok(i as u32);
            }
        }

        // Fallback: return most probable token
        Ok(probs_vec.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0))
    }

    /// Top-p (nucleus) filtering
    fn top_p_filter(&self, logits: &[f32], p: f32) -> Vec<f32> {
        // Convert to probabilities first
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<(usize, f32)> = exp_logits.iter().enumerate().map(|(i, &x)| (i, x / sum_exp)).collect();

        // Sort by probability descending
        let mut sorted_probs = probs.clone();
        sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find cutoff
        let mut cumsum = 0.0f32;
        let mut cutoff_idx = sorted_probs.len();
        for (i, &(_, prob)) in sorted_probs.iter().enumerate() {
            cumsum += prob;
            if cumsum >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Keep only top-p tokens
        let kept: std::collections::HashSet<usize> = sorted_probs[..cutoff_idx].iter().map(|&(i, _)| i).collect();
        logits.iter().enumerate().map(|(i, &l)| {
            if kept.contains(&i) { l } else { f32::NEG_INFINITY }
        }).collect()
    }
}

fn main() -> Result<()> {
    println!("Hunyuan 1.8B GGUF Inference with Candle");
    println!("========================================\n");

    let model_path = Path::new("models/HY-MT1.5-1.8B-GGUF/HY-MT1.5-1.8B-Q8_0.gguf");
    let tokenizer_path = Path::new("models/HY-MT1.5-1.8B/tokenizer.json");

    // Device selection with runtime detection
    let device = select_device();
    println!("Using device: {:?}", device);

    // Load GGUF model
    println!("Loading model from {:?}", model_path);
    let start = std::time::Instant::now();
    let mut file = std::fs::File::open(model_path)
        .with_context(|| format!("Failed to open {:?}", model_path))?;
    let gguf = gguf_file::Content::read(&mut file)
        .context("Failed to read GGUF")?;

    let mut model = HunyuanModel::from_gguf(gguf, &mut file, &device)
        .context("Failed to load model weights")?;
    println!("Model loaded in {:.2}s\n", start.elapsed().as_secs_f32());

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
    println!("Tokenizer loaded\n");

    // Inference config - recommended settings from model documentation
    let seed = rand::thread_rng().gen();
    let temperature = Some(0.7);
    let top_p = Some(0.6);
    let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);

    // Input with proper chat template format
    // Format: <｜hy_begin▁of▁sentence｜><｜hy_User｜>{prompt}<｜hy_Assistant｜>
    let user_content = "Translate the following text into English, without additional explanation:\n\n你好，世界！欢迎使用我们的翻译服务。希望你有美好的一天！";
    let prompt = format!(
        "<｜hy_begin▁of▁sentence｜><｜hy_User｜>{}<｜hy_Assistant｜>",
        user_content
    );
    let tokens = tokenizer.encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {:?}", e))?;
    let prompt_tokens = tokens.get_ids().to_vec();

    println!("User: {}", user_content);
    println!("Input tokens: {}", prompt_tokens.len());
    println!("Generating...\n");

    let start = std::time::Instant::now();
    let eos_token_id: u32 = 120020;  // From model config.json

    // Generator-like token stream
    let token_stream = std::iter::from_fn({
        let mut all_tokens = prompt_tokens.clone();
        let mut pos = 0usize;
        let mut first = true;

        move || -> Option<Result<u32>> {
            if first {
                // First token: process full prompt
                first = false;
                let input = match Tensor::new(all_tokens.as_slice(), &device) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(anyhow::anyhow!("Failed to create input tensor: {:?}", e))),
                };
                let input = match input.unsqueeze(0) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(anyhow::anyhow!("Failed to unsqueeze input: {:?}", e))),
                };
                let logits = match model.forward(&input, pos) {
                    Ok(l) => l,
                    Err(e) => return Some(Err(anyhow::anyhow!("Forward pass failed: {:?}", e))),
                };
                let logits = match logits.squeeze(0) {
                    Ok(l) => l,
                    Err(e) => return Some(Err(anyhow::anyhow!("Failed to squeeze logits: {:?}", e))),
                };
                let token = match logits_processor.sample(&logits) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(anyhow::anyhow!("Sampling failed: {:?}", e))),
                };
                all_tokens.push(token);
                pos = 1;
                return Some(Ok(token));
            }

            // Subsequent tokens: use single token input
            let input_token = *all_tokens.last().unwrap();
            if input_token == eos_token_id {
                return None;
            }

            let input = match Tensor::new(&[input_token], &device) {
                Ok(t) => t,
                Err(e) => return Some(Err(anyhow::anyhow!("Failed to create input tensor: {:?}", e))),
            };
            let input = match input.unsqueeze(0) {
                Ok(t) => t,
                Err(e) => return Some(Err(anyhow::anyhow!("Failed to unsqueeze input: {:?}", e))),
            };
            let logits = match model.forward(&input, pos) {
                Ok(l) => l,
                Err(e) => return Some(Err(anyhow::anyhow!("Forward pass failed: {:?}", e))),
            };
            let logits = match logits.squeeze(0) {
                Ok(l) => l,
                Err(e) => return Some(Err(anyhow::anyhow!("Failed to squeeze logits: {:?}", e))),
            };
            let token = match logits_processor.sample(&logits) {
                Ok(t) => t,
                Err(e) => return Some(Err(anyhow::anyhow!("Sampling failed: {:?}", e))),
            };
            all_tokens.push(token);
            pos += 1;

            Some(Ok(token))
        }
    });

    // Consume the stream
    let mut generated_count = 0;
    for token_result in token_stream {
        match token_result {
            Ok(token) => {
                generated_count += 1;

                if token == eos_token_id {
                    break;
                }

                if let Ok(text) = tokenizer.decode(&[token], true) {
                    print!("{}", text);
                    std::io::stdout().flush()?;
                }
            }
            Err(e) => {
                eprintln!("[ERROR] Generation error: {:?}", e);
                break;
            }
        }
    }

    let elapsed = start.elapsed();
    println!("\n\nGenerated {} tokens in {:.2}s ({:.2} tok/s)",
             generated_count, elapsed.as_secs_f32(), generated_count as f32 / elapsed.as_secs_f32());

    Ok(())
}
