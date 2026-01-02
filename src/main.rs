use anyhow::{Context, Result};
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use rand::Rng;
use std::io::Write;
use std::path::Path;
use tokenizers::Tokenizer;
use crate::hunyuan_model::HunyuanModel;

mod hunyuan_model;

fn main() -> Result<()> {
    println!("Hunyuan 1.8B GGUF Inference with Candle");
    println!("========================================\n");

    let model_path = Path::new("models/HY-MT1.5-1.8B-GGUF/HY-MT1.5-1.8B-Q8_0.gguf");
    let tokenizer_path = Path::new("models/HY-MT1.5-1.8B/tokenizer.json");

    // Device selection - using CPU on macOS due to missing Metal RMSNorm kernel
    #[cfg(target_os = "macos")]
    let device = Device::Cpu;

    #[cfg(not(target_os = "macos"))]
    let device = Device::cuda_if_available(0).unwrap_or_else(|_| Device::Cpu);

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
    let mut all_tokens = prompt_tokens.clone();

    // Process the full prompt first
    let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits)?;
    all_tokens.push(next_token);

    if let Ok(text) = tokenizer.decode(&[next_token], true) {
        print!("{}", text);
        std::io::stdout().flush()?;
    }

    // EOS token ID for Hunyuan
    let eos_token_id = 120020;

    // Continue generation
    for i in 1..100 {
        if next_token == eos_token_id {
            break;
        }

        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens.len() + i - 1)?;
        let logits = logits.squeeze(0)?;

        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);

        if next_token == eos_token_id {
            break;
        }

        if let Ok(text) = tokenizer.decode(&[next_token], true) {
            print!("{}", text);
            std::io::stdout().flush()?;
        }
    }

    let elapsed = start.elapsed();
    let generated = all_tokens.len() - prompt_tokens.len();
    println!("\n\nGenerated {} tokens in {:.2}s ({:.2} tok/s)",
             generated, elapsed.as_secs_f32(), generated as f32 / elapsed.as_secs_f32());

    Ok(())
}
