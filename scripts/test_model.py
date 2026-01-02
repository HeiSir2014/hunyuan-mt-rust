import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "./models/HY-MT1.5-1.8B"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
start_load = time.time()
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
load_time = time.time() - start_load

messages = [
    {"role": "user", "content": "Translate the following text into English, without additional explanation:\n\n你好，世界！欢迎使用我们的翻译服务。希望你有美好的一天！"},
]
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

start_infer = time.time()
outputs = model.generate(
    tokenized_chat.to(model.device),
    max_new_tokens=4096,
    top_k=20,
    top_p=0.6,
    repetition_penalty=1.05,
    temperature=0.7,
)
infer_time = time.time() - start_infer
output_text = tokenizer.decode(outputs[0])

print(f"模型加载时间: {load_time:.2f}s")
print(f"推理时间: {infer_time:.2f}s")
print(f"总用时: {load_time + infer_time:.2f}s")
print("\n输出:")
print(output_text)
