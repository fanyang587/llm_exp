import os
os["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch

# 加载 tokenizer 和 model（需开启 trust_remote_code）
model_id = "Qwen/Qwen2-VL-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)

# 加载图像
image = Image.open("000001.jpg").convert("RGB")

# 构建输入
query = "Describe the image in detail."

# 使用tokenizer构造图文输入（自动封装图像）
inputs = tokenizer(query, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
inputs["images"] = [image]

# 推理
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256)

# 解码输出
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
