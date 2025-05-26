import os
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = FastAPI()

# ------------------------------
# GPU Specification for DeepSeek
# ------------------------------
deepseek_gpus = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = deepseek_gpus

# ------------------------------
# DeepSeek R1 Setup
# ------------------------------
model_name = "deepseek-ai/deepseek-llm-R-1.3B-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

@app.post("/generate-story")
async def generate_story(request: Request):
    """
    续写故事接口（DeepSeek R1，多GPU）
    请求参数: { "prompt": str }
    返回: { "story": str }
    """
    data = await request.json()
    prompt = data.get("prompt", "")
    output = gen_pipe(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8
    )
    return {"story": output[0]["generated_text"]}