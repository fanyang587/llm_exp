import os
from fastapi import FastAPI, Request
from diffusers import StableDiffusionPipeline
from fastapi.responses import JSONResponse
from io import BytesIO
import base64

app = FastAPI()

# ------------------------------
# GPU Specification for SD3.5
# ------------------------------
sd3_gpus = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = sd3_gpus

# ------------------------------
# Stable Diffusion 3.5 Setup
# ------------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5",
    torch_dtype="auto",
    device_map="auto"
)
pipe.enable_attention_slicing()

@app.post("/generate-images")
async def generate_images(request: Request):
    """
    批量生成图像接口（接收文本列表）
    请求参数: { "prompts": List[str] }
    返回: { "images": List[str] } (Base64 编码 PNG)
    """
    body = await request.json()
    prompts = body.get("prompts", [])
    results = pipe(
        prompts,
        num_inference_steps=40,
        guidance_scale=6.5
    )
    images_b64 = []
    for img in results.images:
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode('utf-8')
        images_b64.append(f"data:image/png;base64,{data}")
    return JSONResponse({"images": images_b64})