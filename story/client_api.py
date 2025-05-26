import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import StableDiffusion3Pipeline

# ================================
# 1. Qwen 2.5-7B Instruct Setup
# ================================
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    torch_dtype="auto",
    device_map="auto"
)
qwen_pipe = pipeline(
    "text-generation",
    model=qwen_model,
    tokenizer=qwen_tokenizer,
    device_map="auto"
)

# ================================
# 2. Stable Diffusion 3.5 Large Setup
# ================================
SD_MODEL = "stabilityai/stable-diffusion-3.5-large"
sd_pipe = StableDiffusion3Pipeline.from_pretrained(
    SD_MODEL,
    torch_dtype=torch.bfloat16,
)
sd_pipe.to("cuda")

# ================================
# 3. Combined Generation Function
# ================================
def generate_chapters_and_descriptions(beginning: str, num: int = 8):
    """
    Use Qwen to continue the story, split into `num` chapters,
    and produce a concise (~10-word) description for each.
    Returns a list of dicts: [{'chapter': str, 'description': str}, ...]
    """
    prompt = (
        f"Continue this story and split into {num} chapters. "
        f"Then for each chapter, write a concise (~10-word) scene description. "
        f"Output a JSON list of objects with 'chapter' and 'description'.\n"
        f"Story: {beginning}\nOutput:"
    )
    result = qwen_pipe(
        prompt,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.8
    )
    text = result[0]["generated_text"].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find('[')
        end = text.rfind(']') + 1
        data = json.loads(text[start:end])
    return data

# ================================
# 4. Image Generation
# ================================
def generate_comic_images(entries, output_dir: str = "comic_images"):
    """
    Generate and save images for each entry using SD3.5.
    entries: list of {'chapter', 'description'}
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, item in enumerate(entries, start=1):
        desc = item['description']
        image = sd_pipe(
            desc,
            num_inference_steps=28,
            guidance_scale=7.5
        ).images[0]
        path = os.path.join(output_dir, f"chapter_{i}.png")
        image.save(path)
        print(f"Saved image: {path} - {desc}")

# ================================
# 5. Main Flow
# ================================
if __name__ == "__main__":
    start = input("Enter the story beginning: ")
    entries = generate_chapters_and_descriptions(start)
    print("\nGenerated chapters and descriptions:")
    for i, e in enumerate(entries, start=1):
        print(f"{i}. {e['chapter']}\n   -> {e['description']}")
    print("\nGenerating images...")
    generate_comic_images(entries)
    print("\nAll images saved in 'comic_images/'")