import os
import textwrap
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import StableDiffusionPipeline

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
# 2. Stable Diffusion 3.5 Setup
# ================================
SD_MODEL = "stabilityai/stable-diffusion-3.5-large"
sd_pipe = StableDiffusionPipeline.from_pretrained(
    SD_MODEL,
    torch_dtype=torch.bfloat16,
)
sd_pipe.to("cuda")

# ================================
# 3. Functions
# ================================
def generate_full_story(prompt: str) -> str:
    """
    Use Qwen to continue the story.
    """
    output = qwen_pipe(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8
    )
    return output[0]["generated_text"].strip()


def split_into_chapters(story: str, num_chapters: int = 8) -> list[str]:
    """
    Split the story text evenly into num_chapters parts.
    """
    length = max(1, len(story) // num_chapters)
    return textwrap.wrap(story, width=length)


def generate_scene_descriptions(chapters: list[str]) -> list[str]:
    """
    Generate concise (~10 words) scene descriptions in English for each chapter.
    """
    descriptions = []
    for i, chap in enumerate(chapters, start=1):
        prompt = (
            f"Write a concise children's comic scene description of about 10 words for the following text:\n"
            f"Chapter {i}: {chap}\n"
            f"Description:"
        )
        out = qwen_pipe(
            prompt,
            max_new_tokens=30,
            do_sample=False,
            temperature=0.7
        )
        # Take first line of output
        desc = out[0]["generated_text"].strip().split("\n")[0]
        descriptions.append(desc)
    return descriptions


def generate_comic_images(descriptions: list[str], output_dir: str = "comic_images") -> None:
    """
    Generate children's comic images using Stable Diffusion 3.5.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, desc in enumerate(descriptions, start=1):
        image = sd_pipe(
            desc,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        path = os.path.join(output_dir, f"chapter_{idx}.png")
        image.save(path)
        print(f"Saved image: {path}")

# ================================
# 4. Main Flow
# ================================
def main():
    start = input("Enter the story beginning: ")
    full_story = generate_full_story(start)
    print("\nFull story:\n", full_story)

    chapters = split_into_chapters(full_story)
    descriptions = generate_scene_descriptions(chapters)
    print("\nScene descriptions:")
    for idx, s in enumerate(descriptions, start=1):
        print(f"{idx}. {s}")

    generate_comic_images(descriptions)
    print("\nAll images saved in 'comic_images/'")

if __name__ == "__main__":
    main()