import os
import textwrap
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
# 2. Stable Diffusion 3.5 Setup
# ================================
SD_MODEL = "stabilityai/stable-diffusion-3.5-large"
sd_pipe = StableDiffusion3Pipeline.from_pretrained(
    SD_MODEL,
    torch_dtype=torch.bfloat16,
)
sd_pipe.to("cuda")

# ================================
# 3. Functions
# ================================
def generate_full_story(prompt: str) -> str:
    """
    使用 Qwen 续写故事
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
    简单按字符均分为 num_chapters 章节
    """
    length = max(1, len(story) // num_chapters)
    return textwrap.wrap(story, width=length)


def generate_scene_descriptions(chapters: list[str]) -> list[str]:
    """
    为每个章节生成儿童连环画场景描述
    """
    descriptions = []
    for i, chap in enumerate(chapters, start=1):
        prompt = (
            f"请为以下章节生成一个适合儿童连环画的场景描述：\n"
            f"章节 {i} 内容：{chap}\n"
            f"场景描述："
        )
        out = qwen_pipe(
            prompt,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.7
        )
        desc = out[0]["generated_text"].strip()
        descriptions.append(desc)
    return descriptions


def generate_comic_images(descriptions: list[str], output_dir: str = "comic_images") -> None:
    """
    使用 Stable Diffusion 3.5 根据描述生成儿童连环画图像
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
# 4. 主流程
# ================================

def main():
    start = input("请输入故事开头：")
    print("正在续写故事...")
    full_story = generate_full_story(start)
    print("续写完成。\n")
    print(full_story)

    print("\n拆分为 8 章...")
    chapters = split_into_chapters(full_story)

    print("\n为每章生成场景描述...")
    scenes = generate_scene_descriptions(chapters)
    for i, s in enumerate(scenes, start=1):
        print(f"\n场景 {i}: {s}")

    print("\n开始生成图像...")
    generate_comic_images(scenes)
    print("全部图像已保存。路径：comic_images/")

if __name__ == "__main__":
    main()