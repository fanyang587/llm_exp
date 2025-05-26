import requests
import textwrap
import base64
from pathlib import Path

DEESEEK_URL = "http://localhost:8000/generate-story"
SD3_URL = "http://localhost:8001/generate-images"


def call_deepseek(prompt: str) -> str:
    resp = requests.post(
        DEESEEK_URL,
        json={"prompt": prompt}
    )
    resp.raise_for_status()
    return resp.json().get("story", "")


def split_into_scenes(story: str, num_scenes: int = 8) -> list[str]:
    return textwrap.wrap(story, width=max(1, len(story) // num_scenes))


def call_sd3_batch(prompts: list[str]) -> list[str]:
    resp = requests.post(
        SD3_URL,
        json={"prompts": prompts}
    )
    resp.raise_for_status()
    return resp.json().get("images", [])


def save_images(images_b64: list[str], output_dir: str = "output_images"):
    """
    保存 Base64 编码的图片到本地文件
    """
    Path(output_dir).mkdir(exist_ok=True)
    for idx, img_data in enumerate(images_b64, start=1):
        prefix = "data:image/png;base64,"
        if img_data.startswith(prefix):
            img_str = img_data[len(prefix):]
        else:
            img_str = img_data
        img_bytes = base64.b64decode(img_str)
        file_path = Path(output_dir) / f"scene_{idx}.png"
        with open(file_path, "wb") as f:
            f.write(img_bytes)
        print(f"Saved image: {file_path}")


def main():
    story_prompt = (
        "One day, Bugs Bunny takes Donald Duck to Disneyland, and something unexpected happens..."
    )
    full_story = call_deepseek(story_prompt)
    print("Generated Story:\n", full_story)

    scenes = split_into_scenes(full_story)
    print(f"\nSplit into {len(scenes)} scenes.")

    images_b64 = call_sd3_batch(scenes)
    save_images(images_b64)

if __name__ == "__main__":
    main()