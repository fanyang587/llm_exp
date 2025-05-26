import requests
import textwrap

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


def main():
    story_prompt = (
        "One day, Bugs Bunny takes Donald Duck to Disneyland, and something unexpected happens..."
    )
    # 获取完整故事
    full_story = call_deepseek(story_prompt)
    print("Generated Story:\n", full_story)

    # 拆分为场景
    scenes = split_into_scenes(full_story)
    print(f"\nSplit into {len(scenes)} scenes.")

    # 批量生成图像
    images = call_sd3_batch(scenes)

    # 输出结果
    for idx, (scene, img_data) in enumerate(zip(scenes, images), start=1):
        print(f"\n--- Scene {idx} ---\n{scene}\n")
        print(f"Image Data URI (first 100 chars): {img_data[:100]}...")

if __name__ == "__main__":
    main()