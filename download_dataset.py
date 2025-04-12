from datasets import load_dataset
import json
import base64
from io import BytesIO
from PIL import Image
import os
from tqdm import tqdm

ds_name = "coco"  # change the dataset name here
print("开始读取数据")
dataset = load_dataset("MMInstruction/M3IT", ds_name, trust_remote_code=True, cache_dir="../dataset/m3it")
print("读取数据完毕")
image_dir = "../data/m3it_v0/coco"
os.makedirs(image_dir, exist_ok=True)
vlm_dict_lst = []
print("开始转换数据")
for i in tqdm(range(len(dataset["train"]))):
    # if i > 249:
    #     break
    img_base64 = dataset["train"][i]["image_base64_str"]
    image_list = []
    for j in range(len(img_base64)):
        img_bytes = base64.b64decode(img_base64[0])
        img = Image.open(BytesIO(img_bytes))
        image_name = f"{i:09d}_{j:04d}.jpg"
        image_path = f"{image_dir}/{image_name}"
        img.save(image_path)
        image_list.append(f"../{image_dir}/{image_name}")
    #
    instruction = dataset["train"][i]["instruction"]
    input = dataset["train"][i]["inputs"]
    output = dataset["train"][i]["outputs"]
    vlm_dict = {
        "messages": [
            {
                "content": f"<image>{instruction}",
                "role": "user",
            },
            {
                "content": output,
                "role": "assistant"
            }
        ],
        "images": image_list,
    }
    vlm_dict_lst.append(vlm_dict)
print("处理完毕")
print("开始保存数据")
json_path = "../data/m3it_v0/vlm1.json"
with open(json_path, "w") as f:
    json.dump(vlm_dict_lst, f)
print("保存完毕")