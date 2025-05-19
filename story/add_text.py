import os
from PIL import Image, ImageDraw, ImageFont

def add_wrapped_text_below_image(image, text, font_path=None, font_size=36, padding=10,
                                  text_color=(0, 0, 0), bg_color=(255, 255, 255), max_width=None):
    """
    在图像底部添加自动换行的文字区域，不遮挡原图。
    """
    w, h = image.size
    max_width = max_width or w  # 如果未指定最大宽度，使用图像宽度
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    # 创建临时画布进行文本绘制和尺寸估计
    dummy_img = Image.new("RGB", (w, 100))
    draw = ImageDraw.Draw(dummy_img)

    # 自动换行逻辑
    def wrap_text(text, font, draw, max_width):
        words = text.split()
        lines = []
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_width:
                line = test_line
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    lines = wrap_text(text, font, draw, max_width)

    # 计算总高度
    line_height = draw.textbbox((0, 0), "Test", font=font)[3]
    total_text_height = len(lines) * (line_height + padding) + padding

    # 创建新图像：原图 + 文字区域
    new_height = h + total_text_height
    new_image = Image.new("RGB", (w, new_height), bg_color)
    new_image.paste(image, (0, 0))

    # 绘制文本
    draw_new = ImageDraw.Draw(new_image)
    y = h + padding
    for line in lines:
        bbox = draw_new.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (w - text_width) // 2
        draw_new.text((x, y), line, font=font, fill=text_color)
        y += line_height + padding

    return new_image

image_dir = "E:/startup/story/comic_gen"
save_dir = "E:/startup/story/comic_txt"
os.makedirs(save_dir, exist_ok=True)
images = os.listdir(image_dir)
for image_name in images:
    if not image_name.endswith(".png"):
        continue
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path).convert("RGB")
    text = "This is the test sample, and we have to test the text wrapping function."
    new_image = add_wrapped_text_below_image(image, text, font_path="robot.ttf", font_size=60, padding=10, text_color=(0, 0, 0), bg_color=(255, 255, 255))
    new_image.save(os.path.join(save_dir, f"new_{image_name}"))