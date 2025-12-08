"""
Nano Banana Pro (gemini-3-pro-image-preview) 图像生成工具
Google 最新的图像生成模型
"""

import os
import base64
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import google.genai as genai
from google.genai import types


def get_client():
    """获取 Gemini API 客户端"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 文件中设置 GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)


def generate_image(prompt: str, output_path: str = None) -> str:
    """
    根据文字描述生成图片

    Args:
        prompt: 图片描述（英文效果更好）
        output_path: 输出路径，默认保存到 output 目录

    Returns:
        保存的文件路径
    """
    client = get_client()

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=1.0
    )

    print(f"正在生成图片: {prompt[:50]}...")

    transient_errors = (
        "server disconnected without sending a response",
        "remote protocol error",
        "rst_stream",
        "stream reset",
        "connection reset",
        "connection aborted",
        "timeout",
    )

    image_data = None
    max_retries = 3
    for attempt in range(max_retries):
        image_data = None
        try:
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=config
            )

            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_data = part.inline_data.data

            if image_data:
                break
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
        except Exception as e:
            err_msg = str(e).lower()
            if any(key in err_msg for key in transient_errors) and attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise

    if not image_data:
        raise Exception("未能生成图片")

    # 确定输出路径
    if output_path is None:
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/image_{timestamp}.png"

    # 保存图片
    with open(output_path, 'wb') as f:
        f.write(image_data)

    print(f"图片已保存: {output_path}")
    return output_path


def edit_image(image_path: str, edit_prompt: str, output_path: str = None) -> str:
    """
    编辑现有图片

    Args:
        image_path: 输入图片路径
        edit_prompt: 编辑指令（如 "Add a rainbow in the sky"）
        output_path: 输出路径

    Returns:
        保存的文件路径
    """
    client = get_client()

    # 读取图片
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # 检测 MIME 类型
    mime_type = "image/png"
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = "image/jpeg"
    elif image_path.lower().endswith('.webp'):
        mime_type = "image/webp"

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part.from_text(text=edit_prompt)
            ]
        )
    ]

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=0.8
    )

    print(f"正在编辑图片: {edit_prompt[:50]}...")

    image_data = None
    for chunk in client.models.generate_content_stream(
        model="gemini-3-pro-image-preview",
        contents=contents,
        config=config
    ):
        if chunk.candidates:
            for part in chunk.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data

    if not image_data:
        raise Exception("未能编辑图片")

    # 确定输出路径
    if output_path is None:
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/edited_{timestamp}.png"

    with open(output_path, 'wb') as f:
        f.write(image_data)

    print(f"编辑后的图片已保存: {output_path}")
    return output_path


def generate_with_thinking(prompt: str, output_path: str = None) -> str:
    """
    使用思考模式生成高质量图片（会产生思考图像以优化最终结果）

    Args:
        prompt: 复杂的图片描述
        output_path: 输出路径

    Returns:
        保存的文件路径
    """
    client = get_client()

    # 添加思考模式提示
    thinking_prompt = f"""Think step by step about how to create the best image for this request:
{prompt}

Consider composition, lighting, colors, and details carefully before generating."""

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=thinking_prompt)]
        )
    ]

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=1.0
    )

    print(f"正在使用思考模式生成图片...")

    image_data = None
    for chunk in client.models.generate_content_stream(
        model="gemini-3-pro-image-preview",
        contents=contents,
        config=config
    ):
        if chunk.candidates:
            for part in chunk.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data

    if not image_data:
        raise Exception("未能生成图片")

    if output_path is None:
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/thinking_{timestamp}.png"

    with open(output_path, 'wb') as f:
        f.write(image_data)

    print(f"图片已保存: {output_path}")
    return output_path


# 命令行接口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nano Banana Pro 图像生成工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 生成图片命令
    gen_parser = subparsers.add_parser("generate", help="生成新图片")
    gen_parser.add_argument("prompt", help="图片描述")
    gen_parser.add_argument("-o", "--output", help="输出文件路径")

    # 编辑图片命令
    edit_parser = subparsers.add_parser("edit", help="编辑图片")
    edit_parser.add_argument("image", help="输入图片路径")
    edit_parser.add_argument("prompt", help="编辑指令")
    edit_parser.add_argument("-o", "--output", help="输出文件路径")

    args = parser.parse_args()

    if args.command == "generate":
        generate_image(args.prompt, args.output)
    elif args.command == "edit":
        edit_image(args.image, args.prompt, args.output)
    else:
        parser.print_help()
        print("\n示例:")
        print('  python main.py generate "A cute cat wearing sunglasses"')
        print('  python main.py edit input.png "Add a hat to the cat"')
