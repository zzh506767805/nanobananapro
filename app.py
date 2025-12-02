"""
Nano Banana Pro Web 界面
支持图片生成和编辑，可配置宽高比和分辨率
"""

import os
import time
import uuid
from typing import Any

# 绕过本地代理（解决代理导致的连接问题）
os.environ['NO_PROXY'] = '*'
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

# 从脚本所在目录加载 .env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

import google.genai as genai
from google.genai import types
from PIL import Image

# 存储 Chat session 的内存表，state 中只保存 session_id，避免不可序列化对象
CHAT_SESSION_STORE: dict[str, Any] = {}
_GENAI_CLIENT: genai.Client | None = None

# 可选配置
ASPECT_RATIOS = ["自动", "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
IMAGE_SIZES = ["1K", "2K", "4K"]
MODELS = {
    "Gemini 3 Pro": "gemini-3-pro-image-preview",
    "Nano Banana": "gemini-2.5-flash-image",
}


def get_client(api_key: str = None):
    """获取 Gemini API 客户端"""
    # 优先使用传入的 api_key，其次使用环境变量
    key = api_key.strip() if api_key and api_key.strip() else os.getenv("GOOGLE_API_KEY")
    if not key:
        raise gr.Error("请输入 API Key 或在 .env 文件中设置 GOOGLE_API_KEY")
    return genai.Client(api_key=key)


def generate_image(prompt: str, aspect_ratio: str, image_size: str, model_name: str, api_key: str):
    """生成图片，返回图片路径和模型回复"""
    if not prompt.strip():
        raise gr.Error("请输入图片描述")

    client = get_client(api_key)
    model_id = MODELS.get(model_name, "gemini-3-pro-image-preview")
    is_nano_banana = (model_name == "Nano Banana")

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]

    # 配置生成参数（Nano Banana 不支持 image_size）
    image_config_args = {}
    if not is_nano_banana:
        image_config_args["image_size"] = image_size
    if aspect_ratio != "自动":
        image_config_args["aspect_ratio"] = aspect_ratio

    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(**image_config_args) if image_config_args else None
    )

    image_data = None
    texts = []
    thoughts = []  # 思考过程

    # 自动重试机制（最多3次）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            for chunk in client.models.generate_content_stream(
                model=model_id,
                contents=contents,
                config=config
            ):
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        # 检查是否为思考过程
                        is_thought = getattr(part, 'thought', False)

                        if hasattr(part, 'inline_data') and part.inline_data:
                            if not is_thought:  # 只保存最终图片
                                image_data = part.inline_data.data
                        if hasattr(part, 'text') and part.text:
                            if is_thought:
                                thoughts.append(part.text)
                            else:
                                texts.append(part.text)
            break  # 成功则退出重试
        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                if attempt < max_retries - 1:
                    time.sleep(3)
                    texts = []  # 清空重试
                    continue
                raise gr.Error("服务器繁忙，请稍后再试")
            raise gr.Error(f"API错误: {e}")

    if not image_data:
        raise gr.Error("生成失败，请重试")

    # 保存图片
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/gen_{timestamp}.png"

    with open(output_path, 'wb') as f:
        f.write(image_data)

    # 组装模型回复（包含思考过程）
    response_parts = []
    if thoughts:
        response_parts.append("**💭 思考过程:**\n" + "\n".join(thoughts))
    if texts:
        response_parts.append("\n".join(texts))

    response_text = "\n\n".join(response_parts) if response_parts else "_模型未返回文本_"

    return output_path, response_text


def edit_image(image, edit_prompt: str, aspect_ratio: str, image_size: str, model_name: str, api_key: str):
    """编辑图片，返回图片路径和模型回复"""
    if image is None:
        raise gr.Error("请上传图片")
    if not edit_prompt.strip():
        raise gr.Error("请输入编辑指令")

    client = get_client(api_key)
    model_id = MODELS.get(model_name, "gemini-3-pro-image-preview")
    is_nano_banana = (model_name == "Nano Banana")

    # 用 PIL 打开图片
    pil_image = Image.open(image)

    # 官方示例格式: [prompt, image]
    contents = [edit_prompt, pil_image]

    # 配置生成参数（Nano Banana 不支持 image_size）
    image_config_args = {}
    if not is_nano_banana:
        image_config_args["image_size"] = image_size
    if aspect_ratio != "自动":
        image_config_args["aspect_ratio"] = aspect_ratio

    config = types.GenerateContentConfig(
        response_modalities=["Text", "Image"],
        image_config=types.ImageConfig(**image_config_args) if image_config_args else None
    )

    image_data = None
    texts = []
    thoughts = []  # 思考过程

    # 自动重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 官方用非流式调用
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config
            )

            # 解析响应
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    is_thought = getattr(part, 'thought', False)

                    if part.text is not None:
                        if is_thought:
                            thoughts.append(part.text)
                        else:
                            texts.append(part.text)
                    elif part.inline_data is not None:
                        if not is_thought:  # 只保存最终图片
                            image_data = part.inline_data.data
            break
        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                raise gr.Error("服务器繁忙，请稍后再试")
            raise gr.Error(f"API错误: {e}")

    if not image_data:
        raise gr.Error("编辑失败，请重试")

    # 保存图片
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/edit_{timestamp}.png"

    with open(output_path, 'wb') as f:
        f.write(image_data)

    # 组装模型回复（包含思考过程）
    response_parts = []
    if thoughts:
        response_parts.append("**💭 思考过程:**\n" + "\n".join(thoughts))
    if texts:
        response_parts.append("\n".join(texts))

    response_text = "\n\n".join(response_parts) if response_parts else "_模型未返回文本_"

    return output_path, response_text


def _get_or_create_chat_session(session_id: str | None, config: types.GenerateContentConfig, model_id: str, api_key: str):
    client = get_client(api_key)

    if session_id and session_id in CHAT_SESSION_STORE:
        return session_id, CHAT_SESSION_STORE[session_id]

    new_session = client.chats.create(
        model=model_id,
        config=config
    )
    new_id = str(uuid.uuid4())
    CHAT_SESSION_STORE[new_id] = new_session
    return new_id, new_session


def chat_edit_image(chat_session_id, history, init_image, prompt: str, aspect_ratio: str, image_size: str, model_name: str, api_key: str):
    """多轮对话编辑图片 - 通过 session_id 引用真实 chat"""
    if not prompt.strip():
        raise gr.Error("请输入编辑指令")

    model_id = MODELS.get(model_name, "gemini-3-pro-image-preview")
    is_nano_banana = (model_name == "Nano Banana")

    # 配置生成参数（Nano Banana 不支持 image_size）
    image_config_args = {}
    if not is_nano_banana:
        image_config_args["image_size"] = image_size
    if aspect_ratio != "自动":
        image_config_args["aspect_ratio"] = aspect_ratio

    base_config = types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE'],
        image_config=types.ImageConfig(**image_config_args) if image_config_args else None
    )

    session_id, chat_session = _get_or_create_chat_session(chat_session_id, base_config, model_id, api_key)

    if history is None:
        history = []

    existing_rounds = chat_session.get_history(curated=True)
    if init_image and len(existing_rounds) == 0:
        pil_image = Image.open(init_image)
        message_content = [prompt, pil_image]
    else:
        message_content = prompt

    image_data = None
    texts = []
    thoughts = []

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = chat_session.send_message(
                message_content,
                config=base_config
            )

            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    is_thought = getattr(part, 'thought', False)

                    if part.text is not None:
                        if is_thought:
                            thoughts.append(part.text)
                        else:
                            texts.append(part.text)
                    elif part.inline_data is not None and not is_thought:
                        image_data = part.inline_data.data
            break
        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                raise gr.Error("服务器繁忙，请稍后再试")
            raise gr.Error(f"API错误: {e}")

    if not image_data:
        raise gr.Error("编辑失败，请重试")

    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/chat_{timestamp}.png"

    with open(output_path, 'wb') as f:
        f.write(image_data)

    response_parts = []
    if thoughts:
        response_parts.append("**💭 思考过程:**\n" + "\n".join(thoughts))
    if texts:
        response_parts.append("\n".join(texts))
    response_text = "\n\n".join(response_parts) if response_parts else "_模型未返回文本_"

    history.append({"role": "user", "text": prompt})
    history.append({"role": "ai", "text": response_text})

    history_md = "### 对话历史\n"
    for msg in history:
        if msg["role"] == "user":
            history_md += f"**你:** {msg['text']}\n\n"
        else:
            history_md += f"**AI:** {msg['text']}\n\n---\n\n"

    return output_path, response_text, history_md, session_id, history


def reset_chat(chat_session_id):
    """重置对话，并清理后台 session"""
    if chat_session_id:
        CHAT_SESSION_STORE.pop(chat_session_id, None)
    return None, "", "", None, []


def multi_image_generate(prompt: str, images, aspect_ratio: str, image_size: str, model_name: str, api_key: str):
    """多图参考生成"""
    if not prompt.strip():
        raise gr.Error("请输入合成描述")
    if not images or len(images) == 0:
        raise gr.Error("请上传至少一张参考图片")
    if len(images) > 14:
        raise gr.Error("最多支持14张参考图片")

    client = get_client(api_key)
    model_id = MODELS.get(model_name, "gemini-3-pro-image-preview")
    is_nano_banana = (model_name == "Nano Banana")

    # 构建内容：prompt + 所有图片
    contents = [prompt]
    for img_path in images:
        pil_img = Image.open(img_path)
        contents.append(pil_img)

    # 配置（Nano Banana 不支持 image_size）
    image_config_args = {}
    if not is_nano_banana:
        image_config_args["image_size"] = image_size
    if aspect_ratio != "自动":
        image_config_args["aspect_ratio"] = aspect_ratio

    config = types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE'],
        image_config=types.ImageConfig(**image_config_args) if image_config_args else None
    )

    image_data = None
    texts = []
    thoughts = []  # 思考过程

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config
            )

            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    is_thought = getattr(part, 'thought', False)

                    if part.text is not None:
                        if is_thought:
                            thoughts.append(part.text)
                        else:
                            texts.append(part.text)
                    elif part.inline_data is not None:
                        if not is_thought:
                            image_data = part.inline_data.data
            break
        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                raise gr.Error("服务器繁忙，请稍后再试")
            raise gr.Error(f"API错误: {e}")

    if not image_data:
        raise gr.Error("合成失败，请重试")

    # 保存
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/multi_{timestamp}.png"

    with open(output_path, 'wb') as f:
        f.write(image_data)

    # 组装回复（包含思考过程）
    response_parts = []
    if thoughts:
        response_parts.append("**💭 思考过程:**\n" + "\n".join(thoughts))
    if texts:
        response_parts.append("\n".join(texts))
    response_text = "\n\n".join(response_parts) if response_parts else "_模型未返回文本_"

    return output_path, response_text


# 浏览器缓存 JS
JS_LOAD_KEY = """
function() {
    const key = localStorage.getItem('nb_api_key') || '';
    return key;
}
"""

JS_SAVE_KEY = """
function(key) {
    if (key && key.trim()) {
        localStorage.setItem('nb_api_key', key.trim());
    }
    return key;
}
"""

# 创建界面
with gr.Blocks(title="Nano Banana Pro") as app:
    gr.Markdown("# 🍌 Nano Banana Pro")
    gr.Markdown("Google 最新图像生成模型")

    # API Key 配置
    with gr.Accordion("API Key 配置", open=False):
        api_key_input = gr.Textbox(
            label="Google API Key",
            placeholder="输入你的 API Key（会自动保存到浏览器）",
            type="password",
            elem_id="api_key_input"
        )
        gr.Markdown("*留空则使用服务器 .env 配置*")

    # 页面加载时读取缓存
    app.load(fn=None, inputs=None, outputs=api_key_input, js=JS_LOAD_KEY)
    # 输入时保存到缓存
    api_key_input.change(fn=None, inputs=api_key_input, outputs=api_key_input, js=JS_SAVE_KEY)

    with gr.Tabs():
        # 生成图片 Tab
        with gr.TabItem("✨ 生成图片"):
            with gr.Row():
                with gr.Column():
                    gen_prompt = gr.Textbox(
                        label="图片描述",
                        placeholder="例如: A cute cat wearing sunglasses on a beach",
                        lines=3
                    )
                    with gr.Row():
                        gen_aspect = gr.Dropdown(
                            choices=ASPECT_RATIOS,
                            value="自动",
                            label="宽高比"
                        )
                        gen_size = gr.Dropdown(
                            choices=IMAGE_SIZES,
                            value="2K",
                            label="分辨率"
                        )
                        gen_model = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="Gemini 3 Pro",
                            label="模型"
                        )
                    gen_btn = gr.Button("生成", variant="primary")
                with gr.Column():
                    gen_output = gr.Image(label="生成结果", type="filepath")

            gen_response = gr.Markdown(label="模型回复")

            gen_btn.click(
                fn=generate_image,
                inputs=[gen_prompt, gen_aspect, gen_size, gen_model, api_key_input],
                outputs=[gen_output, gen_response]
            )

            gr.Examples(
                examples=[
                    ["A serene Japanese garden with cherry blossoms and a koi pond"],
                    ["A futuristic cityscape at night with neon lights"],
                    ["A cozy coffee shop interior with warm lighting"],
                ],
                inputs=[gen_prompt]
            )

        # 编辑图片 Tab
        with gr.TabItem("🎨 编辑图片"):
            with gr.Row():
                with gr.Column():
                    edit_input = gr.Image(label="上传图片", type="filepath")
                    edit_prompt = gr.Textbox(
                        label="编辑指令",
                        placeholder="例如: Add a rainbow in the sky",
                        lines=2
                    )
                    with gr.Row():
                        edit_aspect = gr.Dropdown(
                            choices=ASPECT_RATIOS,
                            value="自动",
                            label="宽高比"
                        )
                        edit_size = gr.Dropdown(
                            choices=IMAGE_SIZES,
                            value="2K",
                            label="分辨率"
                        )
                        edit_model = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="Gemini 3 Pro",
                            label="模型"
                        )
                    edit_btn = gr.Button("编辑", variant="primary")
                with gr.Column():
                    edit_output = gr.Image(label="编辑结果", type="filepath")

            edit_response = gr.Markdown(label="模型回复")

            edit_btn.click(
                fn=edit_image,
                inputs=[edit_input, edit_prompt, edit_aspect, edit_size, edit_model, api_key_input],
                outputs=[edit_output, edit_response]
            )

            gr.Examples(
                examples=[
                    ["Make it look like winter with snow"],
                    ["Change the background to a beach"],
                    ["Add dramatic lighting"],
                ],
                inputs=[edit_prompt]
            )

        # 多轮对话编辑 Tab
        with gr.TabItem("💬 多轮编辑"):
            gr.Markdown("上传图片后可以持续对话迭代修改，每次修改基于上一次的结果")

            chat_session_state = gr.State(value=None)  # 保存 chat session id
            chat_history_state = gr.State(value=[])  # 保存 Markdown 历史

            with gr.Row():
                with gr.Column():
                    chat_init_image = gr.Image(label="初始图片（可选）", type="filepath")
                    chat_prompt = gr.Textbox(
                        label="编辑指令",
                        placeholder="描述你想要的修改...",
                        lines=2
                    )
                    with gr.Row():
                        chat_aspect = gr.Dropdown(
                            choices=ASPECT_RATIOS,
                            value="自动",
                            label="宽高比"
                        )
                        chat_size = gr.Dropdown(
                            choices=IMAGE_SIZES,
                            value="2K",
                            label="分辨率"
                        )
                        chat_model = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="Gemini 3 Pro",
                            label="模型"
                        )
                    with gr.Row():
                        chat_btn = gr.Button("发送", variant="primary")
                        chat_reset = gr.Button("重置对话")
                with gr.Column():
                    chat_output = gr.Image(label="当前结果", type="filepath")

            chat_response = gr.Markdown(label="模型回复")
            chat_history_display = gr.Markdown(label="对话历史", value="")

            chat_btn.click(
                fn=chat_edit_image,
                inputs=[chat_session_state, chat_history_state, chat_init_image, chat_prompt, chat_aspect, chat_size, chat_model, api_key_input],
                outputs=[chat_output, chat_response, chat_history_display, chat_session_state, chat_history_state]
            )

            chat_reset.click(
                fn=reset_chat,
                inputs=[chat_session_state],
                outputs=[chat_output, chat_response, chat_history_display, chat_session_state, chat_history_state]
            )

        # 多图参考 Tab
        with gr.TabItem("🎭 多图合成"):
            gr.Markdown("上传多张参考图片（最多6张物体图 + 5张人像），合成一张新图片")

            with gr.Row():
                with gr.Column():
                    multi_prompt = gr.Textbox(
                        label="合成描述",
                        placeholder="描述如何组合这些图片中的元素...",
                        lines=3
                    )
                    multi_images = gr.Files(
                        label="上传参考图片（最多14张）",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    with gr.Row():
                        multi_aspect = gr.Dropdown(
                            choices=ASPECT_RATIOS,
                            value="自动",
                            label="宽高比"
                        )
                        multi_size = gr.Dropdown(
                            choices=IMAGE_SIZES,
                            value="2K",
                            label="分辨率"
                        )
                        multi_model = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="Gemini 3 Pro",
                            label="模型"
                        )
                    multi_btn = gr.Button("合成", variant="primary")
                with gr.Column():
                    multi_output = gr.Image(label="合成结果", type="filepath")

            multi_response = gr.Markdown(label="模型回复")

            multi_btn.click(
                fn=multi_image_generate,
                inputs=[multi_prompt, multi_images, multi_aspect, multi_size, multi_model, api_key_input],
                outputs=[multi_output, multi_response]
            )

    gr.Markdown("---")
    gr.Markdown("💡 英文描述效果更好 | 💰 1K/2K $0.134, 4K $0.24")


if __name__ == "__main__":
    app.launch()
