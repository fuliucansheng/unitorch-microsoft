# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import time
import base64
import requests
from PIL import Image
from pathlib import Path
from typing import Optional, List
from unitorch import get_dir

CLIENT_ID = "Iv1.b507a08c87ecfe98"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "user-agent": "GithubCopilot/1.155.0",
}
TOKEN_FILE = get_dir() + "/.github_copilot_access_token"


def cached_file(file_path: str):
    """Decorator to cache the result of a function to a file."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    return f.read()
            result = func(*args, **kwargs)
            if result is None:
                return None
            with open(file_path, "w") as f:
                f.write(result)
            return result

        return wrapper

    return decorator


@cached_file(TOKEN_FILE)
def get_access_token():
    """获取 GitHub access token"""
    print("🚀 GitHub Copilot Token 获取工具")
    print("=" * 40)

    # 1. 获取设备码
    print("📡 获取设备授权码...")
    response = requests.post(
        "https://github.com/login/device/code",
        json={"client_id": CLIENT_ID, "scope": "read:user"},
        headers=HEADERS,
    )
    auth_data = response.json()

    device_code = auth_data["device_code"]
    user_code = auth_data["user_code"]
    verification_uri = auth_data["verification_uri"]

    print(f"✅ 用户代码: {user_code}")
    print(f"🔗 验证网址: {verification_uri}")

    # 2. 打开浏览器
    input("授权完成后按回车继续...")

    # 3. 轮询获取令牌
    print("🔄 正在获取访问令牌...")
    for i in range(8):
        time.sleep(2)
        try:
            response = requests.post(
                "https://github.com/login/oauth/access_token",
                json={
                    "client_id": CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers=HEADERS,
            )

            if response.status_code == 200:
                result = response.json()
                if "access_token" in result:
                    access_token = result["access_token"]
                    print(f"🎉 获取成功！")
                    print(f"🔑 ACCESS_TOKEN: {access_token}")
                    return access_token
        except:
            pass

        print(f"   尝试 {i+1}/8...")

    print("❌ 获取失败，请重试")
    return None


def get_copilot_token():
    """获取 Copilot 专用 token"""
    url = "https://api.github.com/copilot_internal/v2/token"

    access_token = get_access_token()
    if not access_token:
        raise Exception("Failed to get GitHub access token")
    headers = {
        "Authorization": f"token {access_token}",
        "User-Agent": "GitHub-Copilot-Client/1.0",
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()["token"]
    else:
        raise Exception(
            f"Failed to get Copilot token: {response.status_code} - {response.text}"
        )


def get_response(
    prompt,
    system_prompt: Optional[
        str
    ] = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.",
    images: Optional[Image.Image] = None,
    model: Optional[str] = "gpt-5.2",
    max_tokens: Optional[int] = 16384,
):
    content = [{"type": "text", "text": prompt}]
    images = images if images is not None else []
    if isinstance(images, str):
        images = Image.open(images)
    if isinstance(images, Image.Image):
        images = [images]
    images = [
        im if isinstance(im, Image.Image) else Image.open(im)
        for im in images
        if isinstance(im, Image.Image) or isinstance(im, str)
    ]
    for i, image in enumerate(images):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
                },
            }
        )
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": content},
    ]

    headers = {
        "Authorization": f"Bearer {get_copilot_token()}",
        "User-Agent": "GitHub-Copilot-Client/1.0",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Editor-Version": "vscode/1.85.0",
        "Editor-Plugin-Version": "copilot/1.155.0",
    }
    if images is not None and len(images) > 0:
        headers["Copilot-Vision-Request"] = "true"
    try:
        response = requests.post(
            "https://api.githubcopilot.com/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        ).json()
        result = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""
    return result


def get_tools_response(
    prompt,
    tools: List[dict],
    tool_choice: Optional[str] = "auto",
    system_prompt: Optional[
        str
    ] = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.",
    images: Optional[Image.Image] = None,
    model: Optional[str] = "gpt-5.2",
    max_tokens: Optional[int] = 16384,
):
    content = [{"type": "text", "text": prompt}]
    images = images if images is not None else []
    if isinstance(images, str):
        images = Image.open(images)
    if isinstance(images, Image.Image):
        images = [images]
    images = [
        im if isinstance(im, Image.Image) else Image.open(im)
        for im in images
        if isinstance(im, Image.Image) or isinstance(im, str)
    ]
    for i, image in enumerate(images):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
                },
            }
        )
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": content},
    ]

    headers = {
        "Authorization": f"Bearer {get_copilot_token()}",
        "User-Agent": "GitHub-Copilot-Client/1.0",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Editor-Version": "vscode/1.85.0",
        "Editor-Plugin-Version": "copilot/1.155.0",
    }
    if images is not None and len(images) > 0:
        headers["Copilot-Vision-Request"] = "true"
    try:
        response = requests.post(
            "https://api.githubcopilot.com/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "tools": tools,
                "tool_choice": tool_choice,
            },
        ).json()
        result = response["choices"][0]["message"]
        tool_calls = result.get("tool_calls", [])
        content = result.get("content", None)
        content = content if content is not None else ""
        content = content.strip()
    except Exception as e:
        print(e)
        return {"content": "", "tool_calls": []}
    return {
        "content": content,
        "tool_calls": tool_calls,
    }


def get_chat_response(
    histories,
    message: str,
    system_prompt: Optional[
        str
    ] = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.",
    images: Optional[Image.Image] = None,
    model: Optional[str] = "gpt-5.2",
    max_tokens: Optional[int] = 16384,
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    for msg, ans in histories:
        messages += [
            {
                "role": "user",
                "content": msg,
            },
            {
                "role": "assistant",
                "content": ans,
            },
        ]

    content = [{"type": "text", "text": message}]
    images = images if images is not None else []
    if isinstance(images, str):
        images = Image.open(images)
    if isinstance(images, Image.Image):
        images = [images]
    images = [
        im if isinstance(im, Image.Image) else Image.open(im)
        for im in images
        if isinstance(im, Image.Image) or isinstance(im, str)
    ]
    for i, image in enumerate(images):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
                },
            }
        )

    messages += [
        {
            "role": "user",
            "content": content,
        }
    ]

    headers = {
        "Authorization": f"Bearer {get_copilot_token()}",
        "User-Agent": "GitHub-Copilot-Client/1.0",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Editor-Version": "vscode/1.85.0",
        "Editor-Plugin-Version": "copilot/1.155.0",
    }
    if images is not None and len(images) > 0:
        headers["Copilot-Vision-Request"] = "true"
    try:
        response = requests.post(
            "https://api.githubcopilot.com/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        ).json()
        result = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""

    return result
