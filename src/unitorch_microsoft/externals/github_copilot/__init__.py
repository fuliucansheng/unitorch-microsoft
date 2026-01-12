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


from unitorch_microsoft.externals.github_copilot.text import (
    get_response as get_gpt4_response,
    get_tools_response as get_gpt4_tools_response,
    get_chat_response as get_gpt4_chat_response,
)
