# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import asyncio
import httpx
import logging
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from unitorch.cli import (
    register_fastapi,
)
from unitorch.cli import CoreConfigureParser, GenericFastAPI


class CommandInfo(BaseModel):
    name: str
    description: str


class EntityInfo(BaseModel):
    type: str
    id: str
    name: str
    description: str = ""


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    model_id: str = ""
    provider_id: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class EntityRef(BaseModel):
    type: str
    id: str


class ChatCompletionRequest(BaseModel):
    session_id: str
    message: ChatMessage
    mode: str = "build"
    model_id: str = ""
    provider_id: str = ""
    entities: List[EntityRef] = Field(default_factory=list)
    stream: bool = True


class NewSessionRequest(BaseModel):
    session_id: Optional[str] = None


class DeleteSessionRequest(BaseModel):
    session_id: str


class RenameSessionRequest(BaseModel):
    session_id: str
    name: str


class SessionInfo(BaseModel):
    id: str
    mode: str
    model: str
    name: str
    workspace: str = ""
    created_at: str
    updated_at: str


class ChatHistory(BaseModel):
    id: str
    mode: str
    model: str
    messages: List[ChatMessage]


@register_fastapi("microsoft/apps/studios/chats")
class StudioAgentFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/apps/studios/chats")
        router = config.getoption("router", "/microsoft/apps/studios/chats")
        self._opencode_base_url = config.getoption(
            "opencode_base_url", "http://127.0.0.1:4096"
        )
        self._self_base_url = config.getoption(
            "self_base_url", "http://127.0.0.1:5000"
        )

        self._router = APIRouter(prefix=router)
        studios_folder = config.getoption("studios_folder", "studios")
        self._workspace_root = os.path.join(studios_folder, "workspaces")
        os.makedirs(self._workspace_root, exist_ok=True)
        self._router.add_api_route("/commands", self.get_commands, methods=["GET"])
        self._router.add_api_route("/entities", self.get_entities, methods=["GET"])
        self._router.add_api_route("/models", self.get_models, methods=["GET"])
        self._router.add_api_route("/new", self.new_session, methods=["POST"])
        self._router.add_api_route("/completions", self.completions, methods=["POST"])
        self._router.add_api_route("/history", self.get_history, methods=["GET"])
        self._router.add_api_route("/sessions", self.get_sessions, methods=["GET"])
        self._router.add_api_route("/delete", self.delete_session, methods=["POST"])
        self._router.add_api_route("/name", self.rename_session, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()
        self._running = False
        self._welcome_message = "Welcome to Ads Studio. How can I assist with your ML workflows today?"

    @property
    def router(self):
        return self._router

    def start(self):
        self._running = True
        return "running"

    def stop(self):
        self._running = False
        return "stopped"

    def status(self):
        return "running" if self._running else "stopped"

    async def get_commands(self):
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._opencode_base_url}/command", timeout=30,
            )
            resp.raise_for_status()
            commands = resp.json()
        return [
            CommandInfo(
                name=cmd.get("name", cmd.get("id", "")),
                description=cmd.get("description", ""),
            )
            for cmd in commands
        ]

    async def get_entities(self):
        entity_sources = [
            ("dataset", "/microsoft/apps/studios/datasets"),
            ("job", "/microsoft/apps/studios/jobs"),
            ("label", "/microsoft/apps/studios/labels"),
            ("report", "/microsoft/apps/studios/reports"),
        ]
        entities = []
        async with httpx.AsyncClient() as client:
            for entity_type, path in entity_sources:
                try:
                    resp = await client.get(
                        f"{self._self_base_url}{path}", timeout=10,
                    )
                    resp.raise_for_status()
                    for item in resp.json():
                        entities.append(
                            EntityInfo(
                                type=entity_type,
                                id=item.get("id", ""),
                                name=item.get("name", ""),
                                description=item.get("description", ""),
                            )
                        )
                except Exception:
                    continue
        return entities

    async def get_models(self):
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._opencode_base_url}/provider", timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        connected = set(data.get("connected", []))
        for provider in data.get("all", []):
            provider_id = provider.get("id", "")
            if provider_id not in connected:
                continue
            provider_name = provider.get("name", provider_id)
            for model in provider.get("models", []):
                if isinstance(model, str):
                    model_id = model
                    model_name = model
                    model_desc = f"{provider_name} - {model}"
                else:
                    model_id = model.get("id", "")
                    model_name = model.get("name", model_id)
                    model_desc = model.get("description", f"{provider_name} - {model_id}")
                results.append(
                    ModelInfo(
                        id=model_id,
                        name=model_name,
                        description=model_desc,
                        model_id=model_id,
                        provider_id=provider_id,
                    )
                )
        return results

    async def new_session(self, request: NewSessionRequest):
        # Resolve workspace path first so we can pass cwd at creation time
        # For fork we don't know the new id yet, so we patch afterwards
        is_fork = bool(request.session_id)

        async with httpx.AsyncClient() as client:
            if is_fork:
                resp = await client.post(
                    f"{self._opencode_base_url}/session/{request.session_id}/fork",
                    json={},
                    timeout=30,
                )
            else:
                # Use a temporary placeholder; real workspace resolved below
                resp = await client.post(
                    f"{self._opencode_base_url}/session",
                    json={},
                    timeout=30,
                )
            resp.raise_for_status()
            session = resp.json()

        session_id = session.get("id", "")

        # Create workspace folder
        workspace = os.path.join(self._workspace_root, session_id)
        os.makedirs(workspace, exist_ok=True)

        # Best-effort: try to set directory on opencode session.
        # opencode 1.x does not support changing directory after creation,
        # so we fall back to injecting the workspace path in every message.
        try:
            async with httpx.AsyncClient() as client:
                await client.patch(
                    f"{self._opencode_base_url}/session/{session_id}",
                    json={"directory": os.path.abspath(workspace)},
                    timeout=30,
                )
        except Exception:
            pass

        return {
            "new_session_id": session_id,
            "workspace": workspace,
            "welcome_message": self._welcome_message,
        }

    async def completions(self, request: ChatCompletionRequest):
        logging.info(f"ChatCompletionRequest received: {request}")
        if request.entities:
            request.message.content += "\nEntities:\n"
            request.message.content += "\n".join(
                f"[{entity.type} ID: {entity.id}]"
                for entity in request.entities
            )
        if request.stream:
            return StreamingResponse(
                self._completions_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        return await self._completions_sync(request)

    async def _resolve_model(self, client: httpx.AsyncClient, request: ChatCompletionRequest):
        if request.model_id and request.provider_id:
            return {"modelID": request.model_id, "providerID": request.provider_id}
        # Fall back to the model used in the most recent assistant message of this session
        try:
            resp = await client.get(
                f"{self._opencode_base_url}/session/{request.session_id}/message",
                timeout=30,
            )
            resp.raise_for_status()
            for msg in reversed(resp.json()):
                info = msg.get("info", {})
                if info.get("role") == "assistant" and info.get("modelID"):
                    return {"modelID": info["modelID"], "providerID": info.get("providerID", "")}
        except Exception:
            pass
        return None

    async def _build_payload(self, request: ChatCompletionRequest) -> dict:
        async with httpx.AsyncClient() as client:
            model = await self._resolve_model(client, request)

        text = request.message.content

        payload: dict = {
            "parts": [{"type": "text", "text": text}],
            "agent": request.mode,
        }
        if model:
            payload["model"] = model
        return payload

    async def _completions_stream(self, request: ChatCompletionRequest):
        """Proxy the opencode SSE stream to the client.

        opencode emits newline-delimited JSON objects (NDJSON) or SSE lines.
        We forward each line verbatim so the frontend receives the raw SSE events.
        """
        payload = await self._build_payload(request)
        url = f"{self._opencode_base_url}/session/{request.session_id}/message"
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                timeout=None,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    # Forward each SSE line (event: / data: / blank separator)
                    yield line + "\n"

    async def _completions_sync(self, request: ChatCompletionRequest):
        payload = await self._build_payload(request)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._opencode_base_url}/session/{request.session_id}/message",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json() if resp.content else {}

        content = ""
        for part in data.get("parts", []):
            if part.get("type") == "text":
                content += part.get("text", "")

        return {"content": content}

    async def get_history(self, session_id: str = Query(...)):
        async with httpx.AsyncClient() as client:
            session_resp = await client.get(
                f"{self._opencode_base_url}/session/{session_id}",
                timeout=30,
            )
            session_resp.raise_for_status()
            session = session_resp.json()

            msg_resp = await client.get(
                f"{self._opencode_base_url}/session/{session_id}/message",
                timeout=30,
            )
            msg_resp.raise_for_status()
            raw_messages = msg_resp.json()

        messages = []
        messages.append(ChatMessage(role="assistant", content=self._welcome_message))
        for msg in raw_messages:
            info = msg.get("info", msg)
            role = info.get("role", "user")
            parts = msg.get("parts", [])
            content = ""
            for part in parts:
                if part.get("type") == "text":
                    content += part.get("text", "")
            if content:
                messages.append(ChatMessage(role=role, content=content))

        return ChatHistory(
            id=session_id,
            mode=session.get("mode", "build"),
            model=session.get("model", ""),
            messages=messages,
        )

    async def get_sessions(self):
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._opencode_base_url}/session", timeout=30,
            )
            resp.raise_for_status()
            sessions = resp.json()

        return [
            SessionInfo(
                id=s.get("id", ""),
                mode=s.get("mode", "build"),
                model=s.get("model", ""),
                name=s.get("title", s.get("id", "")),
                workspace=os.path.join(self._workspace_root, s.get("id", "")),
                created_at=s.get("createdAt", s.get("created_at", "")),
                updated_at=s.get("updatedAt", s.get("updated_at", "")),
            )
            for s in sessions
        ]

    async def delete_session(self, request: DeleteSessionRequest):
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{self._opencode_base_url}/session/{request.session_id}",
                timeout=30,
            )
            resp.raise_for_status()
        return {"message": "Chat session deleted successfully."}

    async def rename_session(self, request: RenameSessionRequest):
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                f"{self._opencode_base_url}/session/{request.session_id}",
                json={"title": request.name},
                timeout=30,
            )
            resp.raise_for_status()
        return {"session_id": request.session_id, "name": request.name}
