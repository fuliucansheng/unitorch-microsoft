# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import json
import asyncio
import httpx
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


class ChatMessage(BaseModel):
    role: str
    content: str


class EntityRef(BaseModel):
    type: str
    id: str


class ChatCompletionRequest(BaseModel):
    session_id: str
    message: ChatMessage
    mode: str = "plan"
    model_id: str = ""
    provider_id: str = ""
    entities: List[EntityRef] = Field(default_factory=list)
    stream: bool = True


class NewSessionRequest(BaseModel):
    session_id: Optional[str] = None


class DeleteSessionRequest(BaseModel):
    session_id: str


class SessionInfo(BaseModel):
    id: str
    mode: str
    model: str
    name: str
    created_at: str
    updated_at: str


class ChatHistory(BaseModel):
    id: str
    mode: str
    model: str
    messages: List[ChatMessage]


@register_fastapi("microsoft/apps/studios/agent")
class StudioAgentFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/apps/studios/agent")
        router = config.getoption("router", "/microsoft/apps/studio/chat")
        self._opencode_base_url = config.getoption(
            "opencode_base_url", "http://127.0.0.1:4096"
        )
        self._self_base_url = config.getoption(
            "self_base_url", "http://127.0.0.1:8000"
        )

        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/commands", self.get_commands, methods=["GET"])
        self._router.add_api_route("/entities", self.get_entities, methods=["GET"])
        self._router.add_api_route("/models", self.get_models, methods=["GET"])
        self._router.add_api_route("/new", self.new_session, methods=["POST"])
        self._router.add_api_route("/completions", self.completions, methods=["POST"])
        self._router.add_api_route("/history", self.get_history, methods=["GET"])
        self._router.add_api_route("/sessions", self.get_sessions, methods=["GET"])
        self._router.add_api_route("/delete", self.delete_session, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()
        self._running = False

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
            ("dataset", "/microsoft/apps/studio/datasets"),
            ("job", "/microsoft/apps/studio/jobs"),
            ("label", "/microsoft/apps/studio/labels"),
            ("report", "/microsoft/apps/studio/reports"),
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
                    ModelInfo(id=model_id, name=model_name, description=model_desc)
                )
        return results

    async def new_session(self, request: NewSessionRequest):
        async with httpx.AsyncClient() as client:
            if request.session_id:
                resp = await client.post(
                    f"{self._opencode_base_url}/session/{request.session_id}/fork",
                    json={},
                    timeout=30,
                )
            else:
                resp = await client.post(
                    f"{self._opencode_base_url}/session",
                    json={},
                    timeout=30,
                )
            resp.raise_for_status()
            session = resp.json()

        return {"new_session_id": session.get("id", "")}

    async def completions(self, request: ChatCompletionRequest):
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

    async def _completions_sync(self, request: ChatCompletionRequest):
        async with httpx.AsyncClient() as client:
            model = await self._resolve_model(client, request)
            payload: dict = {
                "parts": [{"type": "text", "text": request.message.content}],
            }
            if model:
                payload["model"] = model
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
            mode=session.get("mode", "plan"),
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
                mode=s.get("mode", "plan"),
                model=s.get("model", ""),
                name=s.get("title", s.get("id", "")),
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
