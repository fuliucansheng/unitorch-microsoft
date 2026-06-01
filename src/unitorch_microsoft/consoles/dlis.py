# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import sys
import json
import time
import inspect
import asyncio
import traceback
import fire
import httpx
import uvicorn
import unitorch.cli
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from unitorch.cli import Config
from unitorch.cli import (
    import_library,
    registered_fastapi,
)
from unitorch_microsoft import cached_path

# ---------------------------------------------------------------------------
# Protobuf helpers — use the modern descriptor_pool API so we work with
# protobuf >= 4.x (the old generated _pb2.py uses the deprecated descriptor
# constructor API which was removed in protobuf 4/5).
# ---------------------------------------------------------------------------

_DLIS_PROTO_SERIALIZED = (
    b"\n+model_serving_client_request_response.proto"
    b"\x12\x17\x44\x65\x65pLearningModelServer"
    b"\"\xc6\x01\n\x19ModelServingClientRequest"
    b"\x12\x0f\n\x07TraceId\x18\x01 \x01(\t"
    b"\x12\x0f\n\x07IsDebug\x18\x02 \x01(\x08"
    b"\x12\x41\n\x06\x41\x63tion\x18\x03 \x01(\x0e\x32\x31"
    b".DeepLearningModelServer.ModelServingClientAction"
    b"\x12\x10\n\x08Requests\x18\x04 \x03(\t"
    b"\x12\x14\n\x0cRequestBlobs\x18\x05 \x03(\x0c"
    b"\x12\n\n\x02Id\x18\x06 \x01(\x03"
    b"\x12\x10\n\x08TraceIds\x18\x07 \x03(\t"
    b"\"\xb3\x01\n\x1aModelServingClientResponse"
    b"\x12\x45\n\x04\x43ode\x18\x01 \x01(\x0e\x32\x37"
    b".DeepLearningModelServer.ModelServingClientResponseCode"
    b"\x12\x11\n\tResponses\x18\x02 \x03(\t"
    b"\x12\x15\n\rResponseBlobs\x18\x03 \x03(\x0c"
    b"\x12\x18\n\x10ModelLatencyInUs\x18\x04 \x01(\r"
    b"\x12\n\n\x02Id\x18\x06 \x01(\x03"
    b"*7\n\x1eModelServingClientResponseCode"
    b"\x12\x0b\n\x07Success\x10\x00\x12\x08\n\x04\x46\x61il\x10\x01"
    b"*.\n\x18ModelServingClientAction"
    b"\x12\x08\n\x04Ping\x10\x00\x12\x08\n\x04\x45val\x10\x01"
    b"b\x06proto3"
)


def _make_proto_classes():
    from google.protobuf import descriptor_pb2, descriptor_pool, symbol_database
    from google.protobuf.message_factory import GetMessageClass

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.ParseFromString(_DLIS_PROTO_SERIALIZED)

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)

    req_desc = pool.FindMessageTypeByName(
        "DeepLearningModelServer.ModelServingClientRequest"
    )
    resp_desc = pool.FindMessageTypeByName(
        "DeepLearningModelServer.ModelServingClientResponse"
    )
    action_enum = pool.FindEnumTypeByName(
        "DeepLearningModelServer.ModelServingClientAction"
    )
    resp_code_enum = pool.FindEnumTypeByName(
        "DeepLearningModelServer.ModelServingClientResponseCode"
    )

    Req = GetMessageClass(req_desc)
    Resp = GetMessageClass(resp_desc)

    Ping = action_enum.values_by_name["Ping"].number
    Success = resp_code_enum.values_by_name["Success"].number
    Fail = resp_code_enum.values_by_name["Fail"].number

    return Req, Resp, Ping, Success, Fail


# Lazy-initialised once on first protobuf request
_proto_classes = None


def _get_proto_classes():
    global _proto_classes
    if _proto_classes is None:
        _proto_classes = _make_proto_classes()
    return _proto_classes


async def _dispatch_request(app: FastAPI, path: str, data) -> httpx.Response:
    """
    Internally dispatch a POST request to `path` on the FastAPI app,
    forwarding `data` (dict/list/str/bytes) as the JSON body.
    """
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        if isinstance(data, (dict, list)):
            resp = await client.post(path, json=data)
        elif isinstance(data, str):
            resp = await client.post(
                path, content=data.encode("utf-8"), headers={"content-type": "text/plain"}
            )
        elif isinstance(data, bytes):
            resp = await client.post(
                path, content=data, headers={"content-type": "application/octet-stream"}
            )
        else:
            resp = await client.post(path, json=data)
    return resp


async def _execute_protobuf(body: bytes, app: FastAPI) -> bytes:
    """
    Decode a DLIS ModelServingClientRequest, dispatch to the appropriate
    FastAPI route via path+data, and encode the response.

    Each Requests[i] is expected to be a JSON string with {"path": ..., "data": ...}.
    For Ping → returns Success with no payload.
    """
    Req, Resp, Ping, Success, Fail = _get_proto_classes()

    request = Req()
    response = Resp()

    try:
        request.ParseFromString(body)
        response.Id = request.Id

        if request.Action == Ping:
            response.Responses.append("Success")
            response.Code = Success
        else:
            t0 = time.time()

            if request.Requests:
                for req_str in request.Requests:
                    envelope = json.loads(req_str)
                    path = envelope["path"]
                    data = envelope.get("data")
                    resp = await _dispatch_request(app, path, data)
                    response.Responses.append(resp.text)
            elif request.RequestBlobs:
                for blob in request.RequestBlobs:
                    envelope = json.loads(blob.decode("utf-8"))
                    path = envelope["path"]
                    data = envelope.get("data")
                    resp = await _dispatch_request(app, path, data)
                    response.ResponseBlobs.append(resp.content)
            else:
                raise ValueError("No valid payload found in protobuf request")

            response.ModelLatencyInUs = int((time.time() - t0) * 1_000_000)
            response.Code = Success

    except Exception:
        formatted = traceback.format_exc()
        print(formatted, file=sys.stderr)
        response.Code = Fail
        response.Responses.append(f"internal server error: {formatted}")

    return response.SerializeToString()


# ---------------------------------------------------------------------------
# FastAPI app builder
# ---------------------------------------------------------------------------


def _is_protobuf(headers) -> bool:
    return headers.get("isprotobuf", "false").lower() == "true"


def _is_binary_content(headers) -> bool:
    return headers.get("content-type", "text/plain") == "application/binary"


def _build_app(fastapi_instances: dict) -> FastAPI:
    """
    Build a FastAPI app that exposes:
      - POST /           DLIS-compatible inference endpoint
      - GET  /health-check
      - all routers from registered fastapi services (same as unitorch-fastapi)

    Request body for POST /:
      - Protobuf (IsProtobuf: true header): each Requests[i] is a JSON string
        {"path": "/some/route", "data": {...}}
      - Plain JSON (default): {"path": "/some/route", "data": {...}}
      - Binary (Content-Type: application/binary): raw bytes, path taken from
        X-Path header (falls back to first registered service route)
    """
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["IsProtobuf", "Content-Type", "UnderlyingModelLatencyInUs"],
    )

    # Mount all service routers — identical to unitorch-fastapi
    for fastapi_instance in fastapi_instances.values():
        app.include_router(fastapi_instance.router)

    @app.get("/health-check")
    async def health_check():
        return {"status": "ok"}

    @app.post("/")
    async def dlis_infer(request: Request) -> Response:
        body = await request.body()
        if not body:
            return Response(content="empty request", status_code=400)

        headers = request.headers

        try:
            if _is_protobuf(headers):
                response_bytes = await _execute_protobuf(body, app)
                return Response(
                    content=response_bytes,
                    media_type="application/binary",
                    headers={"IsProtobuf": "true"},
                )
            elif _is_binary_content(headers):
                # Binary path: expect X-Path header to identify the target route
                path = headers.get("x-path", "")
                if not path:
                    return Response(
                        content="X-Path header is required for binary requests",
                        status_code=400,
                    )
                t0 = time.time()
                resp = await _dispatch_request(app, path, body)
                latency_us = int((time.time() - t0) * 1_000_000)
                return Response(
                    content=resp.content,
                    media_type=resp.headers.get("content-type", "application/binary"),
                    headers={"UnderlyingModelLatencyInUs": str(latency_us)},
                )
            else:
                # Default: JSON envelope {"path": "...", "data": {...}}
                envelope = json.loads(body)
                path = envelope["path"]
                data = envelope.get("data")
                t0 = time.time()
                resp = await _dispatch_request(app, path, data)
                latency_us = int((time.time() - t0) * 1_000_000)
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"),
                    headers={"UnderlyingModelLatencyInUs": str(latency_us)},
                )
        except Exception:
            formatted = traceback.format_exc()
            print(formatted, file=sys.stderr)
            return Response(
                content=f"internal server error: {formatted}",
                status_code=500,
            )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@fire.decorators.SetParseFn(str)
def server(config_path: str, **kwargs):
    config_path = cached_path(config_path)

    params = []
    for k, v in kwargs.items():
        if k.count("@") > 0:
            k0 = k.split("@")[0]
            k1 = "@".join(k.split("@")[1:])
        else:
            k0 = "core/cli"
            k1 = k
        params.append((k0, k1, v))

    config = Config(config_path, params=params)

    depends_libraries = config.getdefault("core/cli", "depends_libraries", None)
    if depends_libraries:
        for library in depends_libraries:
            import_library(library)

    enabled_services = config.getdefault("core/cli", "enabled_services", None)
    assert enabled_services is not None, "enabled_services must be set in [core/cli]"
    if isinstance(enabled_services, str):
        enabled_services = [enabled_services]
    enabled_services = list(enabled_services)

    for name in enabled_services:
        assert name in registered_fastapi, (
            f"fastapi service {name!r} not found in registered_fastapi"
        )

    fastapi_instances = {
        name: registered_fastapi[name]["obj"](config)
        for name in enabled_services
    }

    # autostart services
    autostart = config.getdefault("core/cli", "autostart_services", enabled_services)
    if autostart is True:
        autostart = enabled_services
    elif isinstance(autostart, str):
        autostart = [autostart]
    for name in (autostart or []):
        instance = fastapi_instances[name]
        result = instance.start()
        if inspect.isawaitable(result):
            asyncio.run(result)

    host = config.getdefault("core/cli", "host", "0.0.0.0")
    port = int(config.getdefault("core/cli", "port", 8888))

    app = _build_app(fastapi_instances)

    print(
        f"Will listen on port {port}. "
        f"To invoke manually, run: curl http://localhost:{port} --data <the post content>"
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


@fire.decorators.SetParseFn(str)
def falcon(config_path: str, **kwargs):
    pass


def cli_server():
    fire.Fire(server)


def cli_falcon():
    fire.Fire(falcon)