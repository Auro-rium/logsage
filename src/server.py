"""Authenticated HTTP interface for the LogSage inference runtime.

Run with LOGSAGE_API_KEY set: uvicorn src.server:app --host 127.0.0.1
Terminate TLS in a trusted reverse proxy before exposing this service.
"""
import hmac
import os
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MAX_BODY_BYTES = 65536
MAX_LOG_CHARS = 32000


class AnalyzeRequest(BaseModel):
    logs: str = Field(min_length=1, max_length=MAX_LOG_CHARS)


class BodyLimit:
    """Bound request size before JSON parsing, including chunked requests."""
    def __init__(self, app, limit=MAX_BODY_BYTES):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/v1/analyze":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.limit:
                return await JSONResponse({"detail": "Request body too large"}, status_code=413)(
                    scope, receive, send
                )
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)


def load_runtime():
    from src.inference import LogSageInference
    return LogSageInference.from_env()


def create_app(loader=load_runtime, api_key=None):
    @asynccontextmanager
    async def lifespan(app):
        key = api_key if api_key is not None else os.environ.get("LOGSAGE_API_KEY", "")
        if not key or not key.strip():
            raise RuntimeError("LOGSAGE_API_KEY must be configured")
        app.state.api_key = key
        app.state.runtime = loader()
        app.state.inference_lock = threading.Lock()
        yield
        app.state.runtime = None

    app = FastAPI(title="LogSage", lifespan=lifespan)
    app.add_middleware(BodyLimit)

    def authenticate(authorization: str = Header(default="")):
        expected = "Bearer " + app.state.api_key
        if not hmac.compare_digest(authorization.encode("utf-8"), expected.encode("utf-8")):
            raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health")
    def health():
        return {"status": "ready"}

    @app.post("/v1/analyze", dependencies=[Depends(authenticate)])
    def analyze(request: AnalyzeRequest):
        if not request.logs.strip():
            raise HTTPException(status_code=422, detail="logs must not be blank")
        from src.inference import InputError, InferenceOutputError
        try:
            with app.state.inference_lock:
                return app.state.runtime.analyze(request.logs)
        except InputError:
            raise HTTPException(status_code=422, detail="Input exceeds model limits")
        except InferenceOutputError:
            raise HTTPException(status_code=502, detail="Model returned invalid output")
        except Exception:
            # Do not expose exception details; they may include user logs.
            raise HTTPException(status_code=503, detail="Inference unavailable")

    return app


app = create_app()
