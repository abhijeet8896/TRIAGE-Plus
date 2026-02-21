"""
CDSS FastAPI Application
Production-grade entry point with full middleware stack.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import time, uuid, logging

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.session import engine, Base
from app.api.v1.router import api_router
from app.websocket.manager import ws_manager
from app.ml.model_loader import model_registry

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("🚀 CDSS starting up...")
    # Load ML models into memory at startup (not per-request)
    await model_registry.load_all()
    logger.info("✅ ML models loaded")
    yield
    logger.info("🛑 CDSS shutting down...")
    await model_registry.cleanup()


app = FastAPI(
    title="Hybrid Clinical Decision Support System",
    description="AI-assisted triage and escalation for PHWs and Specialist Doctors",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# ── Middleware Stack ─────────────────────────────────────────────────────────

app.add_middleware(GZipMiddleware, minimum_size=1000)  # Low-bandwidth optimization

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Inject request ID, timing, and PHI audit trail."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.start_time = time.time()

    response = await call_next(request)

    duration_ms = round((time.time() - request.state.start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms}ms"

    logger.info(
        "api_request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": request.client.host if request.client else None,
        }
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "request_id": getattr(request.state, "request_id", None),
        }
    )


# ── Routers ──────────────────────────────────────────────────────────────────

app.include_router(api_router, prefix="/api/v1")


# ── WebSocket ────────────────────────────────────────────────────────────────

from app.websocket.handlers import ws_case_endpoint

app.add_api_websocket_route("/ws/case/{case_id}", ws_case_endpoint)


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": model_registry.is_ready(),
        "version": "1.0.0",
    }
