"""FastAPI main application."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
import uvicorn

from .datasets import router as datasets_router
from .jobs import router as jobs_router
from .artifacts import router as artifacts_router
from .inference import router as inference_router
from .models_api import router as models_api_router
from .colab import router as colab_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    print("🚀 Starting ModelOps API (Lightweight Edition)...")
    yield
    # Shutdown
    print("👋 Shutting down ModelOps API...")


app = FastAPI(
    title="ModelOps Platform API",
    description="Production-grade MLOps platform for LLM fine-tuning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Routers
app.include_router(
    datasets_router,
    prefix="/api/v1/datasets",
    tags=["datasets"]
)
app.include_router(
    jobs_router,
    prefix="/api/v1/jobs",
    tags=["jobs"]
)
app.include_router(
    artifacts_router,
    prefix="/api/v1/artifacts",
    tags=["artifacts"]
)
app.include_router(
    inference_router,
    prefix="/api/v1/inference",
    tags=["inference"]
)
app.include_router(
    models_api_router,
    prefix="/api/v1/models",
    tags=["models"]
)
app.include_router(
    colab_router,
    prefix="/api/v1/colab",
    tags=["colab"]
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "modelops-api"}


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Root endpoint."""
    return {
        "message": "ModelOps Platform API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.rest.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
