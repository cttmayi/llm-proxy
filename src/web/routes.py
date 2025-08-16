"""
Web interface routes for the LLM Proxy.
"""
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from .monitor_routes import router as monitor_router

# Get the current directory
current_dir = Path(__file__).parent

# Create router
router = APIRouter(prefix="/web", tags=["web"])


# Setup templates
templates = Jinja2Templates(directory=current_dir / "templates")


@router.get("/", response_class=HTMLResponse)
async def web_interface(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/monitor", response_class=HTMLResponse)
async def monitor_interface(request: Request):
    """Serve the API monitoring interface."""
    return templates.TemplateResponse("monitor.html", {"request": request})


@router.get("/models", response_class=HTMLResponse)
async def models_interface(request: Request):
    """Serve the models interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/embeddings", response_class=HTMLResponse)
async def embeddings_interface(request: Request):
    """Serve the embeddings interface."""
    return templates.TemplateResponse("index.html", {"request": request})


# Include monitor API routes
router.include_router(monitor_router, prefix="/api")


def create_web_router():
    """Factory function to create the web router."""
    return router