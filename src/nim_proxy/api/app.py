"""FastAPI Application Factory for nim-claude-proxy."""

import secrets
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from nim_proxy import __version__
from nim_proxy.api.admin_routes import admin_router, init_admin_globals
from nim_proxy.api.routes import router, _key_pool, nim_provider
from nim_proxy.config.settings import settings

PUBLIC_PATHS = {"/health", "/v1/models", "/models", "/favicon.ico"}
ADMIN_UI_PATH = "/admin"


def _check_auth(request: Request, expected_token: str) -> bool:
    """Timing-safe check of Bearer token, x-api-key, or ?token= query param."""
    auth_header = request.headers.get("Authorization", "")
    parts = auth_header.strip().split(maxsplit=1)
    if len(parts) == 2 and parts[0].casefold() == "bearer":
        return secrets.compare_digest(
            parts[1].strip().encode("utf-8"), expected_token.encode("utf-8")
        )

    raw_key_header = request.headers.get("x-api-key", "")
    if raw_key_header:
        return secrets.compare_digest(
            raw_key_header.strip().encode("utf-8"), expected_token.encode("utf-8")
        )

    query_token = request.query_params.get("token", "")
    if query_token:
        return secrets.compare_digest(
            query_token.strip().encode("utf-8"), expected_token.encode("utf-8")
        )

    return False


def create_app() -> FastAPI:
    app = FastAPI(
        title="NVIDIA NIM Dedicated Proxy",
        description="High-availability relay connecting AI coding agents to NVIDIA NIM with Multi-Key failover and Admin UI.",
        version=__version__,
    )

    # 1. CORS
    raw_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if raw_origins == ["*"]:
        allow_origins = ["*"]
        allow_credentials = False
    else:
        allow_origins = raw_origins
        allow_credentials = True

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Bearer Token auth middleware (optional, timing-safe)
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        is_public = (
            path == ADMIN_UI_PATH
            or path in PUBLIC_PATHS
            or (path.startswith(ADMIN_UI_PATH + "/") and not path.startswith(ADMIN_UI_PATH + "/api/"))
        )

        if settings.proxy_api_token and not is_public:
            if not _check_auth(request, settings.proxy_api_token):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized proxy access token."},
                )

        return await call_next(request)

    # 3. Admin globals
    init_admin_globals(_key_pool, nim_provider)

    # 4. Routes
    app.include_router(router)
    app.include_router(admin_router)

    NO_CACHE_HEADERS = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    # 5. Admin UI static resources
    static_dir = Path(__file__).parent.parent / "admin_static"
    if static_dir.exists():
        @app.get("/admin", response_class=HTMLResponse)
        async def serve_admin_ui():
            return FileResponse(static_dir / "index.html", headers=NO_CACHE_HEADERS)

        @app.get("/admin/{filename:path}")
        async def serve_admin_static(filename: str):
            file_path = static_dir / filename
            if not file_path.resolve().is_relative_to(static_dir.resolve()):
                return JSONResponse(status_code=404, content={"detail": "Not found"})
            if file_path.is_file():
                return FileResponse(file_path, headers=NO_CACHE_HEADERS)
            return JSONResponse(status_code=404, content={"detail": "Not found"})

    return app


app = create_app()