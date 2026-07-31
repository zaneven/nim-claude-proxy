"""FastAPI Application Factory for nim-claude-proxy."""

from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from nim_proxy.api.admin_routes import admin_router, init_admin_globals
from nim_proxy.api.routes import router, _key_pool, _client
from nim_proxy.config.settings import settings

PUBLIC_PATHS = {"/health", "/v1/models", "/models", "/favicon.ico"}
# 静态控制台页面可匿名访问，但其下的 /admin/api/* 一律需要鉴权，
# 避免未授权读取明文 API Key 或篡改 .env 配置。
ADMIN_UI_PATH = "/admin"


def create_app() -> FastAPI:
    app = FastAPI(
        title="NVIDIA NIM Dedicated Proxy",
        description="High-availability local proxy connecting AI coding agents to NVIDIA NIM with Multi-Key failover and Admin UI.",
        version="0.1.0",
    )

    # 1. 跨域 CORS 支持（明确来源，避免使用 "*" + 凭据的危险组合）
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

    # 2. 可选 Bearer Token 校验中间件 (用于保护 API 代理)
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        # 仅静态 UI 页与白名单路径匿名放行；/admin/api/* 必须带 Token
        is_public = path == ADMIN_UI_PATH or path in PUBLIC_PATHS

        if settings.proxy_api_token and not is_public:
            auth_header = request.headers.get("Authorization", "")
            expected_bearer = f"Bearer {settings.proxy_api_token}"
            raw_key_header = request.headers.get("x-api-key", "")
            
            # 兼容 Bearer Token、x-api-key 与 query param
            if (
                auth_header != expected_bearer
                and raw_key_header != settings.proxy_api_token
                and request.query_params.get("token") != settings.proxy_api_token
            ):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized proxy access token. Please provide valid Authorization header."}
                )

        return await call_next(request)

    # 3. 初始化 Admin API 全局对象
    init_admin_globals(_key_pool, _client)

    # 4. 挂载 路由
    app.include_router(router)
    app.include_router(admin_router)

    # 5. 挂载 Admin UI 静态网页
    static_dir = Path(__file__).parent.parent / "admin_static"
    if static_dir.exists():
        @app.get("/admin", response_class=HTMLResponse)
        async def serve_admin_ui():
            return FileResponse(static_dir / "index.html")

    return app


app = create_app()
