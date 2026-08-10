"""Entrypoint for running nim-claude-proxy server."""

import uvicorn
from loguru import logger
from nim_proxy.api.app import app
from nim_proxy.config.settings import settings
from nim_proxy.fcc_port.config.logging_config import configure_logging


def main():
    # 启用文件日志：写入 logs/server.log，50 MB 轮转，保留 5 个历史文件
    configure_logging(
        "logs/server.log",
        level=settings.log_level,
    )

    keys = settings.parsed_api_keys()
    logger.info(f"Loaded {len(keys)} NVIDIA NIM API key(s) for Multi-Key Round-Robin dispatching.")
    logger.info(f"Default Model: {settings.nim_default_model}")
    logger.info(f"Proxy API Server starting on host={settings.host}, port={settings.port}")
    
    print("\n" + "=" * 65)
    print(f"  NIM Proxy Server Ready!")
    print(f"  --> Admin Dashboard UI URL : http://localhost:{settings.port}/admin")
    print(f"  --> Proxy Endpoint URL     : http://localhost:{settings.port}/v1/messages")
    print("=" * 65 + "\n")

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
