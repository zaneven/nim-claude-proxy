"""Persistence helper for updating .env file and hot-reloading runtime settings."""

import threading
from pathlib import Path
from loguru import logger
from nim_proxy.config.settings import Settings, settings
from nim_proxy.provider.key_pool import KeyPoolManager

# HOST / PORT 仅在进程启动时绑定，运行时修改无法生效，需重启后才能应用
_HOT_RELOAD_UNSUPPORTED = frozenset({"HOST", "PORT"})
_settings_lock = threading.Lock()


def save_settings_to_env(new_values: dict[str, str], env_path: str = ".env") -> bool:
    """Save dictionary of key-values into .env file, creating/updating lines."""
    path = Path(env_path)
    lines = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    updated_keys = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        if "=" in line:
            key, _, _ = line.partition("=")
            key = key.strip()
            if key in new_values:
                val = new_values[key]
                new_lines.append(f"{key}={val}")
                updated_keys.add(key)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # 追加入新的配置
    for key, val in new_values.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    logger.info(f"Successfully saved updated config to {env_path}")
    return True


def apply_settings_update(new_settings_dict: dict[str, str], key_pool: KeyPoolManager | None = None) -> None:
    """Update global settings instance and hot-reload KeyPoolManager.

    HOST / PORT are binding-time only and skipped here (a restart is required);
    all other fields are updated on the module-level settings singleton under a lock.
    """
    save_settings_to_env(new_settings_dict)

    with _settings_lock:
        for key, val in new_settings_dict.items():
            if key in _HOT_RELOAD_UNSUPPORTED:
                logger.warning(f"{key} updated in .env; restart the proxy for it to take effect.")
                continue

            attr_name = key.lower()
            if not hasattr(settings, attr_name):
                continue

            current_val = getattr(settings, attr_name)
            if isinstance(current_val, bool):
                setattr(settings, attr_name, str(val).lower() in ("true", "1", "yes"))
            elif isinstance(current_val, int):
                try:
                    setattr(settings, attr_name, int(val))
                except ValueError:
                    pass
            elif isinstance(current_val, float):
                try:
                    setattr(settings, attr_name, float(val))
                except ValueError:
                    pass
            else:
                setattr(settings, attr_name, str(val))

    # 重新装载 KeyPool (若存在)
    if key_pool is not None:
        new_keys = settings.parsed_api_keys()
        if new_keys:
            key_pool.add_keys(new_keys)
