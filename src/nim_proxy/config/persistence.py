"""Persistence helper for updating .env file and hot-reloading runtime settings."""

import threading
from pathlib import Path
from loguru import logger
from nim_proxy.config.settings import Settings, settings
from nim_proxy.provider.key_pool import KeyPoolManager

# HOST / PORT 仅在进程启动时绑定，运行时修改无法生效，需重启后才能应用
_HOT_RELOAD_UNSUPPORTED = frozenset({"HOST", "PORT"})
_settings_lock = threading.Lock()


def _sync_keys_line(keys: list[str], env_path: str = ".env") -> None:
    """重写 .env 中 NVIDIA_NIM_API_KEYS 单行为逗号分隔的 key 列表。"""
    path = Path(env_path)
    key_name = "NVIDIA_NIM_API_KEYS"
    value = ",".join(keys)

    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    found = False
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key_name}="):
            new_lines.append(f"{key_name}={value}")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"{key_name}={value}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    logger.debug(f"NVIDIA_NIM_API_KEYS synced to .env ({len(keys)} keys)")


def save_settings_to_env(new_values: dict[str, str], env_path: str = ".env") -> bool:
    """将键值字典存入 .env 文件，创建或更新对应行。"""
    path = Path(env_path)
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    updated_keys: set[str] = set()
    new_lines: list[str] = []

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

    # 追加入新的配置项
    for key, val in new_values.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    logger.info(f"Successfully saved updated config to {env_path}")
    return True


async def apply_settings_update(new_settings_dict: dict[str, str], key_pool: KeyPoolManager | None = None) -> None:
    """更新全局 settings 并热重载 KeyPoolManager。

    HOST / PORT 仅在进程启动时绑定，此处跳过（需重启生效）；
    其他字段在锁下更新模块级 settings 单例。
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

    # KeyPool 热重载
    if key_pool is not None:
        # cooldown_seconds 热更新
        if "KEY_COOLDOWN_SECONDS" in new_settings_dict:
            try:
                cooldown_val = float(new_settings_dict["KEY_COOLDOWN_SECONDS"])
                await key_pool.update_cooldown_seconds(cooldown_val)
            except (TypeError, ValueError):
                pass

        # keys 全量同步
        new_keys = settings.parsed_api_keys()
        if new_keys:
            await key_pool.sync_keys(new_keys)