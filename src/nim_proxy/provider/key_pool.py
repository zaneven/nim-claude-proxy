"""Multi-Key Pool Manager for NVIDIA NIM with Round-Robin and Failover Cooldown."""

import asyncio
import time
from dataclasses import dataclass
from loguru import logger


class KeyPoolExhaustedError(RuntimeError):
    """All active keys are in cooldown and the wait exceeds the fail-fast threshold."""


@dataclass
class KeyStatus:
    api_key: str
    is_disabled: bool = False
    cooldown_until: float = 0.0
    failure_count: int = 0

    @property
    def masked_key(self) -> str:
        if len(self.api_key) <= 8:
            return "***"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"


class KeyPoolManager:
    """Manage round-robin dispatching and automatic health cooldown for NVIDIA NIM API keys."""

    def __init__(self, api_keys: list[str], cooldown_seconds: float = 60.0):
        self._keys: list[KeyStatus] = [
            KeyStatus(api_key=k.strip()) for k in api_keys if k.strip()
        ]
        self._cooldown_seconds = cooldown_seconds
        self._index = 0
        self._lock = asyncio.Lock()

    async def add_keys(self, keys: list[str]) -> None:
        async with self._lock:
            existing = {k.api_key for k in self._keys}
            for k in keys:
                clean = k.strip()
                if clean and clean not in existing:
                    self._keys.append(KeyStatus(api_key=clean))

    async def remove_key(self, api_key: str) -> bool:
        """按精确值删除一个 key。返回是否找到并删除。"""
        async with self._lock:
            for i, k in enumerate(self._keys):
                if k.api_key == api_key:
                    self._keys.pop(i)
                    self._index = min(self._index, len(self._keys) - 1) if self._keys else 0
                    return True
        return False

    async def replace_key(self, old_key: str, new_key: str) -> bool:
        """替换已存在的 key 为新值，保持健康状态不变。"""
        async with self._lock:
            for k in self._keys:
                if k.api_key == old_key:
                    k.api_key = new_key
                    return True
        return False

    async def sync_keys(self, keys: list[str]) -> dict[str, int]:
        """全量同步 key 列表：新增的加入，多余的删除。返回 {added, removed}。"""
        async with self._lock:
            existing = {k.api_key for k in self._keys}
            incoming = {k.strip() for k in keys if k.strip()}
            added = incoming - existing
            removed = existing - incoming
            self._keys = [k for k in self._keys if k.api_key not in removed]
            for k in sorted(added):
                self._keys.append(KeyStatus(api_key=k))
            self._index = min(self._index, len(self._keys) - 1) if self._keys else 0
            return {"added": len(added), "removed": len(removed)}

    async def update_cooldown_seconds(self, seconds: float) -> None:
        """动态更新 cooldown 时长。"""
        self._cooldown_seconds = seconds

    async def get_next_key(self) -> str:
        """Get next active API key via Round-Robin, bypassing keys in cooldown/disabled state."""
        if not self._keys:
            raise RuntimeError("No NVIDIA NIM API keys configured.")

        # 最多重试几次：等待最早冷却结束的 Key 后重新轮询
        for _ in range(3):
            async with self._lock:
                now = time.time()
                total = len(self._keys)

                # 轮询查找首个可用 Key
                for _ in range(total):
                    key_obj = self._keys[self._index]
                    self._index = (self._index + 1) % total

                    if key_obj.is_disabled:
                        continue
                    if key_obj.cooldown_until > now:
                        continue

                    return key_obj.api_key

                # 若所有 Key 均在冷却中，查找最早冷却结束的 Key
                active_keys = [k for k in self._keys if not k.is_disabled]
                if not active_keys:
                    raise RuntimeError("All configured NVIDIA NIM API keys are permanently disabled (401 Unauthorized).")

                earliest = min(active_keys, key=lambda k: k.cooldown_until)
                wait_time = max(0.1, earliest.cooldown_until - now)

            # 必须在释放锁之后再等待，避免阻塞其他请求获取 Key
            logger.warning(
                f"All active NIM API keys in cooldown. Waiting {wait_time:.1f}s for earliest key [{earliest.masked_key}]..."
            )
            if wait_time > 10.0:
                raise KeyPoolExhaustedError(f"All active NIM API keys in cooldown. Wait time {wait_time:.1f}s exceeds threshold. Failing fast.")
            await asyncio.sleep(wait_time)

        raise KeyPoolExhaustedError("Unable to acquire a NIM API key after cooldown wait (unexpected).")

    async def report_status(self, api_key: str, status_code: int | None = None, error: Exception | None = None) -> None:
        """Report HTTP status or network error for a key to trigger failover/cooldown."""
        async with self._lock:
            now = time.time()
            for key_obj in self._keys:
                if key_obj.api_key == api_key:
                    key_obj.failure_count += 1

                    # 401 权限无效 -> 永久禁用
                    if status_code == 401:
                        key_obj.is_disabled = True
                        logger.error(
                            f"NIM Key [{key_obj.masked_key}] PERMANENTLY DISABLED due to 401 Unauthorized."
                        )
                    # 429 限流 或 5xx 服务端故障 -> 进入冷却状态
                    elif status_code in (429, 500, 502, 503, 504):
                        key_obj.cooldown_until = now + self._cooldown_seconds
                        logger.warning(
                            f"NIM Key [{key_obj.masked_key}] placed in COOLDOWN for {self._cooldown_seconds}s. (Status {status_code})"
                        )
                    # 网络层异常（连接失败/超时）不冷却 Key：瞬时故障不应烧掉 Key 60s
                    else:
                        logger.debug(
                            f"NIM Key [{key_obj.masked_key}] transient transport error (no cooldown): {error}"
                        )
                    break

    async def report_success(self, api_key: str) -> None:
        """Reset failure counter upon successful response."""
        async with self._lock:
            for key_obj in self._keys:
                if key_obj.api_key == api_key:
                    key_obj.failure_count = 0
                    break

    def get_stats(self) -> dict[str, int]:
        now = time.time()
        active = sum(1 for k in self._keys if not k.is_disabled and k.cooldown_until <= now)
        cooldown = sum(1 for k in self._keys if not k.is_disabled and k.cooldown_until > now)
        disabled = sum(1 for k in self._keys if k.is_disabled)
        return {
            "total": len(self._keys),
            "active": active,
            "cooldown": cooldown,
            "disabled": disabled,
        }
