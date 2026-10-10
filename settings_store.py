"""Validated settings: atomic updates and recoverable damaged files."""
from __future__ import annotations

import hashlib
import errno
import json
import math
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


_PATH_LOCKS: dict[str, Any] = {}
_PATH_LOCKS_GUARD = threading.Lock()


@contextmanager
def _file_lock(path: Path, timeout: float) -> Iterator[None]:
    """Bounded advisory lock shared by independent local server processes."""
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + timeout
        if os.name == "nt":
            import msvcrt
            def try_lock():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            def unlock():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            def try_lock():
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            def unlock():
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        while True:
            try:
                try_lock()
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OSError("다른 실행에서 설정을 저장 중입니다. 잠시 후 다시 저장하세요.") from exc
                time.sleep(min(0.025, remaining))
        try:
            yield
        finally:
            unlock()


class SettingsConflict(ValueError):
    """The edited settings version is no longer current."""


def _validate(values: dict[str, Any]) -> dict[str, Any]:
    threshold = values["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold는 0.05~0.95 사이의 숫자여야 합니다.")
    try:
        finite = math.isfinite(threshold)
    except OverflowError:
        finite = False
    if not finite or not 0.05 <= threshold <= 0.95:
        raise ValueError("threshold는 0.05~0.95 사이의 유한한 숫자여야 합니다.")
    max_results = values["max_results"]
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 50:
        raise ValueError("max_results는 1~50 사이의 정수여야 합니다.")
    if not isinstance(values["retain_body"], bool):
        raise ValueError("retain_body는 true 또는 false여야 합니다.")
    return {"threshold": float(threshold), "max_results": max_results, "retain_body": values["retain_body"]}


def _canonical(values: dict[str, Any]) -> bytes:
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class SettingsStore:
    FIELDS = frozenset({"threshold", "max_results", "retain_body"})
    LOCK_TIMEOUT_SECONDS = 2.0

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        with _PATH_LOCKS_GUARD:
            self._lock = _PATH_LOCKS.setdefault(os.path.normcase(str(self.path)), threading.RLock())
        try:
            configured = float(os.environ.get("PHISH_THRESHOLD", "0.20"))
            if not math.isfinite(configured) or not 0.05 <= configured <= 0.95:
                configured = 0.20
        except (TypeError, ValueError, OverflowError):
            configured = 0.20
        self._defaults = {"threshold": configured, "max_results": 10, "retain_body": True}

    def _read(self) -> tuple[dict[str, Any], bytes | None, bool]:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            values = dict(self._defaults)
            return {**values, "version": hashlib.sha256(_canonical(values)).hexdigest(), "warnings": []}, None, False
        try:
            decoded = json.loads(raw.decode("utf-8"))
            if not isinstance(decoded, dict) or set(decoded) != self.FIELDS:
                raise ValueError("잘못된 설정 필드")
            values = _validate(decoded)
        except (ValueError, UnicodeError, OverflowError, RecursionError):
            values = dict(self._defaults)
            return {
                **values,
                "version": hashlib.sha256(b"invalid-settings\x00" + raw).hexdigest(),
                "warnings": ["설정 파일을 읽을 수 없어 기본값을 사용합니다. 원본 파일은 보존되며 저장 시 복구 백업을 만듭니다."],
            }, raw, True
        return {**values, "version": hashlib.sha256(_canonical(values)).hexdigest(), "warnings": []}, raw, False

    def load(self) -> dict[str, Any]:
        """Read disk again on each call; missing files are not created."""
        with self._lock:
            return self._read()[0]

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("설정은 JSON 객체여야 합니다.")
        if set(payload) - self.FIELDS - {"version"}:
            raise ValueError("지원하지 않는 설정 필드가 있습니다.")
        if "version" in payload and not isinstance(payload["version"], str):
            raise ValueError("version은 문자열이어야 합니다.")
        _validate({**self._defaults, **{name: value for name, value in payload.items() if name in self.FIELDS}})
        if not self._lock.acquire(timeout=self.LOCK_TIMEOUT_SECONDS):
            raise OSError("다른 요청에서 설정을 저장 중입니다. 잠시 후 다시 저장하세요.")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _file_lock(self.path.with_name(self.path.name + ".lock"), self.LOCK_TIMEOUT_SECONDS):
                return self._update_locked(payload)
        finally:
            self._lock.release()

    def _update_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Both the version check and replace are covered by the same file lock.
        with self._lock:
            current, raw, damaged = self._read()
            if "version" in payload and payload["version"] != current["version"]:
                raise SettingsConflict("다른 창에서 설정이 변경되었습니다. 최신 설정을 다시 불러오세요.")
            values = _validate({name: payload.get(name, current[name]) for name in self.FIELDS})
            serialized = json.dumps(values, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
            temporary: Path | None = None
            backup: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                if damaged:
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    backup = self.path.with_name(f"{self.path.name}.corrupt-{stamp}-{uuid.uuid4().hex}.bak")
                    with backup.open("xb") as handle:
                        handle.write(raw or b"")
                        handle.flush()
                        os.fsync(handle.fileno())
                os.replace(temporary, self.path)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            saved = self._read()[0]
            if backup is not None:
                saved["warnings"].append(f"손상된 설정 원본을 {backup.name}에 보관하고 설정을 복구했습니다.")
            return saved
