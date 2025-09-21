"""Session and recording lifecycle manager (FR4).

Responsibilities:
- Create unique session directories and metadata.json.
- Enforce single active session at a time.
- Manage session state transitions: Idle -> Created -> Recording -> Stopped.

This module is independent of the GUI; other components can import and use it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    # Centralized config loader (NFR8)
    from pc_controller.src.config import get as cfg_get
except ImportError:
    try:
        from ..config import get as cfg_get
    except ImportError:
        import sys

        project_root = Path(__file__).parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            from pc_controller.src.config import get as cfg_get
        except ImportError:
            # Ultimate fallback - create a no-op config function
            def cfg_get(key: str, default=None):
                """Fallback config function when config module unavailable."""
                return default


@dataclass
class SessionMetadata:
    version: int
    session_id: str
    name: str
    created_at_ns: int
    created_at: str
    state: str
    start_time_ns: int | None = None
    end_time_ns: int | None = None
    duration_ns: int | None = None


class SessionManager:
    def __init__(self, base_dir: str | None = None) -> None:
        self._base_dir = Path(base_dir or (Path.cwd() / "pc_controller_data")).resolve()
        self._active_id: str | None = None
        self._active_dir: Path | None = None
        self._meta: SessionMetadata | None = None

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def is_active(self) -> bool:
        return (
            self._active_id is not None
            and self._meta is not None
            and self._meta.state != "Stopped"
        )

    @property
    def session_id(self) -> str | None:
        return self._active_id

    @property
    def session_dir(self) -> Path | None:
        return self._active_dir

    @property
    def metadata(self) -> dict[str, Any] | None:
        return asdict(self._meta) if self._meta else None

    def _ensure_base(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _write_metadata(self) -> None:
        assert self._active_dir is not None and self._meta is not None
        p = self._active_dir / "metadata.json"
        with p.open("w", encoding="utf-8") as f:
            json.dump(asdict(self._meta), f, indent=2)

    def create_session(self, name: str) -> str:
        if self.is_active:
            raise RuntimeError(
                "A session is already active; stop it before creating a new one."
            )
        self._ensure_base()
        ts = time.strftime("%Y%m%d_%H%M%S")
        sid = f"{ts}_{_sanitize(name)}" if name else ts
        sdir = self._base_dir / sid
        sdir.mkdir(parents=True, exist_ok=True)
        created_ns = time.time_ns()
        created_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._meta = SessionMetadata(
            version=1,
            session_id=sid,
            name=name or sid,
            created_at_ns=created_ns,
            created_at=created_iso,
            state="Created",
        )
        self._active_id = sid
        self._active_dir = sdir
        self._write_metadata()
        return sid

    def start_recording(self) -> None:
        if not self._meta or not self._active_dir:
            raise RuntimeError("No session created.")
        if self._meta.state == "Recording":
            return
        self._meta.state = "Recording"
        self._meta.start_time_ns = time.time_ns()
        self._write_metadata()

    def start_session(self, name: str) -> str:
        """Convenience method to create a session and immediately start recording.

        Args:
            name: Session name

        Returns:
            Session ID
        """
        session_id = self.create_session(name)
        self.start_recording()
        return session_id

    def stop_recording(self) -> None:
        if not self._meta or not self._active_dir:
            return
        if self._meta.state == "Stopped":
            return
        self._meta.state = "Stopped"
        end_ns = time.time_ns()
        self._meta.end_time_ns = end_ns
        try:
            if self._meta.start_time_ns is not None:
                dur = max(0, int(end_ns - int(self._meta.start_time_ns)))
                self._meta.duration_ns = int(dur)
        except Exception:
            pass
        self._write_metadata()


def _sanitize(name: str) -> str:
    return (
        "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", ".")).strip("._-")
        or "session"
    )
