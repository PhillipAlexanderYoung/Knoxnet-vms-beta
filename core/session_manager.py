from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .camera_manager import CameraManager
from .layout_store import LayoutsAndProfilesStore


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class RunSession:
    id: str
    name: str
    layout_ids: List[str] = field(default_factory=list)
    status: str = "stopped"  # running|stopped
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SessionManager:
    """
    Owns 'run sessions' which can include multiple layouts.

    Key invariant: camera decode work is NOT duplicated; CameraManager is shared.
    This manager only increments/decrements usage of cameras by active sessions.
    """

    def __init__(
        self,
        store: LayoutsAndProfilesStore,
        camera_manager: Optional[CameraManager] = None,
        auto_disconnect_unused: bool = False,
    ):
        self.store = store
        self.camera_manager = camera_manager
        self.auto_disconnect_unused = bool(auto_disconnect_unused)

        self._lock = threading.Lock()
        self._sessions: Dict[str, RunSession] = {}
        self._camera_refcount: Dict[str, int] = {}

    # ---- sessions ----
    def list_sessions(self) -> List[RunSession]:
        with self._lock:
            out = list(self._sessions.values())
        out.sort(key=lambda s: s.created_at)
        return out

    def get_session(self, session_id: str) -> Optional[RunSession]:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def create_session(self, layout_ids: List[str], name: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> RunSession:
        layout_ids = [str(x).strip() for x in (layout_ids or []) if str(x).strip()]
        sid = str(uuid.uuid4())
        sess = RunSession(
            id=sid,
            name=name or f"Session {sid[:8]}",
            layout_ids=layout_ids,
            status="stopped",
            meta=meta or {},
        )
        with self._lock:
            self._sessions[sid] = sess
        return sess

    def delete_session(self, session_id: str) -> bool:
        if not session_id:
            return False
        # Stop first to release cameras
        try:
            self.stop_session(session_id)
        except Exception:
            pass
        with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
        return existed

    def attach_layouts(self, session_id: str, layout_ids: List[str]) -> Optional[RunSession]:
        layout_ids = [str(x).strip() for x in (layout_ids or []) if str(x).strip()]
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return None
            before = list(sess.layout_ids)
            for lid in layout_ids:
                if lid not in sess.layout_ids:
                    sess.layout_ids.append(lid)
            sess.updated_at = _utc_now_iso()
            running = sess.status == "running"
        if running:
            # If running, ensure newly attached layouts' cameras are active
            self._apply_camera_deltas_for_session(session_id, before_layout_ids=before, after_layout_ids=sess.layout_ids)
        return sess

    def detach_layouts(self, session_id: str, layout_ids: List[str]) -> Optional[RunSession]:
        layout_ids = {str(x).strip() for x in (layout_ids or []) if str(x).strip()}
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return None
            before = list(sess.layout_ids)
            sess.layout_ids = [x for x in sess.layout_ids if x not in layout_ids]
            sess.updated_at = _utc_now_iso()
            running = sess.status == "running"
        if running:
            self._apply_camera_deltas_for_session(session_id, before_layout_ids=before, after_layout_ids=sess.layout_ids)
        return sess

    def start_session(self, session_id: str) -> Optional[RunSession]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return None
            if sess.status == "running":
                return sess
            sess.status = "running"
            sess.updated_at = _utc_now_iso()
            layout_ids = list(sess.layout_ids)
        # Acquire cameras
        cam_ids = self._camera_ids_for_layouts(layout_ids)
        self._inc_camera_refs(cam_ids)
        self._ensure_cameras_connected(cam_ids)
        return sess

    def stop_session(self, session_id: str) -> Optional[RunSession]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return None
            if sess.status != "running":
                return sess
            sess.status = "stopped"
            sess.updated_at = _utc_now_iso()
            layout_ids = list(sess.layout_ids)
        cam_ids = self._camera_ids_for_layouts(layout_ids)
        self._dec_camera_refs(cam_ids)
        return sess

    # ---- internal helpers ----
    def _camera_ids_for_layouts(self, layout_ids: List[str]) -> Set[str]:
        out: Set[str] = set()
        for lid in layout_ids or []:
            layout = self.store.get_layout(lid)
            if not layout:
                continue
            for w in layout.widgets:
                if w.type == "camera" and w.camera_id:
                    out.add(str(w.camera_id))
        return out

    def _apply_camera_deltas_for_session(self, session_id: str, before_layout_ids: List[str], after_layout_ids: List[str]) -> None:
        before = self._camera_ids_for_layouts(before_layout_ids)
        after = self._camera_ids_for_layouts(after_layout_ids)
        added = after - before
        removed = before - after
        if added:
            self._inc_camera_refs(added)
            self._ensure_cameras_connected(added)
        if removed:
            self._dec_camera_refs(removed)

    def _inc_camera_refs(self, camera_ids: Set[str]) -> None:
        if not camera_ids:
            return
        with self._lock:
            for cam_id in camera_ids:
                self._camera_refcount[cam_id] = int(self._camera_refcount.get(cam_id, 0)) + 1

    def _dec_camera_refs(self, camera_ids: Set[str]) -> None:
        if not camera_ids:
            return
        to_disconnect: List[str] = []
        with self._lock:
            for cam_id in camera_ids:
                cur = int(self._camera_refcount.get(cam_id, 0))
                nxt = max(0, cur - 1)
                if nxt == 0:
                    self._camera_refcount.pop(cam_id, None)
                    if self.auto_disconnect_unused:
                        to_disconnect.append(cam_id)
                else:
                    self._camera_refcount[cam_id] = nxt
        if to_disconnect:
            self._maybe_disconnect_cameras(to_disconnect)

    def _ensure_cameras_connected(self, camera_ids: Set[str]) -> None:
        if not camera_ids or not self.camera_manager:
            return
        # CameraManager is async. We use its existing loop when possible by calling its sync wrapper if present.
        for cam_id in camera_ids:
            try:
                if cam_id in getattr(self.camera_manager, "active_streams", {}):
                    continue
                # Prefer scheduling on existing loop if present
                if hasattr(self.camera_manager, "_loop") and getattr(self.camera_manager, "_loop", None) is not None:
                    import asyncio
                    loop = getattr(self.camera_manager, "_loop")
                    if loop and loop.is_running():
                        asyncio.run_coroutine_threadsafe(self.camera_manager.connect_camera(cam_id), loop)
                        continue
                # Fallback: best-effort direct asyncio.run
                import asyncio
                asyncio.run(self.camera_manager.connect_camera(cam_id))
            except Exception:
                # Best-effort; keep session running even if camera connect fails
                continue

    def _maybe_disconnect_cameras(self, camera_ids: List[str]) -> None:
        if not camera_ids or not self.camera_manager:
            return
        for cam_id in camera_ids:
            try:
                if cam_id not in getattr(self.camera_manager, "active_streams", {}):
                    continue
                if hasattr(self.camera_manager, "_loop") and getattr(self.camera_manager, "_loop", None) is not None:
                    import asyncio
                    loop = getattr(self.camera_manager, "_loop")
                    if loop and loop.is_running():
                        asyncio.run_coroutine_threadsafe(self.camera_manager.disconnect_camera(cam_id), loop)
                        continue
                import asyncio
                asyncio.run(self.camera_manager.disconnect_camera(cam_id))
            except Exception:
                continue


