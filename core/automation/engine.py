from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .conditions import build_eval_context, matches_rule
from .state import AutomationState

try:
    import json as _pyjson
except Exception:  # pragma: no cover
    _pyjson = None  # type: ignore

try:
    from core.events import EventDetection, EventTrack, EventOverlay, build_event_bundle
except Exception:  # pragma: no cover
    EventDetection = None  # type: ignore
    EventTrack = None  # type: ignore
    EventOverlay = None  # type: ignore
    build_event_bundle = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class AutomationEvent:
    id: str
    kind: str
    camera_id: str
    created_at: str
    payload: Dict[str, Any]


class AutomationEngine:
    """
    Server-side rule evaluation and action runner.

    v1 focuses on safe rule evaluation + producing trigger results.
    Action execution (email/webhook/etc.) plugs into `action_handlers`.
    """

    def __init__(
        self,
        *,
        db_manager: Any,
        stream_server: Any = None,
        socketio: Any = None,
        dry_run: bool = False,
        max_queue: int = 5000,
    ) -> None:
        self.db_manager = db_manager
        self.stream_server = stream_server
        self.socketio = socketio
        self.dry_run = bool(dry_run)

        self._q: "queue.Queue[AutomationEvent]" = queue.Queue(maxsize=max(1, int(max_queue)))
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.state = AutomationState()

        # Action handlers: action_type -> callable(rule, ctx, match_details, action_cfg)
        self.action_handlers: Dict[str, Callable[..., None]] = {}

        # Rule caching (avoid hitting sqlite for every frame event)
        self._rules_cache_ts = 0.0
        self._rules_cache: List[Dict[str, Any]] = []
        self._rules_cache_ttl_sec = 2.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="automation-engine", daemon=True)
        self._thread.start()
        logger.info("✅ AutomationEngine started (dry_run=%s)", self.dry_run)

    def stop(self) -> None:
        self._running = False
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
        except Exception:
            pass
        logger.info("AutomationEngine stopped")

    # --------------------------------------------------------------------- Intake
    def submit(self, kind: str, camera_id: str, payload: Optional[Dict[str, Any]] = None) -> bool:
        evt = AutomationEvent(
            id=f"auto_evt_{uuid.uuid4().hex[:10]}",
            kind=str(kind),
            camera_id=str(camera_id),
            created_at=datetime.now().isoformat(),
            payload=payload or {},
        )
        try:
            self._q.put_nowait(evt)
            return True
        except queue.Full:
            # Prefer dropping over blocking realtime pipelines.
            logger.warning("AutomationEngine queue full; dropping event kind=%s camera=%s", evt.kind, evt.camera_id)
            return False

    # --------------------------------------------------------------------- Internals
    def _load_rules_cached(self) -> List[Dict[str, Any]]:
        now = time.time()
        if self._rules_cache and (now - self._rules_cache_ts) < self._rules_cache_ttl_sec:
            return self._rules_cache
        try:
            rules = []
            if self.db_manager and hasattr(self.db_manager, "list_rules"):
                rules = self.db_manager.list_rules(enabled=True)
            self._rules_cache = rules or []
            self._rules_cache_ts = now
            return self._rules_cache
        except Exception as e:
            logger.debug("Failed to load rules: %s", e)
            return []

    def _get_shape_by_id(self, camera_id: str, shape_id: str) -> Optional[Dict[str, Any]]:
        if not shape_id:
            return None
        try:
            if self.db_manager and hasattr(self.db_manager, "get_camera_shapes"):
                rec = self.db_manager.get_camera_shapes(camera_id)
                if rec and isinstance(rec, dict):
                    for z in rec.get("zones", []) or []:
                        if isinstance(z, dict) and str(z.get("id")) == str(shape_id):
                            out = dict(z)
                            out.setdefault("kind", "zone")
                            return out
                    for l in rec.get("lines", []) or []:
                        if isinstance(l, dict) and str(l.get("id")) == str(shape_id):
                            out = dict(l)
                            out.setdefault("kind", "line")
                            return out
                    for t in rec.get("tags", []) or []:
                        if isinstance(t, dict) and str(t.get("id")) == str(shape_id):
                            out = dict(t)
                            out.setdefault("kind", "tag")
                            return out
        except Exception:
            return None
        return None

    def _worker(self) -> None:
        while self._running:
            try:
                evt = self._q.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                self._process(evt)
            except Exception as e:
                logger.exception("AutomationEngine process error: %s", e)
            finally:
                try:
                    self._q.task_done()
                except Exception:
                    pass

    def _process(self, evt: AutomationEvent) -> None:
        rules = self._load_rules_cached()
        if not rules:
            return

        ctx = build_eval_context(evt.kind, evt.camera_id, evt.payload)

        for rule in rules:
            try:
                rule_id = str(rule.get("id") or "")
                if not rule_id:
                    continue

                # Cooldown
                cooldown = 0.0
                conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
                try:
                    cooldown = float(conditions.get("cooldown_sec", conditions.get("cooldown", 0)) or 0)
                except Exception:
                    cooldown = 0.0
                if self.state.is_in_cooldown(rule_id=rule_id, camera_id=evt.camera_id, cooldown_sec=cooldown):
                    continue

                # Optional shape constraint
                shape = None
                shape_id = rule.get("shape_id")
                if shape_id:
                    shape = self._get_shape_by_id(evt.camera_id, str(shape_id))
                    if shape is None:
                        # shape not found -> treat as non-match (avoids silent false triggers)
                        continue

                ok, details = matches_rule(rule=rule, ctx=ctx, shape=shape)
                if not ok:
                    continue

                # Best-effort dedupe
                signature = f"{evt.kind}:{rule_id}:{details.get('filtered_object_count',0)}:{shape_id or ''}"
                if self.state.is_duplicate(rule_id=rule_id, camera_id=evt.camera_id, signature=signature, window_sec=2.0):
                    continue

                self.state.mark_triggered(rule_id=rule_id, camera_id=evt.camera_id)

                self._on_rule_triggered(rule, details, evt, ctx)
            except Exception:
                continue

    def _on_rule_triggered(self, rule: Dict[str, Any], details: Dict[str, Any], evt: AutomationEvent, ctx: Any) -> None:
        rule_id = str(rule.get("id") or "")
        name = str(rule.get("name") or rule_id)
        logger.info("🚨 Rule triggered: %s (%s) on camera=%s kind=%s", name, rule_id, evt.camera_id, evt.kind)

        # Persist a compact record for history/forensics (EventBundle)
        bundle_id = None
        try:
            if (
                self.db_manager is not None
                and hasattr(self.db_manager, "store_event_bundle")
                and build_event_bundle is not None
                and _pyjson is not None
            ):
                # Only store a snapshot if requested by any action (or forced by env).
                store_snapshot = any(
                    isinstance(a, dict) and bool(a.get("include_snapshot"))
                    for a in (rule.get("actions") if isinstance(rule.get("actions"), list) else [])
                )
                snapshot_base64 = None
                if store_snapshot and self.stream_server is not None and hasattr(self.stream_server, "get_frame_base64"):
                    try:
                        snapshot_base64 = self.stream_server.get_frame_base64(str(evt.camera_id))
                    except Exception:
                        snapshot_base64 = None

                detections = []
                tracks = []
                if EventDetection is not None:
                    for d in getattr(ctx, "detections", []) or []:
                        if not isinstance(d, dict):
                            continue
                        bbox = d.get("bbox") or {}
                        detections.append(
                            EventDetection(
                                class_name=str(d.get("class") or d.get("class_name") or "object"),
                                confidence=float(d.get("confidence", 0.0) or 0.0),
                                bbox={
                                    "x": float(bbox.get("x", 0)),
                                    "y": float(bbox.get("y", 0)),
                                    "w": float(bbox.get("w", 0)),
                                    "h": float(bbox.get("h", 0)),
                                },
                            )
                        )
                if EventTrack is not None:
                    for t in getattr(ctx, "tracks", []) or []:
                        if not isinstance(t, dict):
                            continue
                        bbox = t.get("bbox") or {}
                        tracks.append(
                            EventTrack(
                                id=int(t.get("id", 0) or 0),
                                bbox={
                                    "x": float(bbox.get("x", 0)),
                                    "y": float(bbox.get("y", 0)),
                                    "w": float(bbox.get("w", 0)),
                                    "h": float(bbox.get("h", 0)),
                                },
                                history=[],
                            )
                        )

                overlays = None
                # If we can, include overlays in pixel space for future UI rendering.
                try:
                    if EventOverlay is not None and hasattr(self.db_manager, "get_camera_shapes"):
                        shapes = self.db_manager.get_camera_shapes(str(evt.camera_id)) or {}
                        fw = int(getattr(ctx, "frame_w", 0) or 0)
                        fh = int(getattr(ctx, "frame_h", 0) or 0)
                        if fw > 0 and fh > 0 and isinstance(shapes, dict):
                            zones_px = []
                            for z in (shapes.get("zones") or []):
                                if not isinstance(z, dict) or z.get("enabled") is False:
                                    continue
                                pts = z.get("points") or []
                                if not isinstance(pts, list) or len(pts) < 3:
                                    continue
                                poly = []
                                for p in pts:
                                    if not isinstance(p, dict):
                                        continue
                                    poly.append({"x": float(p.get("x", 0.0)) * fw, "y": float(p.get("y", 0.0)) * fh})
                                if len(poly) >= 3:
                                    zones_px.append(poly)
                            lines_px = []
                            for l in (shapes.get("lines") or []):
                                if not isinstance(l, dict) or l.get("enabled") is False:
                                    continue
                                p1 = l.get("p1") or {}
                                p2 = l.get("p2") or {}
                                lines_px.append(
                                    {
                                        "p1": {"x": float(p1.get("x", 0.0)) * fw, "y": float(p1.get("y", 0.0)) * fh},
                                        "p2": {"x": float(p2.get("x", 1.0)) * fw, "y": float(p2.get("y", 1.0)) * fh},
                                    }
                                )
                            tags_px = []
                            for t in (shapes.get("tags") or []):
                                if not isinstance(t, dict) or t.get("enabled") is False:
                                    continue
                                tags_px.append({"x": float(t.get("x", 0.0)) * fw, "y": float(t.get("y", 0.0)) * fh})
                            overlays = EventOverlay(zones=zones_px, lines=lines_px, tags=tags_px)
                except Exception:
                    overlays = None

                bundle_id = str(uuid.uuid4())
                bundle = build_event_bundle(
                    bundle_id=bundle_id,
                    camera_id=str(evt.camera_id),
                    kind="rule_triggered",
                    detections=detections,
                    tracks=tracks,
                    overlays=overlays,
                    snapshot_base64=snapshot_base64,
                    metadata={
                        "rule": {"id": rule_id, "name": name},
                        "event": {"kind": evt.kind, "created_at": evt.created_at},
                        "details": details,
                    },
                )
                self.db_manager.store_event_bundle(
                    bundle_id=bundle.id,
                    camera_id=bundle.camera_id,
                    kind=bundle.kind,
                    created_at=bundle.created_at,
                    bundle_json=_pyjson.dumps(bundle.to_dict()),
                )
        except Exception as e:
            logger.debug("Failed to persist automation bundle: %s", e)

        # Optional realtime emission for UI/desktop (observability will formalize this later)
        try:
            if self.socketio is not None:
                payload = {
                    "id": f"rule_evt_{uuid.uuid4().hex[:10]}",
                    "type": "rule_triggered",
                    "timestamp": datetime.now().isoformat(),
                    "camera_id": evt.camera_id,
                    "rule": {
                        "id": rule_id,
                        "name": name,
                    },
                    "event": {
                        "kind": evt.kind,
                        "created_at": evt.created_at,
                    },
                    "details": details,
                    "bundle_id": bundle_id,
                }
                try:
                    self.socketio.emit("automation_alert", payload, room=f"camera:{evt.camera_id}")
                except Exception:
                    self.socketio.emit("automation_alert", payload)
        except Exception:
            pass

        # Run actions
        actions = rule.get("actions") if isinstance(rule.get("actions"), list) else []
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("type") or "").strip().lower()
            if not action_type:
                continue

            handler = self.action_handlers.get(action_type)
            if not handler:
                continue

            if self.dry_run:
                logger.info("Dry-run: would execute action=%s for rule=%s camera=%s", action_type, rule_id, evt.camera_id)
                continue
            try:
                handler(rule=rule, ctx=ctx, details=details, action=action, event=evt)
            except Exception as e:
                logger.warning("Action handler failed type=%s rule=%s: %s", action_type, rule_id, e)


