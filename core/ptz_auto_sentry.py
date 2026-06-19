"""
PTZ auto-sentry tracking primitives.

This module is deliberately hardware-free: it normalizes detector/motion
payloads, keeps a smoothed target estimate, and produces bounded pan/tilt
velocity commands for PTZManager to send through the active controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


@dataclass
class AutoSentryConfig:
    target_classes: List[str] = field(default_factory=lambda: [
        "person", "car", "truck", "bus", "motorcycle", "bicycle", "dog",
    ])
    min_confidence: float = 0.35
    track_motion: bool = True
    motion_min_score: float = 0.08
    deadzone_x: float = 0.075
    deadzone_y: float = 0.090
    smoothing_factor: float = 0.35
    prediction_seconds: float = 0.20
    proportional_gain_pan: float = 1.15
    proportional_gain_tilt: float = 0.95
    max_speed: float = 0.42
    min_speed: float = 0.035
    max_accel_per_sec: float = 1.20
    update_interval: float = 0.18
    command_duration: float = 0.16
    command_cooldown: float = 0.10
    lost_target_ttl: float = 1.35
    reacquire_ttl: float = 4.0
    manual_pause_seconds: float = 3.0
    scan_enabled: bool = True
    scan_speed: float = 0.16
    scan_tilt_speed: float = 0.0
    scan_period: float = 7.5
    scan_after_seconds: float = 1.2
    audio_search_seconds: float = 8.0
    audio_search_speed: float = 0.22
    lock_iou_gate: float = 0.10
    lock_center_gate: float = 0.22
    lock_class_bonus: float = 0.18

    @classmethod
    def from_params(cls, params: Optional[Dict[str, Any]]) -> "AutoSentryConfig":
        cfg = cls()
        if not isinstance(params, dict):
            return cfg

        aliases = {
            "classes": "target_classes",
            "targetClasses": "target_classes",
            "minConfidence": "min_confidence",
            "trackMotion": "track_motion",
            "motionMinScore": "motion_min_score",
            "deadzoneX": "deadzone_x",
            "deadzoneY": "deadzone_y",
            "deadzone": ("deadzone_x", "deadzone_y"),
            "smoothing": "smoothing_factor",
            "smoothingFactor": "smoothing_factor",
            "predictionSeconds": "prediction_seconds",
            "maxSpeed": "max_speed",
            "minSpeed": "min_speed",
            "maxAccelPerSec": "max_accel_per_sec",
            "updateInterval": "update_interval",
            "commandDuration": "command_duration",
            "commandCooldown": "command_cooldown",
            "lostTargetTtl": "lost_target_ttl",
            "lost_target_ttl_sec": "lost_target_ttl",
            "manualPauseSeconds": "manual_pause_seconds",
            "scanEnabled": "scan_enabled",
            "scanSpeed": "scan_speed",
            "scanTiltSpeed": "scan_tilt_speed",
            "scanPeriod": "scan_period",
            "scanAfterSeconds": "scan_after_seconds",
            "audioSearchSeconds": "audio_search_seconds",
            "audioSearchSpeed": "audio_search_speed",
            "lockIouGate": "lock_iou_gate",
            "lockCenterGate": "lock_center_gate",
            "lockClassBonus": "lock_class_bonus",
        }

        for raw_key, raw_value in params.items():
            key = aliases.get(raw_key, raw_key)
            if isinstance(key, tuple):
                for k in key:
                    setattr(cfg, k, _clamp(_as_float(raw_value, getattr(cfg, k)), 0.0, 0.45))
                continue
            if not hasattr(cfg, str(key)):
                continue
            if key == "target_classes":
                if isinstance(raw_value, str):
                    classes = [c.strip().lower() for c in raw_value.split(",") if c.strip()]
                elif isinstance(raw_value, Sequence):
                    classes = [str(c).strip().lower() for c in raw_value if str(c).strip()]
                else:
                    classes = []
                cfg.target_classes = classes
            elif isinstance(getattr(cfg, key), bool):
                setattr(cfg, key, bool(raw_value))
            else:
                setattr(cfg, key, _as_float(raw_value, getattr(cfg, key)))

        cfg.min_confidence = _clamp(cfg.min_confidence, 0.0, 1.0)
        cfg.motion_min_score = _clamp(cfg.motion_min_score, 0.0, 1.0)
        cfg.deadzone_x = _clamp(cfg.deadzone_x, 0.0, 0.45)
        cfg.deadzone_y = _clamp(cfg.deadzone_y, 0.0, 0.45)
        cfg.smoothing_factor = _clamp(cfg.smoothing_factor, 0.01, 1.0)
        cfg.prediction_seconds = _clamp(cfg.prediction_seconds, 0.0, 1.0)
        cfg.max_speed = _clamp(cfg.max_speed, 0.02, 1.0)
        cfg.min_speed = _clamp(cfg.min_speed, 0.0, cfg.max_speed)
        cfg.max_accel_per_sec = _clamp(cfg.max_accel_per_sec, 0.05, 10.0)
        cfg.update_interval = _clamp(cfg.update_interval, 0.08, 2.0)
        cfg.command_duration = _clamp(cfg.command_duration, 0.08, 2.0)
        cfg.command_cooldown = _clamp(cfg.command_cooldown, 0.02, 2.0)
        cfg.lost_target_ttl = _clamp(cfg.lost_target_ttl, 0.2, 10.0)
        cfg.reacquire_ttl = _clamp(cfg.reacquire_ttl, cfg.lost_target_ttl, 30.0)
        cfg.manual_pause_seconds = _clamp(cfg.manual_pause_seconds, 0.0, 30.0)
        cfg.scan_speed = _clamp(cfg.scan_speed, 0.0, cfg.max_speed)
        cfg.scan_tilt_speed = _clamp(cfg.scan_tilt_speed, -cfg.max_speed, cfg.max_speed)
        cfg.scan_period = _clamp(cfg.scan_period, 1.0, 120.0)
        cfg.scan_after_seconds = _clamp(cfg.scan_after_seconds, 0.0, 30.0)
        cfg.audio_search_seconds = _clamp(cfg.audio_search_seconds, 1.0, 60.0)
        cfg.audio_search_speed = _clamp(cfg.audio_search_speed, 0.0, cfg.max_speed)
        cfg.lock_iou_gate = _clamp(cfg.lock_iou_gate, 0.0, 1.0)
        cfg.lock_center_gate = _clamp(cfg.lock_center_gate, 0.02, 1.0)
        cfg.lock_class_bonus = _clamp(cfg.lock_class_bonus, 0.0, 1.0)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_classes": list(self.target_classes),
            "min_confidence": self.min_confidence,
            "track_motion": self.track_motion,
            "motion_min_score": self.motion_min_score,
            "deadzone_x": self.deadzone_x,
            "deadzone_y": self.deadzone_y,
            "smoothing_factor": self.smoothing_factor,
            "prediction_seconds": self.prediction_seconds,
            "proportional_gain_pan": self.proportional_gain_pan,
            "proportional_gain_tilt": self.proportional_gain_tilt,
            "max_speed": self.max_speed,
            "min_speed": self.min_speed,
            "max_accel_per_sec": self.max_accel_per_sec,
            "update_interval": self.update_interval,
            "command_duration": self.command_duration,
            "command_cooldown": self.command_cooldown,
            "lost_target_ttl": self.lost_target_ttl,
            "manual_pause_seconds": self.manual_pause_seconds,
            "scan_enabled": self.scan_enabled,
            "scan_speed": self.scan_speed,
            "scan_tilt_speed": self.scan_tilt_speed,
            "scan_period": self.scan_period,
            "scan_after_seconds": self.scan_after_seconds,
            "audio_search_seconds": self.audio_search_seconds,
            "audio_search_speed": self.audio_search_speed,
            "lock_iou_gate": self.lock_iou_gate,
            "lock_center_gate": self.lock_center_gate,
            "lock_class_bonus": self.lock_class_bonus,
        }


@dataclass
class TargetObservation:
    target_id: str
    cls: str
    confidence: float
    nx: float
    ny: float
    area: float
    source: str
    ts: float
    vx: float = 0.0
    vy: float = 0.0
    bbox: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackingCommand:
    pan_speed: float = 0.0
    tilt_speed: float = 0.0
    mode: str = "idle"
    reason: str = ""
    target: Optional[Dict[str, Any]] = None
    should_send: bool = True


def _frame_dims(payload: Dict[str, Any]) -> Tuple[int, int]:
    motion = payload.get("motion") if isinstance(payload.get("motion"), dict) else {}
    fw = _as_int(payload.get("frame_width") or payload.get("frame_w") or motion.get("frame_width"), 0)
    fh = _as_int(payload.get("frame_height") or payload.get("frame_h") or motion.get("frame_height"), 0)
    return fw, fh


def _center_from_obj(obj: Dict[str, Any], fw: int, fh: int) -> Optional[Tuple[float, float, Dict[str, Any], float]]:
    center = obj.get("center") if isinstance(obj.get("center"), dict) else {}
    if center:
        nx = _as_float(center.get("nx", center.get("x")), -1.0)
        ny = _as_float(center.get("ny", center.get("y")), -1.0)
        if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
            return nx, ny, {}, 0.0

    bbox = obj.get("bbox") or obj.get("box") or {}
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        bbox = {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}
    if not isinstance(bbox, dict):
        return None
    x = _as_float(bbox.get("x"), 0.0)
    y = _as_float(bbox.get("y"), 0.0)
    w = _as_float(bbox.get("w", bbox.get("width")), 0.0)
    h = _as_float(bbox.get("h", bbox.get("height")), 0.0)
    local_fw = _as_int(bbox.get("frame_width") or bbox.get("frame_w"), fw)
    local_fh = _as_int(bbox.get("frame_height") or bbox.get("frame_h"), fh)
    if local_fw <= 0 or local_fh <= 0 or w <= 0 or h <= 0:
        return None
    nx = _clamp((x + w / 2.0) / float(local_fw), 0.0, 1.0)
    ny = _clamp((y + h / 2.0) / float(local_fh), 0.0, 1.0)
    area = _clamp((w * h) / float(max(1, local_fw * local_fh)), 0.0, 1.0)
    return nx, ny, dict(bbox), area


def observations_from_payload(payload: Dict[str, Any], source: str = "detections") -> List[TargetObservation]:
    if not isinstance(payload, dict):
        return []
    now = time.time()
    fw, fh = _frame_dims(payload)
    motion = payload.get("motion") if isinstance(payload.get("motion"), dict) else {}
    items: List[Tuple[str, Dict[str, Any]]] = []

    for key in ("target", "locked_target", "clicked_target", "object"):
        if isinstance(payload.get(key), dict):
            items.append(("locked", payload[key]))
    if isinstance(payload.get("tracks"), list):
        items.extend(("track", obj) for obj in payload.get("tracks") or [] if isinstance(obj, dict))
    if isinstance(payload.get("detections"), list):
        items.extend(("detection", obj) for obj in payload.get("detections") or [] if isinstance(obj, dict))
    if isinstance(motion.get("tracks"), list):
        items.extend(("motion_track", obj) for obj in motion.get("tracks") or [] if isinstance(obj, dict))
    if isinstance(motion.get("regions"), list):
        score = _as_float(motion.get("score"), 0.0)
        for idx, region in enumerate(motion.get("regions") or []):
            if isinstance(region, dict):
                obj = {"bbox": region, "class": "motion", "confidence": score, "id": f"region-{idx}"}
                items.append(("motion_region", obj))

    out: List[TargetObservation] = []
    for idx, (kind, obj) in enumerate(items):
        center = _center_from_obj(obj, fw, fh)
        if center is None:
            continue
        nx, ny, bbox, area = center
        if bbox and fw > 0 and fh > 0:
            bbox.setdefault("frame_width", fw)
            bbox.setdefault("frame_height", fh)
        velocity = obj.get("velocity") if isinstance(obj.get("velocity"), dict) else {}
        cls = str(obj.get("class") or obj.get("class_name") or obj.get("label") or "object").strip().lower()
        conf = _as_float(obj.get("confidence"), 0.99 if "track" in kind else 0.0)
        tid = str(obj.get("id") or obj.get("track_id") or obj.get("detection_id") or f"{kind}-{idx}")
        out.append(TargetObservation(
            target_id=f"{kind}:{tid}",
            cls=cls,
            confidence=_clamp(conf, 0.0, 1.0),
            nx=nx,
            ny=ny,
            area=area,
            source=source or kind,
            ts=now,
            vx=_as_float(velocity.get("vx_norm_per_sec"), _as_float(velocity.get("vx"), 0.0)),
            vy=_as_float(velocity.get("vy_norm_per_sec"), _as_float(velocity.get("vy"), 0.0)),
            bbox=bbox,
        ))
    return out


class SmoothTargetTracker:
    def __init__(self) -> None:
        self.target_id: Optional[str] = None
        self.cls: str = ""
        self.confidence: float = 0.0
        self.nx: float = 0.5
        self.ny: float = 0.5
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.area: float = 0.0
        self.source: str = ""
        self.bbox: Dict[str, Any] = {}
        self.locked: bool = False
        self.lock_label: str = ""
        self.locked_at: float = 0.0
        self.last_confirmed_ts: float = 0.0
        self.last_seen: float = 0.0
        self.last_update: float = 0.0
        self.last_command_ts: float = 0.0
        self.last_pan_speed: float = 0.0
        self.last_tilt_speed: float = 0.0
        self.scan_direction: int = 1
        self.last_scan_flip: float = time.time()
        self.last_mode: str = "idle"

    def ingest(self, observations: Iterable[TargetObservation], cfg: AutoSentryConfig) -> Optional[TargetObservation]:
        obs = [o for o in observations if self._allowed(o, cfg)]
        if not obs:
            return None
        selected = self._select(obs)
        self._apply(selected, cfg)
        return selected

    def force_scan(self, direction: Optional[int] = None) -> None:
        if direction is not None:
            self.scan_direction = -1 if direction < 0 else 1
        self.last_scan_flip = 0.0
        self.last_seen = 0.0
        if not self.locked:
            self.target_id = None

    def lock(self, observation: TargetObservation, cfg: AutoSentryConfig, label: str = "") -> None:
        self.locked = True
        self.lock_label = label or observation.cls or observation.target_id
        self.locked_at = observation.ts
        self._apply(observation, cfg, force=True)

    def unlock(self) -> None:
        self.locked = False
        self.lock_label = ""
        self.locked_at = 0.0
        self.target_id = None
        self.bbox = {}
        self.last_mode = "idle"

    def compute(self, cfg: AutoSentryConfig, now: Optional[float] = None, *, scanning_allowed: bool = True) -> TrackingCommand:
        now = time.time() if now is None else float(now)
        if now - self.last_command_ts < cfg.command_cooldown:
            return TrackingCommand(mode=self.last_mode, reason="rate_limited", should_send=False, target=self.target_dict())

        has_recent_target = self.target_id is not None and (now - self.last_seen) <= cfg.lost_target_ttl
        if has_recent_target:
            dt_lost = max(0.0, now - self.last_seen)
            pred_x = _clamp(self.nx + self.vx * min(cfg.prediction_seconds, dt_lost), 0.0, 1.0)
            pred_y = _clamp(self.ny + self.vy * min(cfg.prediction_seconds, dt_lost), 0.0, 1.0)
            err_x = pred_x - 0.5
            err_y = 0.5 - pred_y
            pan = self._axis_to_speed(err_x, cfg.deadzone_x, cfg.proportional_gain_pan, cfg)
            tilt = self._axis_to_speed(err_y, cfg.deadzone_y, cfg.proportional_gain_tilt, cfg)
            pan, tilt = self._ramp(pan, tilt, cfg, now)
            self.last_command_ts = now
            self.last_mode = "tracking"
            return TrackingCommand(
                pan_speed=pan,
                tilt_speed=tilt,
                mode="tracking",
                reason="target_locked",
                target=self.target_dict(pred_x=pred_x, pred_y=pred_y),
            )

        target_was_recent = self.last_seen > 0 and (now - self.last_seen) <= cfg.reacquire_ttl
        if target_was_recent and (now - self.last_seen) < cfg.scan_after_seconds:
            age = max(0.0, now - self.last_seen)
            pred_x = _clamp(self.nx + self.vx * min(age, cfg.prediction_seconds + 0.6), 0.0, 1.0)
            pred_y = _clamp(self.ny + self.vy * min(age, cfg.prediction_seconds + 0.6), 0.0, 1.0)
            pan = self._axis_to_speed(pred_x - 0.5, cfg.deadzone_x, cfg.proportional_gain_pan * 0.65, cfg)
            tilt = self._axis_to_speed(0.5 - pred_y, cfg.deadzone_y, cfg.proportional_gain_tilt * 0.55, cfg)
            pan, tilt = self._ramp(pan, tilt, cfg, now)
            self.last_command_ts = now
            self.last_mode = "predictive_search"
            return TrackingCommand(
                pan_speed=pan,
                tilt_speed=tilt,
                mode="predictive_search",
                reason="brief_dropout",
                target=self.target_dict(pred_x=pred_x, pred_y=pred_y, state="lost"),
            )

        if cfg.scan_enabled and scanning_allowed:
            if now - self.last_scan_flip >= cfg.scan_period:
                self.scan_direction *= -1
                self.last_scan_flip = now
            scan_speed = cfg.scan_speed * self.scan_direction
            pan, tilt = self._ramp(scan_speed, cfg.scan_tilt_speed, cfg, now)
            self.last_command_ts = now
            self.last_mode = "scan"
            return TrackingCommand(pan_speed=pan, tilt_speed=tilt, mode="scan", reason="searching", target=self.target_dict())

        pan, tilt = self._ramp(0.0, 0.0, cfg, now)
        self.last_command_ts = now
        self.last_mode = "idle"
        return TrackingCommand(pan_speed=pan, tilt_speed=tilt, mode="idle", reason="no_target", target=self.target_dict())

    def target_dict(self, *, pred_x: Optional[float] = None, pred_y: Optional[float] = None,
                    state: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.target_id is None:
            return None
        now = time.time()
        age = max(0.0, now - self.last_seen) if self.last_seen else 0.0
        target_state = state or ("tracking" if age <= 0.4 else ("lost" if age > 0 else "acquiring"))
        predicted_bbox = self._predicted_bbox(
            self.nx if pred_x is None else pred_x,
            self.ny if pred_y is None else pred_y,
        )
        return {
            "id": self.target_id,
            "class": self.cls,
            "label": self.lock_label or self.cls,
            "confidence": self.confidence,
            "state": target_state,
            "locked": self.locked,
            "locked_at": self.locked_at,
            "last_confirmed_at": self.last_confirmed_ts,
            "center": {"nx": self.nx, "ny": self.ny},
            "predicted_center": {
                "nx": self.nx if pred_x is None else pred_x,
                "ny": self.ny if pred_y is None else pred_y,
            },
            "bbox": dict(self.bbox),
            "predicted_bbox": predicted_bbox,
            "velocity": {"vx_norm_per_sec": self.vx, "vy_norm_per_sec": self.vy},
            "area": self.area,
            "source": self.source,
            "last_seen": self.last_seen,
            "lost_age": age,
        }

    def _allowed(self, obs: TargetObservation, cfg: AutoSentryConfig) -> bool:
        if obs.cls == "motion":
            return bool(cfg.track_motion) and obs.confidence >= cfg.motion_min_score
        allowed = [c.lower() for c in cfg.target_classes if c]
        if allowed and obs.cls not in allowed:
            return False
        return obs.confidence >= cfg.min_confidence

    def _select(self, observations: List[TargetObservation]) -> TargetObservation:
        if self.target_id:
            same_id = [obs for obs in observations if obs.target_id == self.target_id]
            if same_id:
                return same_id[0]
            reassociated = self._reassociate_locked(observations)
            if reassociated is not None:
                return reassociated

        def score(obs: TargetObservation) -> float:
            center_penalty = math.hypot(obs.nx - 0.5, obs.ny - 0.5) * 0.25
            source_bonus = 0.08 if obs.source in ("tracks", "detections") else 0.0
            class_bonus = 0.05 if obs.cls == "person" else 0.0
            return obs.confidence + min(0.20, obs.area * 2.0) + source_bonus + class_bonus - center_penalty

        return max(observations, key=score)

    def _reassociate_locked(self, observations: List[TargetObservation]) -> Optional[TargetObservation]:
        if not (self.locked or self.target_id):
            return None
        best: Optional[TargetObservation] = None
        best_score = -1.0
        for obs in observations:
            if self.cls and obs.cls and obs.cls != self.cls and obs.cls != "motion":
                continue
            iou = _bbox_iou(self.bbox, obs.bbox)
            dist = math.hypot(obs.nx - self.nx, obs.ny - self.ny)
            if iou < 0.10 and dist > 0.28:
                continue
            score = (iou * 1.8) + max(0.0, 1.0 - dist / 0.5) + obs.confidence
            if self.cls and obs.cls == self.cls:
                score += 0.18
            if score > best_score:
                best = obs
                best_score = score
        return best

    def _apply(self, obs: TargetObservation, cfg: AutoSentryConfig, *, force: bool = False) -> None:
        alpha = cfg.smoothing_factor
        now = obs.ts
        if force or self.target_id != obs.target_id or not self.last_seen:
            self.nx = obs.nx
            self.ny = obs.ny
            self.vx = obs.vx
            self.vy = obs.vy
        else:
            dt = max(1e-3, now - self.last_seen)
            measured_vx = obs.vx if abs(obs.vx) > 1e-6 else (obs.nx - self.nx) / dt
            measured_vy = obs.vy if abs(obs.vy) > 1e-6 else (obs.ny - self.ny) / dt
            self.nx = (1.0 - alpha) * self.nx + alpha * obs.nx
            self.ny = (1.0 - alpha) * self.ny + alpha * obs.ny
            self.vx = (1.0 - alpha) * self.vx + alpha * measured_vx
            self.vy = (1.0 - alpha) * self.vy + alpha * measured_vy
        if not self.locked or force or self.target_id is None:
            self.target_id = obs.target_id
        self.cls = obs.cls
        self.confidence = obs.confidence
        self.area = obs.area
        self.source = obs.source
        if obs.bbox:
            self.bbox = dict(obs.bbox)
        self.last_confirmed_ts = now
        self.last_seen = now
        self.last_update = now

    def _axis_to_speed(self, error: float, deadzone: float, gain: float, cfg: AutoSentryConfig) -> float:
        abs_err = abs(error)
        if abs_err <= deadzone:
            return 0.0
        usable = (abs_err - deadzone) / max(1e-6, 0.5 - deadzone)
        speed = _clamp(usable * gain * cfg.max_speed, 0.0, cfg.max_speed)
        if 0.0 < speed < cfg.min_speed:
            speed = cfg.min_speed
        return math.copysign(speed, error)

    def _ramp(self, pan: float, tilt: float, cfg: AutoSentryConfig, now: float) -> Tuple[float, float]:
        dt = max(cfg.command_cooldown, now - self.last_command_ts) if self.last_command_ts else cfg.update_interval
        max_delta = cfg.max_accel_per_sec * dt

        def step(prev: float, target: float) -> float:
            delta = _clamp(target - prev, -max_delta, max_delta)
            value = _clamp(prev + delta, -cfg.max_speed, cfg.max_speed)
            if abs(value) < cfg.min_speed and abs(target) < cfg.min_speed:
                return 0.0
            return value

        self.last_pan_speed = step(self.last_pan_speed, pan)
        self.last_tilt_speed = step(self.last_tilt_speed, tilt)
        return self.last_pan_speed, self.last_tilt_speed

    def _predicted_bbox(self, pred_x: float, pred_y: float) -> Dict[str, Any]:
        if not self.bbox:
            return {}
        out = dict(self.bbox)
        w = _as_float(out.get("w", out.get("width")), 0.0)
        h = _as_float(out.get("h", out.get("height")), 0.0)
        fw = _as_float(out.get("frame_width") or out.get("frame_w"), 0.0)
        fh = _as_float(out.get("frame_height") or out.get("frame_h"), 0.0)
        if w > 0 and h > 0 and fw > 0 and fh > 0:
            out["x"] = int(_clamp(pred_x * fw - w / 2.0, 0.0, max(0.0, fw - w)))
            out["y"] = int(_clamp(pred_y * fh - h / 2.0, 0.0, max(0.0, fh - h)))
            out["predicted"] = True
        return out


def _bbox_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    if not a or not b:
        return 0.0
    ax = _as_float(a.get("x"), 0.0)
    ay = _as_float(a.get("y"), 0.0)
    aw = _as_float(a.get("w", a.get("width")), 0.0)
    ah = _as_float(a.get("h", a.get("height")), 0.0)
    bx = _as_float(b.get("x"), 0.0)
    by = _as_float(b.get("y"), 0.0)
    bw = _as_float(b.get("w", b.get("width")), 0.0)
    bh = _as_float(b.get("h", b.get("height")), 0.0)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
