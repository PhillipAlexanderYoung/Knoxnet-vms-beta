"""Self-protective load shedder for the Knoxnet desktop app.

Monitors host CPU / RAM / swap, classifies the host hardware tier at
startup, and decides on load-shed actions (FPS throttling, worker
suspension, widget release, recording stop) when the system is under
strain.  Recording is treated as the most important feature and is
only stopped at the EMERGENCY level, after every other lever has been
pulled.

Design constraints:
  * Pure data; no Qt or PySide imports.  The desktop driver is the
    only thing that touches widgets/timers/UI.
  * State machine has hysteresis (sustained-time gates on rise; even
    longer gates on fall) so transient spikes do not flap the level.
  * `update()` is idempotent: it can be called every poll and only
    returns *new* actions when the level actually changes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class LoadLevel(IntEnum):
    """Severity of host load.  Higher = more aggressive shedding."""

    NORMAL = 0
    ELEVATED = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4

    @property
    def label(self) -> str:
        return self.name.title() if self != LoadLevel.NORMAL else "Normal"


# Default per-level throttle ladder.  Values of 0 mean "stop the worker
# entirely" (used at CRITICAL for detectors / depth).
DEFAULT_THROTTLES: Dict[str, Dict[str, int]] = {
    "elevated": {"paint_fps": 10, "motion_fps": 10, "detector_fps": 4, "depth_fps": 2},
    "high":     {"paint_fps": 6,  "motion_fps": 5,  "detector_fps": 2, "depth_fps": 1},
    "critical": {"paint_fps": 4,  "motion_fps": 3,  "detector_fps": 0, "depth_fps": 0},
}


# Default thresholds keyed by machine class.  These values were chosen
# to keep weaker hosts comfortable (they trip at lower utilization)
# while not nagging powerful boxes that legitimately run hot.
DEFAULT_THRESHOLDS_BY_CLASS: Dict[str, Dict[str, int]] = {
    "low": {
        "elevated_cpu": 80, "elevated_ram": 82,
        "high_cpu": 90,     "high_ram": 90,
        "critical_cpu": 95, "critical_ram": 94, "critical_swap": 30,
        "emergency_ram": 97, "emergency_swap": 60,
    },
    "mid": {
        "elevated_cpu": 85, "elevated_ram": 85,
        "high_cpu": 92,     "high_ram": 92,
        "critical_cpu": 96, "critical_ram": 95, "critical_swap": 40,
        "emergency_ram": 98, "emergency_swap": 70,
    },
    "high": {
        "elevated_cpu": 92, "elevated_ram": 90,
        "high_cpu": 96,     "high_ram": 94,
        "critical_cpu": 98, "critical_ram": 97, "critical_swap": 55,
        "emergency_ram": 99, "emergency_swap": 80,
    },
}

# How long sustained CPU/RAM pressure must be observed before the
# state machine escalates.  Combined with the per-tick poll cadence
# (2 s) this means a brief spike under e.g. 30 s never shifts level.
# Recovery uses similar gates so the level only drops once load has
# *actually* eased.


# How long the GUI thread must be stuck before we treat it as a hang.
# Legitimate operations (camera sync, layout load, large dialog open,
# DB writes during recording start) routinely take 5-10s on slower
# hosts; only react to genuine hangs that exceed this floor.
STUCK_LOOP_THRESHOLD_SEC: float = 15.0


# How long a condition must persist before we escalate to a higher
# level.  Recovery uses similar (slightly longer) gates.
RISE_GATES_SEC: Dict[LoadLevel, float] = {
    LoadLevel.ELEVATED: 30.0,
    LoadLevel.HIGH: 30.0,
    LoadLevel.CRITICAL: 60.0,
    LoadLevel.EMERGENCY: 0.0,   # immediate
}
FALL_GATES_SEC: Dict[LoadLevel, float] = {
    LoadLevel.ELEVATED: 30.0,
    LoadLevel.HIGH: 30.0,
    LoadLevel.CRITICAL: 30.0,
    LoadLevel.EMERGENCY: 60.0,
}


@dataclass
class SystemMetrics:
    """Snapshot of host load. Populated by the desktop driver via psutil."""

    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    swap_percent: float = 0.0
    event_loop_stuck_sec: float = 0.0  # 0 when GUI thread is responsive


@dataclass
class ShedDecision:
    """Result of a single LoadShedder.update() call.

    `level_changed` is True only on the tick where a new level is
    entered/left, so the driver can perform one-shot actions (e.g.
    log an event, push a debug-overlay reason) without spamming.
    """

    level: LoadLevel
    previous_level: LoadLevel
    level_changed: bool
    throttles: Dict[str, int]              # active throttle values for the current level
    reason: str                            # short human-readable explanation
    metrics: SystemMetrics                 # carry the metrics that drove the decision


# ---------------------------------------------------------------------------
# Hardware classification
# ---------------------------------------------------------------------------


def detect_machine_class(
    cpu_count: Optional[int] = None,
    ram_gb: Optional[float] = None,
) -> str:
    """Classify the host as 'low', 'mid', or 'high' based on CPU + RAM.

    If `cpu_count` or `ram_gb` are not provided, attempts to autodetect
    via psutil; falls back to 'mid' if psutil is unavailable.
    """
    try:
        if cpu_count is None or ram_gb is None:
            import psutil
            if cpu_count is None:
                cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 4
            if ram_gb is None:
                ram_gb = float(psutil.virtual_memory().total) / (1024.0 ** 3)
    except Exception:
        return "mid"

    cpu_count = max(1, int(cpu_count or 4))
    ram_gb = float(ram_gb or 8.0)

    if cpu_count <= 4 or ram_gb <= 8.0:
        return "low"
    if cpu_count >= 8 and ram_gb >= 16.0:
        return "high"
    return "mid"


# ---------------------------------------------------------------------------
# Core state machine
# ---------------------------------------------------------------------------


class LoadShedder:
    """State machine that maps system metrics to a LoadLevel + actions.

    The driver calls `update(metrics)` on each poll and applies the
    returned decision.  The shedder remembers when each candidate
    level was first reached so it can enforce sustained-time gates.
    """

    def __init__(
        self,
        prefs: Optional[dict] = None,
        machine_class: Optional[str] = None,
    ):
        self._machine_class = machine_class or detect_machine_class()
        self._prefs_block: dict = self._normalize_prefs(prefs or {})

        self._level: LoadLevel = LoadLevel.NORMAL
        self._candidate_level: LoadLevel = LoadLevel.NORMAL
        self._candidate_since: float = 0.0
        self._last_change_ts: float = time.time()
        self._last_reason: str = "Normal load"
        # Kept for backwards-compat with callers that read this; stall
        # detection no longer drives level escalation so this is always
        # False under the current logic.
        self._emergency_was_stall_only: bool = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _normalize_prefs(self, prefs: dict) -> dict:
        """Pull the auto_protect block out of a desktop_prefs dict and
        apply any user overrides on top of the auto-tier defaults."""
        block = dict(prefs.get("auto_protect") or {})
        machine_pref = str(block.get("machine_class") or "auto")
        if machine_pref != "auto":
            self._machine_class = machine_pref

        tier_thresholds = dict(
            DEFAULT_THRESHOLDS_BY_CLASS.get(
                self._machine_class, DEFAULT_THRESHOLDS_BY_CLASS["mid"]
            )
        )

        # User overrides on top of tier defaults
        user_thresh = block.get("thresholds") or {}
        if isinstance(user_thresh, dict):
            for k, v in user_thresh.items():
                try:
                    tier_thresholds[k] = int(v)
                except Exception:
                    continue

        # Throttles: deep-copy defaults then overlay user values
        throttles = {k: dict(v) for k, v in DEFAULT_THROTTLES.items()}
        user_throttles = block.get("throttles") or {}
        if isinstance(user_throttles, dict):
            for level_key, level_vals in user_throttles.items():
                if not isinstance(level_vals, dict):
                    continue
                if level_key not in throttles:
                    continue
                for k, v in level_vals.items():
                    try:
                        throttles[level_key][k] = max(0, int(v))
                    except Exception:
                        continue

        return {
            "enabled": bool(block.get("enabled", True)),
            "protect_recording": bool(block.get("protect_recording", True)),
            "thresholds": tier_thresholds,
            "throttles": throttles,
            "primary_widget_count": int(block.get("primary_widget_count", 2) or 2),
            "exit_on_emergency_after_sec": int(
                block.get("exit_on_emergency_after_sec", 60) or 60
            ),
            "show_overlay": bool(block.get("show_overlay", True)),
            "machine_class": self._machine_class,
        }

    def reload_prefs(self, prefs: dict) -> None:
        """Re-read prefs (after the user changes sliders, etc.)."""
        self._prefs_block = self._normalize_prefs(prefs or {})

    @property
    def enabled(self) -> bool:
        return bool(self._prefs_block.get("enabled", True))

    @property
    def protect_recording(self) -> bool:
        return bool(self._prefs_block.get("protect_recording", True))

    @property
    def machine_class(self) -> str:
        return str(self._prefs_block.get("machine_class") or "mid")

    @property
    def thresholds(self) -> Dict[str, int]:
        return dict(self._prefs_block.get("thresholds") or {})

    @property
    def throttles(self) -> Dict[str, Dict[str, int]]:
        return {k: dict(v) for k, v in (self._prefs_block.get("throttles") or {}).items()}

    @property
    def primary_widget_count(self) -> int:
        return int(self._prefs_block.get("primary_widget_count", 2) or 2)

    @property
    def exit_on_emergency_after_sec(self) -> int:
        return int(self._prefs_block.get("exit_on_emergency_after_sec", 60) or 60)

    @property
    def current_level(self) -> LoadLevel:
        return self._level

    @property
    def candidate_level(self) -> LoadLevel:
        """The level the *current* metrics warrant, before sustained-time
        gates.  Exposed so UI / diagnostics can show 'evaluating: HIGH
        (25s to commit)' instead of being silent during the gate."""
        return self._candidate_level

    @property
    def candidate_since(self) -> float:
        """Wall-clock timestamp when the current candidate became active."""
        return self._candidate_since

    def candidate_gate_remaining(self) -> float:
        """Seconds until the candidate would actually commit and change
        the level.  Returns 0 when the candidate matches the committed
        level (nothing pending) or when the gate is already met.
        """
        if self._candidate_level == self._level:
            return 0.0
        rising = self._candidate_level > self._level
        if rising:
            gate = RISE_GATES_SEC.get(self._candidate_level, 30.0)
        else:
            gate = FALL_GATES_SEC.get(self._level, 30.0)
        elapsed = time.time() - self._candidate_since
        return max(0.0, gate - elapsed)

    @property
    def last_reason(self) -> str:
        return self._last_reason

    @property
    def emergency_was_stall_only(self) -> bool:
        """True when the current EMERGENCY level was triggered only by
        a transient GUI stall (no memory pressure).  The driver checks
        this before deciding to auto-quit -- a brief hang during
        legitimate work should not kill the app.
        """
        return bool(self._emergency_was_stall_only)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_target_level(self, m: SystemMetrics) -> Tuple[LoadLevel, str]:
        """Decide what level the metrics *currently* warrant, ignoring
        sustained-time gates.  Returns (level, short reason).

        IMPORTANT: stall detection is intentionally NOT used to escalate
        the shed level.  A brief GUI hang during normal operations
        (camera sync, layout load, dialog open, file write) would
        otherwise turn into aggressive throttling that makes the system
        feel WORSE than the original hang, with no underlying resource
        problem to justify it.  The driver still logs long stalls so
        they remain visible for diagnostics, but they don't drive the
        state machine.
        """
        t = self.thresholds

        # EMERGENCY — only triggered by genuine resource exhaustion,
        # not by transient stalls.
        if m.ram_percent >= t.get("emergency_ram", 98):
            return LoadLevel.EMERGENCY, f"RAM {m.ram_percent:.0f}%"
        if m.swap_percent >= t.get("emergency_swap", 65):
            return LoadLevel.EMERGENCY, f"Swap {m.swap_percent:.0f}%"

        # CRITICAL
        if (
            m.cpu_percent >= t.get("critical_cpu", 92)
            or m.ram_percent >= t.get("critical_ram", 90)
            or m.swap_percent >= t.get("critical_swap", 25)
        ):
            parts = []
            if m.cpu_percent >= t.get("critical_cpu", 92):
                parts.append(f"CPU {m.cpu_percent:.0f}%")
            if m.ram_percent >= t.get("critical_ram", 90):
                parts.append(f"RAM {m.ram_percent:.0f}%")
            if m.swap_percent >= t.get("critical_swap", 25):
                parts.append(f"Swap {m.swap_percent:.0f}%")
            return LoadLevel.CRITICAL, " · ".join(parts) or "critical load"

        # HIGH
        if (
            m.cpu_percent >= t.get("high_cpu", 85)
            or m.ram_percent >= t.get("high_ram", 85)
        ):
            parts = []
            if m.cpu_percent >= t.get("high_cpu", 85):
                parts.append(f"CPU {m.cpu_percent:.0f}%")
            if m.ram_percent >= t.get("high_ram", 85):
                parts.append(f"RAM {m.ram_percent:.0f}%")
            return LoadLevel.HIGH, " · ".join(parts) or "high load"

        # ELEVATED
        if (
            m.cpu_percent >= t.get("elevated_cpu", 75)
            or m.ram_percent >= t.get("elevated_ram", 75)
        ):
            parts = []
            if m.cpu_percent >= t.get("elevated_cpu", 75):
                parts.append(f"CPU {m.cpu_percent:.0f}%")
            if m.ram_percent >= t.get("elevated_ram", 75):
                parts.append(f"RAM {m.ram_percent:.0f}%")
            return LoadLevel.ELEVATED, " · ".join(parts) or "elevated load"

        return LoadLevel.NORMAL, "Normal load"

    def update(self, metrics: SystemMetrics) -> ShedDecision:
        """Apply the metrics to the state machine and return a decision.

        The level only changes when the candidate has been sustained
        beyond the rise/fall gate for that direction — except EMERGENCY
        which fires immediately when triggered by genuine resource
        exhaustion.

        Stall detection (event_loop_stuck_sec) is intentionally NOT a
        level-driver here: a brief GUI hang during legitimate work
        (camera sync, layout load, dialog open) should never make the
        app worse by aggressively throttling features.  The driver
        logs long stalls separately for diagnostics.
        """
        now = time.time()

        if not self.enabled:
            previous = self._level
            self._level = LoadLevel.NORMAL
            self._candidate_level = LoadLevel.NORMAL
            self._candidate_since = now
            return ShedDecision(
                level=self._level,
                previous_level=previous,
                level_changed=(previous != self._level),
                throttles={},
                reason="Auto-protection disabled",
                metrics=metrics,
            )

        target, target_reason = self._evaluate_target_level(metrics)

        if target != self._candidate_level:
            self._candidate_level = target
            self._candidate_since = now

        rising = target > self._level
        falling = target < self._level
        elapsed = now - self._candidate_since
        commit = False

        if target == LoadLevel.EMERGENCY and rising:
            commit = True
        elif rising:
            gate = RISE_GATES_SEC.get(target, 30.0)
            if elapsed >= gate:
                commit = True
        elif falling:
            # Drop only one level at a time so we don't flip from
            # CRITICAL straight back to NORMAL after a single quiet tick.
            one_step_target = LoadLevel(int(self._level) - 1)
            gate = FALL_GATES_SEC.get(self._level, 30.0)
            if elapsed >= gate:
                commit_target = max(one_step_target, target, key=int)
                if commit_target != self._level:
                    target = commit_target
                    commit = True

        previous = self._level
        if commit:
            self._level = target
            self._last_change_ts = now
            self._last_reason = target_reason

        active_throttles: Dict[str, int] = {}
        if self._level >= LoadLevel.ELEVATED:
            key = self._level.name.lower()
            # CRITICAL inherits the table for "critical"; EMERGENCY uses
            # the same throttle row as CRITICAL since it's already the
            # most aggressive setting.
            tbl_key = key if key in self._prefs_block["throttles"] else "critical"
            active_throttles = dict(self._prefs_block["throttles"].get(tbl_key, {}))

        return ShedDecision(
            level=self._level,
            previous_level=previous,
            level_changed=(previous != self._level),
            throttles=active_throttles,
            reason=self._last_reason,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Helpers for the driver
    # ------------------------------------------------------------------

    def get_throttles_for_level(self, level: LoadLevel) -> Dict[str, int]:
        """Return the throttle table for a specific level (used by UI
        previews of what each level will do)."""
        if level <= LoadLevel.NORMAL:
            return {}
        key = level.name.lower()
        tbl_key = key if key in self._prefs_block["throttles"] else "critical"
        return dict(self._prefs_block["throttles"].get(tbl_key, {}))

    def describe_level(self, level: LoadLevel) -> str:
        """Short human-readable phrase summarizing what the given
        level does, for tooltips / banners."""
        if level == LoadLevel.NORMAL:
            return "All features running normally."
        if level == LoadLevel.ELEVATED:
            return "Live views and AI throttled to keep system responsive."
        if level == LoadLevel.HIGH:
            return "Substream forced, AI features reduced. Recording is still on."
        if level == LoadLevel.CRITICAL:
            return "Live views minimized, depth and object detection paused. Recording is still on."
        if level == LoadLevel.EMERGENCY:
            if self.protect_recording:
                return "EMERGENCY — recording stopped to prevent system crash."
            return "EMERGENCY — most features stopped to prevent system crash."
        return ""


# ---------------------------------------------------------------------------
# Event log helper (used by the UI panel)
# ---------------------------------------------------------------------------


@dataclass
class ShedEvent:
    """A single transition recorded for display in the System Manager."""

    ts: float
    from_level: LoadLevel
    to_level: LoadLevel
    reason: str
    summary: str = ""

    def format_line(self) -> str:
        ts_str = time.strftime("%H:%M:%S", time.localtime(self.ts))
        arrow = ">"
        return (
            f"{ts_str}  {self.from_level.label} {arrow} {self.to_level.label}  "
            f"({self.reason})"
            + (f" — {self.summary}" if self.summary else "")
        )


class ShedEventLog:
    """Bounded ring of recent ShedEvents (default keeps last 20)."""

    def __init__(self, capacity: int = 20):
        self._cap = max(1, int(capacity))
        self._events: List[ShedEvent] = []

    def append(self, event: ShedEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._cap:
            self._events = self._events[-self._cap:]

    def recent(self, n: int = 5) -> List[ShedEvent]:
        return list(self._events[-int(max(1, n)):])

    def clear(self) -> None:
        self._events.clear()
