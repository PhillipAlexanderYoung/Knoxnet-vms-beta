import time
import heapq
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple, Callable


@dataclass(order=True)
class EventTask:
    sort_index: Tuple[int, float] = field(init=False, repr=False)
    priority: int
    created_at: float
    camera_id: str
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    not_before: Optional[float] = None

    def __post_init__(self):
        # Lower priority number = higher priority in heap
        ts = self.not_before if self.not_before is not None else self.created_at
        self.sort_index = (self.priority, ts)


@dataclass
class ROIItem:
    camera_id: str
    rect: Tuple[int, int, int, int]  # x, y, w, h
    timestamp: float


class Scheduler:
    """
    Multi-camera scheduler with global priority queue, cooldowns, ROI batching, and quota monitoring.
    """

    def __init__(self,
                 max_queue_size: int = 10000,
                 roi_batch_size: int = 8,
                 roi_batch_timeout_s: float = 0.25):
        self.max_queue_size = max_queue_size
        self._lock = threading.Lock()

        # Global priority queue (min-heap on (priority, time))
        self._global_heap: List[EventTask] = []

        # Per-camera queues (counts only for visibility)
        self._camera_counts: Dict[str, int] = {}

        # Cooldowns: key -> next_allowed_time
        self._cooldowns: Dict[str, float] = {}

        # Quotas: key -> (window_seconds, max_count, [timestamps])
        self._quotas: Dict[str, Tuple[float, int, List[float]]] = {}

        # ROI batching per camera
        self._roi_buffers: Dict[str, List[ROIItem]] = {}
        self._roi_last_flush: Dict[str, float] = {}
        self.roi_batch_size = roi_batch_size
        self.roi_batch_timeout_s = roi_batch_timeout_s

    # ==================== QUEUE ====================

    def enqueue(self, task: EventTask) -> bool:
        with self._lock:
            if len(self._global_heap) >= self.max_queue_size:
                return False
            heapq.heappush(self._global_heap, task)
            self._camera_counts[task.camera_id] = self._camera_counts.get(task.camera_id, 0) + 1
            return True

    def get_next_task(self) -> Optional[EventTask]:
        now = time.time()
        with self._lock:
            while self._global_heap:
                task = heapq.heappop(self._global_heap)

                # Not-before scheduling
                if task.not_before and task.not_before > now:
                    # Put back with same sort index
                    heapq.heappush(self._global_heap, task)
                    break

                # Cooldown check
                cd_key = self._cooldown_key(task.camera_id, task.kind)
                next_allowed = self._cooldowns.get(cd_key)
                if next_allowed and next_allowed > now:
                    # Defer a little
                    task.not_before = next_allowed
                    task.sort_index = (task.priority, task.not_before)
                    heapq.heappush(self._global_heap, task)
                    continue

                # Quota check
                if not self._check_and_consume_quota(self._quota_key(task.camera_id, task.kind), now):
                    # Push back slightly later when quota may free up
                    task.not_before = now + 0.5
                    task.sort_index = (task.priority, task.not_before)
                    heapq.heappush(self._global_heap, task)
                    continue

                self._camera_counts[task.camera_id] = max(0, self._camera_counts.get(task.camera_id, 1) - 1)
                return task
        return None

    # ==================== COOLDOWNS ====================

    def set_cooldown(self, camera_id: str, kind: str, seconds: float) -> None:
        with self._lock:
            self._cooldowns[self._cooldown_key(camera_id, kind)] = time.time() + seconds

    def _cooldown_key(self, camera_id: str, kind: str) -> str:
        return f"{camera_id}:{kind}:cooldown"

    # ==================== QUOTAS ====================

    def configure_quota(self, camera_id: str, kind: str, window_seconds: float, max_count: int) -> None:
        key = self._quota_key(camera_id, kind)
        with self._lock:
            self._quotas[key] = (window_seconds, max_count, [])

    def _quota_key(self, camera_id: str, kind: str) -> str:
        return f"{camera_id}:{kind}:quota"

    def _check_and_consume_quota(self, key: str, now: float) -> bool:
        if key not in self._quotas:
            return True
        window, max_count, stamps = self._quotas[key]
        # drop old timestamps
        cutoff = now - window
        while stamps and stamps[0] < cutoff:
            stamps.pop(0)
        if len(stamps) >= max_count:
            return False
        stamps.append(now)
        self._quotas[key] = (window, max_count, stamps)
        return True

    # ==================== ROI BATCHING ====================

    def add_roi(self, camera_id: str, rect: Tuple[int, int, int, int]) -> None:
        now = time.time()
        with self._lock:
            buf = self._roi_buffers.setdefault(camera_id, [])
            buf.append(ROIItem(camera_id=camera_id, rect=rect, timestamp=now))
            self._roi_last_flush.setdefault(camera_id, now)

            if len(buf) >= self.roi_batch_size or (now - self._roi_last_flush[camera_id]) >= self.roi_batch_timeout_s:
                self._flush_roi_locked(camera_id)

    def flush_all_roi(self) -> None:
        with self._lock:
            for camera_id in list(self._roi_buffers.keys()):
                if self._roi_buffers[camera_id]:
                    self._flush_roi_locked(camera_id)

    def _flush_roi_locked(self, camera_id: str) -> None:
        batch = self._roi_buffers.get(camera_id, [])
        if not batch:
            return
        # Create a combined ROI task (payload contains list of ROIs)
        payload = {"rois": [dict(x=r.rect[0], y=r.rect[1], w=r.rect[2], h=r.rect[3], ts=r.timestamp) for r in batch]}
        task = EventTask(priority=5, created_at=time.time(), camera_id=camera_id, kind="roi_batch", payload=payload)
        heapq.heappush(self._global_heap, task)
        self._camera_counts[camera_id] = self._camera_counts.get(camera_id, 0) + 1
        self._roi_buffers[camera_id] = []
        self._roi_last_flush[camera_id] = time.time()

    # ==================== STATUS ====================

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            quotas = {}
            now = time.time()
            for key, (window, max_count, stamps) in self._quotas.items():
                cutoff = now - window
                current = len([s for s in stamps if s >= cutoff])
                quotas[key] = {"window": window, "max": max_count, "current": current}

            return {
                "queue_size": len(self._global_heap),
                "camera_counts": dict(self._camera_counts),
                "cooldowns": {k: max(0.0, v - now) for k, v in self._cooldowns.items()},
                "quotas": quotas,
                "roi_buffers": {k: len(v) for k, v in self._roi_buffers.items()},
            }


