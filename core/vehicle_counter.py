import base64
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .motion import MotionRegion, SimpleMotionDetector
from .object_detector import ObjectDetector

logger = logging.getLogger(__name__)


@dataclass
class VehicleCandidate:
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: int
    timestamp: float
    confidence: float
    confirmation: str
    frame: np.ndarray
    normalized_bbox: Tuple[float, float, float, float]


@dataclass
class VehicleTrack:
    track_id: int
    created_at: float
    centroids: deque = field(default_factory=lambda: deque(maxlen=64))
    projections: deque = field(default_factory=lambda: deque(maxlen=64))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=64))
    confidences: deque = field(default_factory=lambda: deque(maxlen=64))
    confirmation_sources: deque = field(default_factory=lambda: deque(maxlen=64))
    first_bbox: Optional[Tuple[int, int, int, int]] = None
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    counted: bool = False
    direction: Optional[str] = None

    def add_observation(
        self,
        centroid: Tuple[float, float],
        projection: float,
        bbox: Tuple[int, int, int, int],
        timestamp: float,
        confidence: float,
        confirmation: str,
    ) -> None:
        if not self.first_bbox:
            self.first_bbox = bbox
        self.last_bbox = bbox
        self.centroids.append(centroid)
        self.projections.append(projection)
        self.timestamps.append(timestamp)
        self.confidences.append(confidence)
        self.confirmation_sources.append(confirmation)

    @property
    def best_confidence(self) -> float:
        return float(max(self.confidences) if self.confidences else 0.0)

    @property
    def primary_confirmation(self) -> Optional[str]:
        if not self.confirmation_sources:
            return None
        counts: Dict[str, int] = {}
        for source in self.confirmation_sources:
            counts[source] = counts.get(source, 0) + 1
        return max(counts.items(), key=lambda item: item[1])[0]

    def travel_distance(self) -> float:
        if len(self.projections) < 2:
            return 0.0
        return float(self.projections[-1] - self.projections[0])


class SimpleAssociationTracker:
    """
    Minimal centroid-based tracker suitable for low-footprint vehicle counting.
    """

    def __init__(self, max_distance: float = 90.0, max_idle_frames: int = 10) -> None:
        self.max_distance = float(max_distance)
        self.max_idle_frames = int(max_idle_frames)
        self._tracks: Dict[int, VehicleTrack] = {}
        self._idle_counts: Dict[int, int] = {}
        self._next_id = 1

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx * dx + dy * dy)

    def update(
        self,
        detections: List[VehicleCandidate],
        projection_fn: Any,
    ) -> Dict[int, VehicleTrack]:
        # Age existing tracks
        for track_id in list(self._tracks.keys()):
            self._idle_counts[track_id] = self._idle_counts.get(track_id, 0) + 1

        unmatched = list(range(len(detections)))

        for track_id, track in list(self._tracks.items()):
            best_idx = -1
            best_distance = self.max_distance
            last_centroid = track.centroids[-1] if track.centroids else None

            if not last_centroid:
                continue

            for idx in unmatched:
                detection = detections[idx]
                distance = self._distance(last_centroid, detection.centroid)
                if distance < best_distance:
                    best_distance = distance
                    best_idx = idx

            if best_idx >= 0:
                det = detections[best_idx]
                projection = projection_fn(det.centroid)
                track.add_observation(
                    det.centroid,
                    projection,
                    det.bbox,
                    det.timestamp,
                    det.confidence,
                    det.confirmation,
                )
                self._idle_counts[track_id] = 0
                unmatched.remove(best_idx)

        # Create tracks for unmatched
        for idx in unmatched:
            det = detections[idx]
            projection = projection_fn(det.centroid)
            track_id = self._next_id
            self._next_id += 1
            track = VehicleTrack(track_id=track_id, created_at=det.timestamp)
            track.add_observation(
                det.centroid,
                projection,
                det.bbox,
                det.timestamp,
                det.confidence,
                det.confirmation,
            )
            self._tracks[track_id] = track
            self._idle_counts[track_id] = 0

        # Remove stale tracks
        for track_id in list(self._tracks.keys()):
            if self._idle_counts.get(track_id, 0) > self.max_idle_frames:
                del self._tracks[track_id]
                self._idle_counts.pop(track_id, None)

        return self._tracks


class RoadwayModel:
    """
    Maintains a lightweight heat-map of motion to infer roadway extent and orientation.
    """

    def __init__(self, frame_shape: Tuple[int, int, int]) -> None:
        self.height, self.width = frame_shape[:2]
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        self.vehicle_centroids: List[Tuple[float, float]] = []
        self.last_update = time.time()

    def update(self, regions: List[MotionRegion], detections: List[VehicleCandidate]) -> None:
        for region in regions:
            x1 = max(0, int(region.x))
            y1 = max(0, int(region.y))
            x2 = min(self.width, int(region.x + region.w))
            y2 = min(self.height, int(region.y + region.h))
            if x2 > x1 and y2 > y1:
                self.heatmap[y1:y2, x1:x2] += 1.0

        for det in detections:
            self.vehicle_centroids.append(det.centroid)

    def roadway_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        if self.heatmap.max() < 1.0:
            return None

        threshold = max(3.0, self.heatmap.max() * 0.35)
        mask = (self.heatmap >= threshold).astype(np.uint8)
        if mask.sum() < 500:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return int(x), int(y), int(x + w), int(y + h)

    def axis(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.vehicle_centroids) >= 4:
            pts = np.array(self.vehicle_centroids, dtype=np.float32)
            mean = np.mean(pts, axis=0)
            centered = pts - mean
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, np.argmax(eigvals)]
            if principal[0] < 0:
                principal = -principal
            return principal / np.linalg.norm(principal), mean

        bbox = self.roadway_bbox()
        if bbox:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            if width >= height:
                return np.array([1.0, 0.0]), np.array([x1 + width / 2, y1 + height / 2])
            return np.array([0.0, 1.0]), np.array([x1 + width / 2, y1 + height / 2])

        # Default orientation horizontal
        return np.array([1.0, 0.0]), np.array([self.width / 2, self.height / 2])


class VehicleCounter:
    """
    High-level vehicle counting pipeline optimized for roadway scenes.
    """

    VEHICLE_LABELS = {'car', 'truck', 'bus', 'motorbike', 'motorcycle', 'vehicle', 'suv', 'van'}

    def __init__(
        self,
        stream_server: Any,
        camera_id: str,
        duration_seconds: float = 10.0,
        sample_fps: float = 5.0,
        min_confidence: float = 0.4,
        use_high_precision: bool = True,
    ) -> None:
        self.stream_server = stream_server
        self.camera_id = camera_id
        self.duration_seconds = max(2.0, float(duration_seconds))
        self.sample_fps = max(1.0, min(12.0, sample_fps))
        self.min_confidence = max(0.1, min(0.9, min_confidence))
        self.use_high_precision = use_high_precision

        self.motion_detector = SimpleMotionDetector(camera_id=camera_id, enable_learning=False)
        self.fast_detector = ObjectDetector(model_type="mobilenet")
        self.precision_detector: Optional[ObjectDetector] = None

        self.tracker = SimpleAssociationTracker(max_distance=110.0, max_idle_frames=8)
        self.road_model: Optional[RoadwayModel] = None

        self.events: List[Dict[str, Any]] = []
        self.forward_count = 0
        self.backward_count = 0

    def _ensure_precision_detector(self) -> None:
        if not self.use_high_precision:
            return
        if self.precision_detector is None:
            try:
                self.precision_detector = ObjectDetector(model_type="yolo")
            except Exception as exc:
                logger.warning("Unable to load YOLO model for precision confirmation: %s", exc)
                self.use_high_precision = False

    def _decode_frame(self, frame_bytes: Optional[bytes]) -> Optional[np.ndarray]:
        if not frame_bytes:
            return None
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        return frame

    def _select_motion_regions(
        self,
        motion_regions: List[MotionRegion],
        frame_shape: Tuple[int, int, int],
    ) -> List[Tuple[int, int, int, int]]:
        if not motion_regions:
            return []
        h, w = frame_shape[:2]
        frame_area = float(h * w)

        selected: List[Tuple[int, int, int, int]] = []
        for region in motion_regions:
            x, y, width, height = region.x, region.y, region.w, region.h
            if width <= 0 or height <= 0:
                continue
            area = width * height
            area_ratio = area / frame_area
            if area_ratio < 0.0008 or area_ratio > 0.25:
                continue
            aspect = width / max(1.0, float(height))
            if aspect < 0.6 or aspect > 6.5:
                continue
            selected.append((int(x), int(y), int(width), int(height)))

        return selected

    def _extract_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        return frame[y1:y2, x1:x2].copy()

    def _normalize_bbox(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> Tuple[float, float, float, float]:
        h, w = frame_shape[:2]
        x, y, bw, bh = bbox
        return (
            round(x / w, 4),
            round(y / h, 4),
            round(bw / w, 4),
            round(bh / h, 4),
        )

    def _confirm_with_detector(
        self,
        detector: ObjectDetector,
        roi: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[float, Optional[str]]:
        detections = detector.detect(roi, conf_threshold=self.min_confidence)
        best_conf = 0.0
        best_label = None
        for det in detections:
            label = str(det.get("class", "")).lower()
            if label in self.VEHICLE_LABELS:
                conf = float(det.get("confidence", 0.0))
                if conf > best_conf:
                    best_conf = conf
                    best_label = label
        return best_conf, best_label

    def _confirm_vehicle(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[VehicleCandidate]:
        roi = self._extract_roi(frame, bbox)
        if roi.size == 0:
            return None

        fast_conf, fast_label = self._confirm_with_detector(self.fast_detector, roi, bbox)

        best_conf = fast_conf
        confirmation = fast_label if fast_label else "heuristic"

        if fast_conf < self.min_confidence and self.use_high_precision:
            self._ensure_precision_detector()
            if self.precision_detector:
                precise_conf, precise_label = self._confirm_with_detector(self.precision_detector, roi, bbox)
                if precise_conf > best_conf:
                    best_conf = precise_conf
                    confirmation = precise_label or "yolo"

        if best_conf >= self.min_confidence or confirmation == "heuristic":
            x, y, w, h = bbox
            centroid = (x + w / 2.0, y + h / 2.0)
            return VehicleCandidate(
                bbox=bbox,
                centroid=centroid,
                area=w * h,
                timestamp=time.time(),
                confidence=best_conf,
                confirmation=confirmation,
                frame=roi,
                normalized_bbox=self._normalize_bbox(bbox, frame.shape),
            )

        return None

    def _filter_by_roadway(
        self,
        detections: List[VehicleCandidate],
        roadway_bbox: Optional[Tuple[int, int, int, int]],
    ) -> List[VehicleCandidate]:
        if not roadway_bbox:
            return detections
        x1, y1, x2, y2 = roadway_bbox
        filtered = []
        for det in detections:
            cx, cy = det.centroid
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                filtered.append(det)
        return filtered

    def _projection_fn(self, axis: np.ndarray, origin: np.ndarray) -> Any:
        def project(point: Tuple[float, float]) -> float:
            vec = np.array(point) - origin
            return float(np.dot(vec, axis))

        return project

    def _direction_labels(self, axis: np.ndarray) -> Tuple[str, str]:
        angle_deg = math.degrees(math.atan2(axis[1], axis[0]))
        angle_deg = (angle_deg + 360.0) % 360.0
        if 45 <= angle_deg < 135:
            return "down", "up"
        if 135 <= angle_deg < 225:
            return "left", "right"
        if 225 <= angle_deg < 315:
            return "up", "down"
        return "right", "left"

    def _evaluate_tracks(self, axis: np.ndarray, origin: np.ndarray) -> None:
        forward_label, backward_label = self._direction_labels(axis)
        min_displacement = 45.0

        for track in self.tracker._tracks.values():
            if track.counted or len(track.projections) < 4:
                continue
            start_proj = track.projections[0]
            end_proj = track.projections[-1]
            displacement = end_proj - start_proj
            if abs(displacement) < min_displacement:
                continue

            if start_proj < 0 <= end_proj:
                direction = forward_label if displacement > 0 else backward_label
            elif start_proj > 0 >= end_proj:
                direction = backward_label if displacement > 0 else forward_label
            else:
                continue

            track.counted = True
            track.direction = direction

            if direction == forward_label:
                self.forward_count += 1
            else:
                self.backward_count += 1

            duration = track.timestamps[-1] - track.timestamps[0] if track.timestamps else 0.0
            speed = abs(displacement) / duration if duration > 0 else 0.0

            self.events.append(
                {
                    "track_id": track.track_id,
                    "direction": direction,
                    "entered_at": track.timestamps[0] if track.timestamps else None,
                    "exited_at": track.timestamps[-1] if track.timestamps else None,
                    "duration_seconds": round(duration, 3),
                    "normalized_speed": round(speed, 4),
                    "confidence": round(track.best_confidence, 4),
                    "confirmation": track.primary_confirmation,
                    "first_bbox": track.first_bbox,
                    "last_bbox": track.last_bbox,
                }
            )

    def run(self) -> Dict[str, Any]:
        start_time = time.time()
        end_time = start_time + self.duration_seconds

        frame_interval = 1.0 / self.sample_fps
        frames_processed = 0
        accepted_candidates: List[VehicleCandidate] = []

        while time.time() < end_time:
            frame_bytes = self.stream_server.get_frame(self.camera_id)
            frame = self._decode_frame(frame_bytes)
            if frame is None:
                time.sleep(frame_interval)
                continue

            if self.road_model is None:
                self.road_model = RoadwayModel(frame.shape)

            motion_result = self.motion_detector.detect(frame)
            candidate_bboxes = self._select_motion_regions(motion_result.regions, frame.shape)

            detections: List[VehicleCandidate] = []
            for bbox in candidate_bboxes:
                candidate = self._confirm_vehicle(frame, bbox)
                if candidate:
                    detections.append(candidate)

            roadway_bbox = self.road_model.roadway_bbox() if self.road_model else None
            detections = self._filter_by_roadway(detections, roadway_bbox)

            if self.road_model:
                self.road_model.update(motion_result.regions, detections)

            axis, origin = self.road_model.axis() if self.road_model else (np.array([1.0, 0.0]), np.array([frame.shape[1] / 2, frame.shape[0] / 2]))
            projection_fn = self._projection_fn(axis, origin)
            tracks = self.tracker.update(detections, projection_fn)
            self._evaluate_tracks(axis, origin)

            accepted_candidates.extend(detections)
            frames_processed += 1
            time.sleep(frame_interval)

        duration_actual = time.time() - start_time
        axis, origin = self.road_model.axis() if self.road_model else (np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        forward_label, backward_label = self._direction_labels(axis)

        roadway_bbox = self.road_model.roadway_bbox() if self.road_model else None
        road_normalized = None
        if roadway_bbox:
            x1, y1, x2, y2 = roadway_bbox
            road_normalized = {
                "x": round(x1 / self.road_model.width, 4),
                "y": round(y1 / self.road_model.height, 4),
                "width": round((x2 - x1) / self.road_model.width, 4),
                "height": round((y2 - y1) / self.road_model.height, 4),
            }

        summary = {
            "success": True,
            "camera_id": self.camera_id,
            "duration_seconds": round(duration_actual, 3),
            "frames_processed": frames_processed,
            "sampling_rate_fps": round(frames_processed / max(duration_actual, 1e-5), 3),
            "total_vehicles": self.forward_count + self.backward_count,
            "direction_counts": {
                forward_label: self.forward_count,
                backward_label: self.backward_count,
            },
            "road_orientation_degrees": round(math.degrees(math.atan2(axis[1], axis[0])), 2),
            "road_region": road_normalized,
            "events": self.events,
            "detections": [
                {
                    "bbox": det.normalized_bbox,
                    "timestamp": det.timestamp,
                    "confidence": round(det.confidence, 4),
                    "confirmation": det.confirmation,
                }
                for det in accepted_candidates
            ],
        }

        return summary


def encode_event_preview(event: Dict[str, Any], frame: np.ndarray) -> Optional[str]:
    try:
        if frame.size == 0:
            return None
        resized = cv2.resize(frame, (160, 120))
        success, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not success:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        return None

