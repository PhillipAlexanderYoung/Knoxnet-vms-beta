"""
Detector Manager with Integrated SORT Tracking
Production-ready solution for object detection and tracking
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .object_detector import ObjectDetector
from .sort_tracker import SortTracker
from .utils.detector_device import DetectorDeviceConfig, resolve_detector_device

logger = logging.getLogger(__name__)


class DetectorManager:
    """
    Manages object detection and tracking for multiple cameras.
    Uses MobileNet SSD as default detector with SORT tracking for persistent object IDs.
    Includes intelligent detection skipping for stable tracked objects to minimize CPU load.
    """
    
    def __init__(self):
        """Initialize the detector manager."""
        self.default_detector = None
        self.custom_detectors: Dict[str, ObjectDetector] = {}
        self.trackers: Dict[str, SortTracker] = {}
        self._global_device_config: DetectorDeviceConfig = resolve_detector_device(None)
        self._global_device_pref: str = self._global_device_config.preference
        self._camera_device_overrides: Dict[str, str] = {}
        
        # Camera-specific configurations
        self.camera_configs: Dict[str, Dict[str, Any]] = {}
        # Detector label spaces
        self._coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        self._voc_classes = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        
        # Detection statistics
        self.stats: Dict[str, Dict[str, Any]] = {}
        
        # Smart detection scheduling - track when we last ran actual detection per camera
        import time
        self._last_full_detection: Dict[str, float] = {}  # camera_id -> timestamp
        self._track_last_positions: Dict[str, Dict[int, Tuple[float, float, float, float]]] = {}  # camera_id -> {track_id -> bbox}
        self._track_stability_scores: Dict[str, Dict[int, int]] = {}  # camera_id -> {track_id -> stable_frames_count}
        
        # Smart detection configuration
        self._smart_detection_config = {
            'stable_threshold': 3,  # Frames needed to consider a track stable
            'redetect_interval': 10.0,  # Seconds between re-detection for verification
            'movement_threshold': 0.05,  # 5% of frame width/height to trigger re-detection
            'max_skip_time': 15.0,  # Maximum seconds to skip detection before forcing re-check
        }
        
        # Initialize default detector (MobileNet SSD default, YOLO optional)
        # Lazy load the detector to speed up startup
        self._default_detector_initialized = False
    
    def _ensure_default_detector(self):
        """Lazy load the default detector if needed."""
        if self.default_detector is not None or self._default_detector_initialized:
            return

        try:
            self.default_detector = ObjectDetector(
                model_type="mobilenet",
                device=self._global_device_pref,
            )
            logger.info("✓ Default detector (MobileNet SSD) initialized")
        except Exception as e:
            logger.error(f"Failed to initialize default MobileNet detector: {e}")
            # Try YOLO as fallback
            try:
                self.default_detector = ObjectDetector(
                    model_type="yolo",
                    device=self._global_device_pref,
                )
                logger.info("✓ Fallback detector (YOLOv8) initialized")
            except Exception as e2:
                logger.error(f"Failed to initialize fallback detector: {e2}")
        
        self._default_detector_initialized = True

    def set_camera_detector(
        self,
        camera_id: str,
        model_type: str = "default",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Set detector for a specific camera.
        
        Args:
            camera_id: Camera identifier
            model_type: Type of model ("default", "mobilenet", "yolo", "custom")
            model_path: Path to custom model file (for custom models)
        """
        try:
            device_pref = device if device is not None else self._camera_device_overrides.get(camera_id)
            changed = False

            if model_type == "default":
                self._ensure_default_detector()
                if device_pref and device_pref != self._global_device_pref:
                    base_type = self.default_detector.model_type if self.default_detector else "mobilenet"
                    detector = ObjectDetector(model_type=base_type, model_path=model_path, device=device_pref)
                    self.custom_detectors[camera_id] = detector
                    self._camera_device_overrides[camera_id] = device_pref
                    logger.info(
                        f"Camera {camera_id} using dedicated {base_type} detector on {device_pref}"
                    )
                    changed = True
                else:
                    if camera_id in self.custom_detectors:
                        del self.custom_detectors[camera_id]
                        changed = True
                    self._camera_device_overrides.pop(camera_id, None)
                    logger.info(f"Camera {camera_id} set to use default detector")
            else:
                detector = ObjectDetector(
                    model_type=model_type,
                    model_path=model_path,
                    device=device_pref or self._global_device_pref,
                )
                self.custom_detectors[camera_id] = detector
                if device_pref:
                    self._camera_device_overrides[camera_id] = device_pref
                else:
                    self._camera_device_overrides.pop(camera_id, None)
                logger.info(f"Camera {camera_id} set to use {model_type} detector")
                changed = True

            if changed:
                self.reset_tracker(camera_id)
                
        except Exception as e:
            logger.error(f"Failed to set detector for camera {camera_id}: {e}")
    
    def get_detector(self, camera_id: str) -> Optional[ObjectDetector]:
        """Get the detector for a camera."""
        if camera_id in self.custom_detectors:
            return self.custom_detectors[camera_id]
        
        self._ensure_default_detector()
        return self.default_detector

    def set_global_device(self, preference: str) -> DetectorDeviceConfig:
        """Set the default device preference for detectors."""

        new_config = resolve_detector_device(preference)
        if new_config == self._global_device_config:
            return new_config

        self._global_device_config = new_config
        self._global_device_pref = new_config.preference
        logger.info(f"Global detector device preference set to {new_config.summary()}")

        if self.default_detector and self.default_detector.set_device(new_config.preference):
            # Reset all trackers because shared detector changed execution device
            for cam_id in list(self.trackers.keys()):
                self.reset_tracker(cam_id)

        for cam_id, detector in list(self.custom_detectors.items()):
            if cam_id in self._camera_device_overrides:
                continue
            if detector.set_device(new_config.preference):
                self.reset_tracker(cam_id)

        return new_config
    
    def get_tracker(self, camera_id: str) -> SortTracker:
        """Get or create tracker for a camera."""
        if camera_id not in self.trackers:
            # Configure tracker for smoother IDs during rapid camera motion.
            self.trackers[camera_id] = SortTracker(
                max_age=8,
                min_hits=2,
                iou_threshold=0.25
            )
        return self.trackers[camera_id]
    
    def _should_skip_detection(self, camera_id: str, current_tracks: List[Dict[str, Any]], 
                                frame_shape: tuple) -> bool:
        """
        Determine if we can skip running detection this frame based on track stability.
        Returns True if all tracks are stable and we haven't exceeded max skip time.
        """
        import time
        
        current_time = time.time()
        last_detection = self._last_full_detection.get(camera_id, 0)
        
        # Force detection if we haven't run it in a while
        if current_time - last_detection > self._smart_detection_config['max_skip_time']:
            return False
        
        # Force detection if we have no tracks (need to detect new objects)
        if not current_tracks:
            return False
        
        # Check if redetection interval has passed (periodic verification)
        if current_time - last_detection > self._smart_detection_config['redetect_interval']:
            return False
        
        # Check track stability - if any track is unstable or moved significantly, run detection
        frame_h, frame_w = frame_shape[:2]
        movement_threshold_px_x = frame_w * self._smart_detection_config['movement_threshold']
        movement_threshold_px_y = frame_h * self._smart_detection_config['movement_threshold']
        
        # Get track states
        last_positions = self._track_last_positions.get(camera_id, {})
        stability_scores = self._track_stability_scores.get(camera_id, {})
        
        for track in current_tracks:
            track_id = track.get('id')
            if track_id is None:
                continue
            
            bbox = track.get('bbox', {})
            current_pos = (bbox.get('x', 0), bbox.get('y', 0), bbox.get('w', 0), bbox.get('h', 0))
            
            # Check if this is a new track or if it moved significantly
            if track_id not in last_positions:
                return False  # New track, need detection
            
            last_pos = last_positions[track_id]
            dx = abs(current_pos[0] - last_pos[0])
            dy = abs(current_pos[1] - last_pos[1])
            
            # If track moved significantly, run detection
            if dx > movement_threshold_px_x or dy > movement_threshold_px_y:
                return False
            
            # Check if track is stable enough
            if stability_scores.get(track_id, 0) < self._smart_detection_config['stable_threshold']:
                return False
        
        # All tracks are stable, we can skip detection
        return True
    
    def _update_track_stability(self, camera_id: str, tracks: List[Dict[str, Any]]):
        """Update stability tracking for current tracks."""
        if camera_id not in self._track_last_positions:
            self._track_last_positions[camera_id] = {}
        if camera_id not in self._track_stability_scores:
            self._track_stability_scores[camera_id] = {}
        
        last_positions = self._track_last_positions[camera_id]
        stability_scores = self._track_stability_scores[camera_id]
        
        # Track IDs present in current frame
        current_track_ids = set()
        
        for track in tracks:
            track_id = track.get('id')
            if track_id is None:
                continue
            
            current_track_ids.add(track_id)
            bbox = track.get('bbox', {})
            current_pos = (bbox.get('x', 0), bbox.get('y', 0), bbox.get('w', 0), bbox.get('h', 0))
            
            # Check if position is similar to last position
            if track_id in last_positions:
                last_pos = last_positions[track_id]
                dx = abs(current_pos[0] - last_pos[0])
                dy = abs(current_pos[1] - last_pos[1])
                
                # If position is stable (minimal movement), increment stability
                if dx < 5 and dy < 5:  # Less than 5 pixels movement
                    stability_scores[track_id] = stability_scores.get(track_id, 0) + 1
                else:
                    stability_scores[track_id] = 0  # Reset stability on movement
            else:
                stability_scores[track_id] = 0  # New track
            
            last_positions[track_id] = current_pos
        
        # Clean up old tracks that are no longer present
        all_track_ids = set(last_positions.keys())
        for old_id in all_track_ids - current_track_ids:
            last_positions.pop(old_id, None)
            stability_scores.pop(old_id, None)
    
    def detect_and_track(self, camera_id: str, frame: np.ndarray, 
                        conf_threshold: Optional[float] = None, 
                        force_detection: bool = False) -> Dict[str, Any]:
        """
        Detect objects and track them with persistent IDs.
        Intelligently skips detection for stable tracked objects to minimize CPU usage.
        
        Args:
            camera_id: Camera identifier
            frame: Input frame (BGR format)
            conf_threshold: Confidence threshold override
            force_detection: If True, always run detection regardless of track stability
            
        Returns:
            Dictionary with detections and tracks:
            {
                'detections': [...],  # Raw detections (empty if skipped)
                'tracks': [...],      # Tracked objects with persistent IDs
                'stats': {...},       # Detection statistics
                'detection_skipped': bool  # True if detection was skipped
            }
        """
        import time
        
        result = {
            'detections': [],
            'tracks': [],
            'stats': {
                'detection_count': 0,
                'track_count': 0,
                'classes': {},
                'detection_skipped': False
            }
        }
        
        try:
            cfg = self.camera_configs.get(camera_id) or {}
            detector = self.get_detector(camera_id)
            if detector is None:
                logger.warning(f"No detector available for camera {camera_id}")
                return result

            device_pref = (
                self._camera_device_overrides.get(camera_id)
                or cfg.get('device')
                or self._global_device_pref
            )
            target_device = self._global_device_pref if detector is self.default_detector else device_pref

            detector_changed = False
            if hasattr(detector, 'set_device') and target_device:
                try:
                    current_pref = getattr(getattr(detector, '_device_config', None), 'preference', None)
                except Exception:
                    current_pref = None
                if current_pref != target_device:
                    try:
                        detector_changed = detector.set_device(target_device)
                    except Exception as exc:
                        logger.debug(f"Failed to apply device preference for {camera_id}: {exc}")
                        detector_changed = False

            if detector_changed:
                self.reset_tracker(camera_id)

            tracker = self.get_tracker(camera_id)

            # Get existing tracks from tracker (without running detection yet)
            current_tracks = list(tracker._tracks.values()) if hasattr(tracker, '_tracks') else []

            # Convert internal track objects to dicts for stability checking
            track_dicts = []
            for track in current_tracks:
                if hasattr(track, 'bbox') and hasattr(track, 'id'):
                    x, y, w, h = track.bbox
                    track_dicts.append({
                        'id': track.id,
                        'bbox': {'x': float(x), 'y': float(y), 'w': float(w), 'h': float(h)},
                        'class': getattr(track, 'stable_label', getattr(track, 'class_name', 'object')),
                        'confidence': float(getattr(track, 'label_confidence', getattr(track, 'confidence', 0.0))),
                        'age': getattr(track, 'age', 0),
                        'hits': getattr(track, 'hits', 0)
                    })

            # Decide if we can skip detection
            should_skip = (
                not detector_changed
                and not force_detection
                and self._should_skip_detection(camera_id, track_dicts, frame.shape)
            )
            
            if should_skip and track_dicts:
                # Skip expensive detection, but still age/prune tracks to avoid lingering boxes
                aged_tracks = tracker.update([])  # Advances time_since_update and prunes stale
                logger.debug(f"⚡ Skipping detection for {camera_id} - returning {len(aged_tracks)} aged tracks")
                result['tracks'] = aged_tracks
                result['stats']['track_count'] = len(aged_tracks)
                result['stats']['detection_skipped'] = True
                result['detection_skipped'] = True

                # Update track stability with the aged (filtered) set
                self._update_track_stability(camera_id, aged_tracks)

                # Count classes
                for track in aged_tracks:
                    class_name = track.get('class', 'unknown')
                    result['stats']['classes'][class_name] = result['stats']['classes'].get(class_name, 0) + 1

                return result
            
            # Apply any per-camera overrides for MobileNet parameters
            if hasattr(detector, 'set_nms_threshold') and 'nms' in cfg:
                try:
                    detector.set_nms_threshold(float(cfg.get('nms')))
                except Exception:
                    pass
            if hasattr(detector, 'set_max_detections') and 'max_det' in cfg:
                try:
                    detector.set_max_detections(int(cfg.get('max_det')))
                except Exception:
                    pass

            # Run detection
            logger.debug(f"🔍 Running full detection for {camera_id}")
            detection_conf = conf_threshold
            if detection_conf is None and 'confidence' in cfg:
                try:
                    detection_conf = float(cfg.get('confidence'))
                except Exception:
                    detection_conf = None
            detections = detector.detect(frame, detection_conf)
            
            # Apply class filter pre-tracking, mapped to the model label space
            detections = self._filter_by_classes(camera_id, detections, detector)
            result['detections'] = detections
            result['stats']['detection_count'] = len(detections)
            
            # Update tracker with new detections
            tracks = tracker.update(detections)
            result['tracks'] = tracks
            result['stats']['track_count'] = len(tracks)
            result['detection_skipped'] = False
            
            # Mark this as a full detection time
            self._last_full_detection[camera_id] = time.time()
            
            # Update track stability
            self._update_track_stability(camera_id, tracks)
            
            # Count classes
            for track in tracks:
                class_name = track.get('class', 'unknown')
                result['stats']['classes'][class_name] = result['stats']['classes'].get(class_name, 0) + 1
            
            # Update statistics
            self._update_stats(camera_id, result)
            
        except Exception as e:
            logger.error(f"Detection and tracking failed for camera {camera_id}: {e}")
        
        return result
    
    def detect_in_region(self, camera_id: str, frame: np.ndarray, 
                        region: Dict[str, int], conf_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Detect objects within a specific region (ROI).
        
        Args:
            camera_id: Camera identifier
            frame: Input frame
            region: Region dict with keys: x, y, w, h
            conf_threshold: Confidence threshold
            
        Returns:
            List of detections within the region
        """
        try:
            # Extract region
            x = max(0, region.get('x', 0))
            y = max(0, region.get('y', 0))
            w = max(1, region.get('w', 1))
            h = max(1, region.get('h', 1))
            
            # Ensure region is within frame bounds
            h_frame, w_frame = frame.shape[:2]
            x = min(x, w_frame - 1)
            y = min(y, h_frame - 1)
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)
            
            # Crop region
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0:
                return []
            
            # Run detection on ROI
            detector = self.get_detector(camera_id)
            if detector is None:
                return []
            
            detections = detector.detect(roi, conf_threshold)
            detections = self._filter_by_classes(camera_id, detections, detector)
            
            # Adjust bbox coordinates to full frame
            for det in detections:
                det['bbox']['x'] += x
                det['bbox']['y'] += y
            
            return detections
            
        except Exception as e:
            logger.error(f"Region detection failed: {e}")
            return []

    def _resolve_model_space(self, detector: Optional[ObjectDetector]) -> str:
        try:
            if detector is None:
                return 'voc'
            model_type = getattr(detector, 'model_type', 'mobilenet')
            return 'coco' if str(model_type).lower() in ('yolo', 'custom') else 'voc'
        except Exception:
            return 'voc'

    def _normalize_label(self, label: str) -> str:
        # Minor normalization for semantic equivalents
        name = (label or 'object').strip().lower()
        if name in ("car", "truck", "bus", "van", "automobile", "pickup"):
            return "car"
        if name in ("motorbike", "motorcycle"):
            return "motorcycle"
        if name == "aeroplane":
            return "airplane"
        return name

    def _map_requested_classes(self, requested: List[str], model_space: str) -> List[str]:
        # Convert requested classes into the detector's label space
        out: List[str] = []
        for raw in requested or []:
            name = self._normalize_label(str(raw))
            if model_space == 'voc':
                # Map to VOC names
                mapping = {
                    'airplane': 'aeroplane',
                    'motorcycle': 'motorbike',
                    'dining table': 'diningtable',
                    'potted plant': 'pottedplant',
                    'tv': 'tvmonitor'
                }
                name_voc = mapping.get(name, name)
                if name_voc in self._voc_classes:
                    out.append(name_voc)
            else:
                # COCO space
                mapping = {
                    'aeroplane': 'airplane',
                    'motorbike': 'motorcycle',
                    'diningtable': 'dining table',
                    'pottedplant': 'potted plant',
                    'tvmonitor': 'tv'
                }
                name_coco = mapping.get(name, name)
                if name_coco in self._coco_classes:
                    out.append(name_coco)
        # De-duplicate while preserving order
        seen = set()
        deduped: List[str] = []
        for n in out:
            if n not in seen:
                seen.add(n)
                deduped.append(n)
        return deduped

    def _filter_by_classes(self, camera_id: str, detections: List[Dict[str, Any]], detector: Optional[ObjectDetector]) -> List[Dict[str, Any]]:
        try:
            cfg = self.camera_configs.get(camera_id) or {}
            requested: List[str] = list(cfg.get('classes') or cfg.get('target_classes') or [])
            if not requested:
                return detections or []
            model_space = self._resolve_model_space(detector)
            allowed = set(self._map_requested_classes(requested, model_space))
            if not allowed:
                return []
            filtered: List[Dict[str, Any]] = []
            for d in detections or []:
                cls_name = self._normalize_label(str(d.get('class', 'object')))
                # Map the detection class into the model's canonical space for comparison
                if model_space == 'voc':
                    # detection label already in VOC; normalize synonyms to VOC
                    back_map = {
                        'airplane': 'aeroplane',
                        'motorcycle': 'motorbike',
                        'dining table': 'diningtable',
                        'potted plant': 'pottedplant',
                        'tv': 'tvmonitor'
                    }
                    cls_cmp = back_map.get(cls_name, cls_name)
                else:
                    # COCO space
                    back_map = {
                        'aeroplane': 'airplane',
                        'motorbike': 'motorcycle',
                        'diningtable': 'dining table',
                        'pottedplant': 'potted plant',
                        'tvmonitor': 'tv'
                    }
                    cls_cmp = back_map.get(cls_name, cls_name)
                if cls_cmp in allowed:
                    filtered.append(d)
            return filtered
        except Exception as e:
            logger.debug(f"Class filtering error for {camera_id}: {e}")
            return detections or []
    
    def load_custom_model(self, model_name: str, model_path: str, 
                         model_type: str = "yolo") -> bool:
        """
        Load a custom model that can be assigned to cameras.
        
        Args:
            model_name: Name identifier for the model
            model_path: Path to model file
            model_type: Type of model ("yolo", "custom")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            detector = ObjectDetector(model_type=model_type, model_path=model_path)
            self.custom_detectors[f"model_{model_name}"] = detector
            logger.info(f"✓ Loaded custom model '{model_name}' from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load custom model '{model_name}': {e}")
            return False
    
    def set_detection_config(self, camera_id: str, config: Dict[str, Any]):
        """
        Set detection configuration for a camera.
        
        Args:
            camera_id: Camera identifier
            config: Configuration dictionary with keys like:
                - enabled: bool
                - confidence: float
                - model: str
                - classes: List[str] (filter specific classes)
        """
        self.camera_configs[camera_id] = config
        logger.info(f"Updated detection config for camera {camera_id}")

        model = config.get('model') or 'default'
        device_pref = config.get('device')
        model_path = config.get('model_path') or config.get('path')

        if model or device_pref:
            self.set_camera_detector(camera_id, model_type=model, model_path=model_path, device=device_pref)
    
    def get_detection_config(self, camera_id: str) -> Dict[str, Any]:
        """Get detection configuration for a camera."""
        return self.camera_configs.get(camera_id, {
            'enabled': True,
            'confidence': 0.25,
            'model': 'default',
            'classes': []
        })
    
    def _update_stats(self, camera_id: str, result: Dict[str, Any]):
        """Update detection statistics."""
        if camera_id not in self.stats:
            self.stats[camera_id] = {
                'total_detections': 0,
                'total_tracks': 0,
                'class_counts': {}
            }
        
        stats = self.stats[camera_id]
        stats['total_detections'] += result['stats']['detection_count']
        stats['total_tracks'] += result['stats']['track_count']
        
        for class_name, count in result['stats']['classes'].items():
            stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + count
    
    def get_stats(self, camera_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detection statistics.
        
        Args:
            camera_id: Specific camera ID, or None for all cameras
            
        Returns:
            Statistics dictionary
        """
        if camera_id:
            return self.stats.get(camera_id, {})
        return self.stats
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        self._ensure_default_detector()
        models = ['default (MobileNet SSD)']
        
        if self.default_detector:
            models.append(f"default ({self.default_detector.model_type})")
        
        # Add custom models
        for name in self.custom_detectors.keys():
            if name.startswith('model_'):
                models.append(name.replace('model_', ''))
        
        return models
    
    def get_model_info(self, camera_id: Optional[str] = None) -> Dict[str, Any]:
        """Get information about the detector model."""
        detector = self.get_detector(camera_id) if camera_id else self.default_detector
        
        if detector:
            return detector.get_model_info()
        
        return {'error': 'No detector available'}
    
    def reset_tracker(self, camera_id: str):
        """Reset tracker for a camera."""
        if camera_id in self.trackers:
            del self.trackers[camera_id]
            logger.info(f"Reset tracker for camera {camera_id}")
        
        # Also reset smart detection state
        self._last_full_detection.pop(camera_id, None)
        self._track_last_positions.pop(camera_id, None)
        self._track_stability_scores.pop(camera_id, None)
    
    def set_smart_detection_config(self, config: Dict[str, Any]):
        """
        Configure smart detection parameters.
        
        Args:
            config: Dictionary with keys:
                - stable_threshold: Frames needed to consider a track stable (default: 3)
                - redetect_interval: Seconds between re-detection for verification (default: 10.0)
                - movement_threshold: Fraction of frame size to trigger re-detection (default: 0.05)
                - max_skip_time: Maximum seconds to skip detection (default: 15.0)
        """
        self._smart_detection_config.update(config)
        logger.info(f"Smart detection config updated: {self._smart_detection_config}")
    
    def get_smart_detection_config(self) -> Dict[str, Any]:
        """Get current smart detection configuration."""
        return self._smart_detection_config.copy()
    
    def get_detection_skip_stats(self, camera_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about detection skipping.
        
        Args:
            camera_id: Specific camera or None for all cameras
            
        Returns:
            Dictionary with skip statistics
        """
        import time
        
        if camera_id:
            last_detection = self._last_full_detection.get(camera_id, 0)
            time_since_detection = time.time() - last_detection if last_detection > 0 else 0
            
            return {
                'camera_id': camera_id,
                'last_full_detection': last_detection,
                'time_since_detection': time_since_detection,
                'tracked_objects': len(self._track_last_positions.get(camera_id, {})),
                'stable_tracks': sum(
                    1 for score in self._track_stability_scores.get(camera_id, {}).values()
                    if score >= self._smart_detection_config['stable_threshold']
                )
            }
        else:
            # Return stats for all cameras
            stats = {}
            for cam_id in set(list(self._last_full_detection.keys()) + 
                             list(self._track_last_positions.keys())):
                stats[cam_id] = self.get_detection_skip_stats(cam_id)
            return stats
    
    def cleanup(self):
        """Cleanup resources."""
        self.custom_detectors.clear()
        self.trackers.clear()
        self.camera_configs.clear()
        self.stats.clear()
        self._last_full_detection.clear()
        self._track_last_positions.clear()
        self._track_stability_scores.clear()
        logger.info("Detector manager cleaned up")


# Global singleton instance
_detector_manager_instance: Optional[DetectorManager] = None


def get_detector_manager() -> DetectorManager:
    """Get or create the global detector manager instance."""
    global _detector_manager_instance
    if _detector_manager_instance is None:
        _detector_manager_instance = DetectorManager()
    return _detector_manager_instance
