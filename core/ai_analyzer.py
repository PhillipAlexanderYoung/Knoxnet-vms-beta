import cv2
import numpy as np
import logging
import threading
import queue
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import os

from core.paths import get_models_dir

logger = logging.getLogger(__name__)


class AIAnalyzer:
    def __init__(self):
        self.models = {}
        # Preferred YOLO path (commercial-safe): ONNXRuntime backend (optional, if a BYO ONNX model exists)
        self._yolo_backend = None
        self.analysis_queue = queue.Queue(maxsize=1000)
        self.results_cache = {}
        self.running = True
        # simple cache for ROI verifications: key=(camera_id, x,y,w,h, model_set)
        self._roi_cache: Dict[str, Dict[str, Any]] = {}
        self._detection_config: Dict[str, Any] = {
            'tier2_min_confidence': 0.35,
            'tier2_max_models': 1
        }
        # YOLO runtime parameters (global)
        self._yolo_params: Dict[str, Any] = {
            'conf': 0.25,        # confidence threshold
            'iou': 0.45,         # NMS IoU threshold
            'imgsz': 640,        # inference image size (square)
            'max_det': 100,      # max detections per image
            'agnostic_nms': False,
            'classes': None      # optional filter: list of class ids
        }
        self.load_models()

        # Detection classes for YOLO
        self.coco_classes = [
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

        # Security-relevant classes
        self.security_classes = {
            'person': {'priority': 'high', 'alert_threshold': 0.5},
            'car': {'priority': 'medium', 'alert_threshold': 0.6},
            'truck': {'priority': 'medium', 'alert_threshold': 0.6},
            'motorcycle': {'priority': 'medium', 'alert_threshold': 0.6},
            'bicycle': {'priority': 'low', 'alert_threshold': 0.7},
            'backpack': {'priority': 'medium', 'alert_threshold': 0.7},
            'handbag': {'priority': 'low', 'alert_threshold': 0.8},
            'suitcase': {'priority': 'medium', 'alert_threshold': 0.7}
        }

    def load_models(self):
        """Load AI models for different types of analysis"""
        try:
            logger.info("Loading object detection models from models directory…")

            # Resolve models directory
            try:
                models_dir = str(get_models_dir())
            except Exception:
                models_dir = 'models'

            # 1) Preferred: load a BYO YOLO ONNX model (models/byo/<slug>/model.onnx) if available.
            try:
                from core.model_library.byo_models import list_installed_manifests, read_labels_file
                from core.detection.registry import create_backend, resolve_device_for_backend

                ms = list_installed_manifests()
                if ms:
                    m0 = ms[0]
                    backend = create_backend("onnxruntime")
                    dev = resolve_device_for_backend("onnxruntime", "auto")
                    labels = read_labels_file(m0.labels_abs_path()) if m0.labels_abs_path() else None
                    backend.load(model_path=str(m0.model_abs_path()), labels=labels, device=dev, input_size=int(m0.input_size or 640))
                    self._yolo_backend = backend
                    self.models["yolo"] = backend
                    logger.info(f"Loaded BYO YOLO ONNX model '{m0.slug}' from {m0.model_abs_path()}")
            except Exception as e:
                self._yolo_backend = None
                logger.info(f"BYO ONNX YOLO not available: {e}")

            # 2) Optional plugin: auto-discover and load YOLO .pt models if Ultralytics is installed.
            try:
                try:
                    from ultralytics import YOLO  # type: ignore
                    ultralytics_ok = True
                except Exception:
                    ultralytics_ok = False
                    YOLO = None  # type: ignore

                if ultralytics_ok and os.path.isdir(models_dir):
                    for fname in os.listdir(models_dir):
                        if not fname.lower().endswith(".pt"):
                            continue
                        model_path = os.path.join(models_dir, fname)
                        model_name = os.path.splitext(fname)[0]
                        if model_name in self.models:
                            continue
                        try:
                            self.models[model_name] = YOLO(model_path)  # type: ignore[misc]
                            logger.info(f"Loaded YOLO model '{model_name}' from {model_path}")
                        except Exception as e:
                            logger.warning(f"Failed to load YOLO model '{model_name}' at {model_path}: {e}")
            except Exception as e:
                logger.warning(f"YOLO (.pt) auto-discovery failed: {e}")

            # Choose default alias 'yolo' for plugin (.pt) models if ONNX backend isn't already set.
            if "yolo" not in self.models:
                preferred_order = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
                active_name = None
                for name in preferred_order:
                    if name in self.models:
                        active_name = name
                        break
                if active_name is None:
                    # pick any loaded plugin model
                    candidates = [k for k in self.models.keys() if k not in ("mobilenet",)]
                    active_name = sorted(candidates)[0] if candidates else None
                if active_name is not None:
                    self.models["yolo"] = self.models[active_name]
                    logger.info(f"Default YOLO alias set to '{active_name}'")

            # 2) Load MobileNet SSD (Caffe) for lightweight verification if available
            mobilenet_caffemodel = os.path.join(models_dir, 'mobilenet_iter_73000.caffemodel')
            mobilenet_prototxt = os.path.join(models_dir, 'deploy1.prototxt')
            if not os.path.exists(mobilenet_prototxt):
                mobilenet_prototxt = os.path.join(models_dir, 'MobileNetSSD_deploy.prototxt')

            # Standard MobileNet SSD class labels (VOC set)
            self.mobilenet_classes = [
                "background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]

            if os.path.exists(mobilenet_caffemodel) and os.path.exists(mobilenet_prototxt):
                try:
                    net = cv2.dnn.readNetFromCaffe(mobilenet_prototxt, mobilenet_caffemodel)
                    self.models['mobilenet'] = net
                    logger.info("Loaded MobileNet SSD model successfully")
                except Exception as e:
                    logger.warning(f"Failed to load MobileNet SSD: {e}")
            else:
                logger.info("MobileNet SSD model files not found; skipping mobilenet load")

            logger.info(f"AI models available: {sorted(list(self.models.keys()))}")

        except Exception as e:
            logger.error(f"Error loading AI models: {e}")
            self.models = {}

    def set_detection_config(self, config: Dict[str, Any]) -> None:
        """Update analyzer detection config (optional call from API)."""
        try:
            self._detection_config.update({
                'tier2_min_confidence': float(config.get('tier2_min_confidence', self._detection_config['tier2_min_confidence'])),
                'tier2_max_models': int(config.get('tier2_max_models', self._detection_config['tier2_max_models']))
            })
        except Exception as e:
            logger.warning(f"Failed to update detection config: {e}")

    def queue_frame(self, camera_id: str, frame: np.ndarray):
        """Queue frame for AI analysis"""
        try:
            timestamp = datetime.now()

            # Skip if queue is full (drop oldest frames)
            if self.analysis_queue.full():
                try:
                    self.analysis_queue.get_nowait()
                except queue.Empty:
                    pass

            self.analysis_queue.put((camera_id, frame.copy(), timestamp), block=False)

        except queue.Full:
            logger.warning(f"Analysis queue full, dropping frame from camera {camera_id}")
        except Exception as e:
            logger.error(f"Error queueing frame for analysis: {e}")

    def get_analysis_batch(self, batch_size: int = 5) -> List[Tuple[str, np.ndarray, datetime]]:
        """Get batch of frames for analysis"""
        batch = []

        try:
            for _ in range(batch_size):
                if not self.analysis_queue.empty():
                    item = self.analysis_queue.get_nowait()
                    batch.append(item)
                else:
                    break
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error getting analysis batch: {e}")

        return batch

    def get_queue_size(self) -> int:
        """Get current analysis queue size"""
        return self.analysis_queue.qsize()

    def get_available_models(self) -> List[str]:
        """Get list of available AI models"""
        return list(self.models.keys())

    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """Return lightweight info for loaded models (for status endpoints)."""
        out: List[Dict[str, Any]] = []
        for name, model in self.models.items():
            try:
                if hasattr(model, 'model'):
                    device = next(model.model.parameters()).device.type
                else:
                    device = 'cpu'
            except Exception:
                device = 'unknown'
            out.append({'name': name, 'device': device})
        return out

    def set_active_model(self, model_name: str) -> bool:
        """Set the active model for detection"""
        if model_name in self.models:
            self.models['yolo'] = self.models[model_name]
            logger.info(f"Switched to {model_name} model")
            return True
        else:
            logger.warning(f"Model {model_name} not available")
            return False

    def load_custom_model(self, name: str, model_path: str) -> bool:
        """Load a custom YOLO model from a .pt path and register it under the given name."""
        try:
            if not name:
                name = os.path.splitext(os.path.basename(model_path))[0]
            mdl = YOLO(model_path)
            self.models[name] = mdl
            # Do not switch active automatically; user can choose via per-camera config
            logger.info(f"Loaded custom model '{name}' from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load custom model '{name}' at {model_path}: {e}")
            return False

    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        try:
            mdl = self.models.get(model_name)
            if mdl is None:
                return None
            info = {
                'name': model_name,
                'type': 'YOLO' if hasattr(mdl, 'model') else 'Unknown',
                'device': 'cpu'
            }
            try:
                if hasattr(mdl, 'model'):
                    info['device'] = next(mdl.model.parameters()).device.type
            except Exception:
                pass
            return info
        except Exception:
            return None

    def get_models_health(self) -> Dict[str, Any]:
        """Return a simple health summary for models."""
        models: List[Dict[str, Any]] = []
        for name in self.get_available_models():
            models.append({
                'name': name,
                'status': 'available',
                'last_check': datetime.now().isoformat()
            })
        return {
            'overall_status': 'healthy' if models else 'empty',
            'models': models,
            'last_check': datetime.now().isoformat()
        }

    def get_yolo_params(self) -> Dict[str, Any]:
        """Return current YOLO runtime parameters."""
        return dict(self._yolo_params)

    def set_yolo_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update YOLO runtime parameters (sanitized). Returns current params."""
        try:
            if 'conf' in params:
                self._yolo_params['conf'] = max(0.0, min(1.0, float(params['conf'])))
            if 'iou' in params:
                self._yolo_params['iou'] = max(0.0, min(1.0, float(params['iou'])))
            if 'imgsz' in params:
                v = int(params['imgsz'])
                # clamp to typical YOLO sizes
                v = max(256, min(1536, v))
                # Prefer multiples of 32
                v = int(round(v / 32.0) * 32)
                self._yolo_params['imgsz'] = v
            if 'max_det' in params:
                self._yolo_params['max_det'] = max(1, int(params['max_det']))
            if 'agnostic_nms' in params:
                self._yolo_params['agnostic_nms'] = bool(params['agnostic_nms'])
            if 'classes' in params:
                # Accept list of class names or ids
                classes_param = params['classes']
                resolved = self._resolve_class_filter(classes_param)
                self._yolo_params['classes'] = resolved
        except Exception as e:
            logger.warning(f"Failed to update YOLO params: {e}")
        return dict(self._yolo_params)

    def _resolve_class_filter(self, cls_param: Any) -> Optional[List[int]]:
        """Convert a list of class names or ids into YOLO class id list; return None to disable filter."""
        try:
            if cls_param is None:
                return None
            out: List[int] = []
            if isinstance(cls_param, (list, tuple)):
                for v in cls_param:
                    try:
                        if isinstance(v, (int, float)):
                            out.append(int(v))
                        else:
                            name = str(v).strip().lower()
                            if name in self.coco_classes:
                                out.append(self.coco_classes.index(name))
                    except Exception:
                        continue
            # Ensure unique and sorted
            out = sorted(list({i for i in out if 0 <= i < len(self.coco_classes)}))
            return out if out else None
        except Exception:
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        model_info = {}
        for name, model in self.models.items():
            if hasattr(model, 'model'):
                model_info[name] = {
                    'type': 'YOLO',
                    'parameters': sum(p.numel() for p in model.model.parameters()),
                    'device': next(model.model.parameters()).device.type
                }
            else:
                model_info[name] = {
                    'type': 'Unknown',
                    'parameters': 0,
                    'device': 'unknown'
                }
        return model_info

    def analyze_frame(self, camera_id: str, frame: np.ndarray, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Analyze single frame with AI models"""
        try:
            if not self.models:
                return None

            analysis_result = {
                'camera_id': camera_id,
                'timestamp': timestamp.isoformat(),
                'frame_size': frame.shape,
                'detections': [],
                'analysis_duration': 0,
                'alerts': []
            }

            start_time = time.time()

            # Object detection with YOLO (default tier-2 verifier)
            if 'yolo' in self.models:
                detections = self.detect_objects(frame)
                analysis_result['detections'] = detections

                # Check for security alerts
                alerts = self.check_security_alerts(camera_id, detections, frame)
                analysis_result['alerts'] = alerts

            # Additional analysis can be added here
            # - Face detection and recognition
            # - License plate recognition
            # - Motion detection
            # - Behavior analysis

            analysis_result['analysis_duration'] = time.time() - start_time

            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing frame from camera {camera_id}: {e}")
            return None

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in frame using YOLO"""
        try:
            if 'yolo' not in self.models:
                return []

            detections = []

            # ONNX backend path (preferred)
            if self._yolo_backend is not None and hasattr(self._yolo_backend, "detect"):
                yp = self._yolo_params
                dets = self._yolo_backend.detect(
                    frame,
                    min_confidence=float(yp.get("conf", 0.25)),
                    max_det=int(yp.get("max_det", 100)),
                    allowed_classes=None,
                )
                for d in dets or []:
                    bb = d.get("bbox") if isinstance(d, dict) else None
                    if not isinstance(bb, dict):
                        continue
                    x1 = float(bb.get("x", 0.0) or 0.0)
                    y1 = float(bb.get("y", 0.0) or 0.0)
                    w = float(bb.get("w", 0.0) or 0.0)
                    h = float(bb.get("h", 0.0) or 0.0)
                    x2 = x1 + w
                    y2 = y1 + h
                    detections.append(
                        {
                            "class": str(d.get("class") or "object"),
                            "class_id": int(d.get("class_id", -1) or -1),
                            "confidence": float(d.get("confidence", 0.0) or 0.0),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": w, "height": h},
                            "center": {"x": float((x1 + x2) / 2), "y": float((y1 + y2) / 2)},
                        }
                    )
                return detections

            # Ultralytics plugin path (legacy)
            yp = self._yolo_params
            results = self.models["yolo"](
                frame,
                verbose=False,
                conf=yp.get("conf", 0.25),
                iou=yp.get("iou", 0.45),
                imgsz=yp.get("imgsz", 640),
                max_det=yp.get("max_det", 100),
                agnostic_nms=yp.get("agnostic_nms", False),
                classes=yp.get("classes", None),
            )
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.coco_classes[class_id] if 0 <= class_id < len(self.coco_classes) else "object"
                    detections.append(
                        {
                            "class": class_name,
                            "class_id": class_id,
                            "confidence": float(confidence),
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "width": float(x2 - x1),
                                "height": float(y2 - y1),
                            },
                            "center": {"x": float((x1 + x2) / 2), "y": float((y1 + y2) / 2)},
                        }
                    )

            return detections

        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []

    def _detect_mobilenet_top(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Run MobileNet SSD on an image and return the top class and confidence."""
        try:
            if 'mobilenet' not in self.models or image is None or image.size == 0:
                return None, 0.0

            net = self.models['mobilenet']
            blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            best_conf = 0.0
            best_name: Optional[str] = None
            h, w = image.shape[:2]
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                cid = int(detections[0, 0, i, 1])
                if 0 <= cid < len(self.mobilenet_classes) and conf > best_conf:
                    best_conf = conf
                    best_name = self.mobilenet_classes[cid]
            return best_name, best_conf
        except Exception as e:
            logger.warning(f"MobileNet detection failed: {e}")
            return None, 0.0

    def verify_regions_multi_model(
        self,
        camera_id: str,
        frame: np.ndarray,
        regions: List[Tuple[int,int,int,int]],
        models: Optional[List[str]] = None,
        max_models: int = 1,
        cache_ttl_sec: float = 2.0,
        min_conf: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Verify motion ROIs using one or more models with simple caching.

        Returns list of detections with bbox in pixel coords and per-model confidences.
        """
        out: List[Dict[str, Any]] = []
        try:
            if not regions:
                return out
            h, w = frame.shape[:2]
            use_models: List[str] = []
            if models:
                use_models = [m for m in models if m in self.models]
            if not use_models:
                # Prefer YOLO, then MobileNet if configured for multiple models
                base = ['yolo', 'mobilenet']
                use_models = [m for m in base if m in self.models]
            use_models = use_models[:max(1, int(max_models))]

            now = time.time()
            for (x, y, rw, rh) in regions:
                x0 = max(0, int(x)); y0 = max(0, int(y)); x1 = min(w, int(x+rw)); y1 = min(h, int(y+rh))
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = frame[y0:y1, x0:x1]
                model_conf: Dict[str, float] = {}
                class_name: Optional[str] = None

                # Cache key
                key = f"{camera_id}:{x0}:{y0}:{x1-x0}:{y1-y0}:{','.join(use_models)}"
                hit = False
                if key in self._roi_cache:
                    rec = self._roi_cache[key]
                    if now - rec.get('ts', 0) <= cache_ttl_sec:
                        model_conf = rec.get('model_conf', {})
                        class_name = rec.get('class_name')
                        hit = True
                if not hit:
                    # Run requested models
                    if 'yolo' in use_models and 'yolo' in self.models:
                        best_c = 0.0
                        best_name = None
                        if self._yolo_backend is not None and hasattr(self._yolo_backend, "detect"):
                            yp = self._yolo_params
                            dets = self._yolo_backend.detect(
                                crop,
                                min_confidence=float(yp.get("conf", 0.25)),
                                max_det=int(yp.get("max_det", 100)),
                            )
                            for d in dets or []:
                                conf = float(d.get("confidence", 0.0) or 0.0)
                                name = str(d.get("class") or "object")
                                if conf > best_c:
                                    best_c = conf
                                    best_name = name
                        else:
                            yp = self._yolo_params
                            results = self.models["yolo"](
                                crop,
                                verbose=False,
                                conf=yp.get("conf", 0.25),
                                iou=yp.get("iou", 0.45),
                                imgsz=yp.get("imgsz", 640),
                                max_det=yp.get("max_det", 100),
                                agnostic_nms=yp.get("agnostic_nms", False),
                                classes=yp.get("classes", None),
                            )
                            for res in results:
                                boxes = res.boxes
                                if boxes is None:
                                    continue
                                for box in boxes:
                                    conf = float(box.conf[0].cpu().numpy())
                                    cid = int(box.cls[0].cpu().numpy())
                                    name = self.coco_classes[cid] if 0 <= cid < len(self.coco_classes) else "object"
                                    if conf > best_c:
                                        best_c = conf
                                        best_name = name
                        if best_name is not None:
                            model_conf['yolo'] = best_c
                            class_name = best_name

                    if 'mobilenet' in use_models and 'mobilenet' in self.models:
                        name_m, conf_m = self._detect_mobilenet_top(crop)
                        if name_m is not None:
                            model_conf['mobilenet'] = float(conf_m)
                            # Prefer person/vehicle classes from MobileNet if YOLO is missing or weaker
                            prefer_names = {'person', 'car', 'bus', 'truck', 'motorbike', 'bicycle'}
                            if (class_name is None) or (name_m in prefer_names and float(conf_m) > float(model_conf.get('yolo', 0.0))):
                                class_name = name_m

                    # persist cache
                    self._roi_cache[key] = {
                        'ts': now,
                        'model_conf': model_conf,
                        'class_name': class_name,
                    }

                # keep if any model above threshold
                max_conf = max(model_conf.values()) if model_conf else 0.0
                threshold = float(min_conf if min_conf is not None else self._detection_config['tier2_min_confidence'])
                if max_conf >= threshold:
                    out.append({
                        'class': class_name or 'object',
                        'confidence': max_conf,
                        'model_conf': model_conf,
                        'bbox': {'x': x0, 'y': y0, 'w': (x1-x0), 'h': (y1-y0)},
                    })
        except Exception as e:
            logger.error(f"verify_regions_multi_model error: {e}")
        return out

    def check_security_alerts(self, camera_id: str, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[
        Dict[str, Any]]:
        """Check detections for security alerts"""
        alerts = []

        try:
            for detection in detections:
                class_name = detection['class']
                confidence = detection['confidence']

                # Check if this is a security-relevant detection
                if class_name in self.security_classes:
                    security_config = self.security_classes[class_name]
                    threshold = security_config['alert_threshold']
                    priority = security_config['priority']

                    if confidence >= threshold:
                        alert = {
                            'type': 'object_detection',
                            'camera_id': camera_id,
                            'timestamp': datetime.now().isoformat(),
                            'priority': priority,
                            'object_class': class_name,
                            'confidence': confidence,
                            'location': detection['bbox'],
                            'description': f"{class_name.title()} detected with {confidence:.2%} confidence"
                        }

                        alerts.append(alert)

            # Additional alert logic
            alerts.extend(self.check_intrusion_alerts(camera_id, detections))
            alerts.extend(self.check_loitering_alerts(camera_id, detections))

        except Exception as e:
            logger.error(f"Error checking security alerts: {e}")

        return alerts

    def check_intrusion_alerts(self, camera_id: str, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for intrusion alerts based on detection zones"""
        alerts = []

        try:
            # This would typically load detection zones from camera configuration
            # For now, we'll use a simple example

            person_detections = [d for d in detections if d['class'] == 'person' and d['confidence'] > 0.7]

            if len(person_detections) > 0:
                for detection in person_detections:
                    # Check if person is in restricted area (example logic)
                    center_y = detection['center']['y']

                    # Example: Alert if person detected in upper portion of frame (restricted area)
                    if center_y < 200:  # This would be configurable per camera
                        alert = {
                            'type': 'intrusion',
                            'camera_id': camera_id,
                            'timestamp': datetime.now().isoformat(),
                            'priority': 'high',
                            'description': 'Person detected in restricted area',
                            'location': detection['bbox'],
                            'confidence': detection['confidence']
                        }
                        alerts.append(alert)

        except Exception as e:
            logger.error(f"Error checking intrusion alerts: {e}")

        return alerts

    def check_loitering_alerts(self, camera_id: str, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for loitering based on person tracking"""
        alerts = []

        try:
            # This would require person tracking across frames
            # For now, this is a placeholder for future implementation

            person_count = len([d for d in detections if d['class'] == 'person'])

            if person_count > 3:  # Example threshold
                alert = {
                    'type': 'crowd_detection',
                    'camera_id': camera_id,
                    'timestamp': datetime.now().isoformat(),
                    'priority': 'medium',
                    'description': f'Multiple persons detected ({person_count})',
                    'person_count': person_count
                }
                alerts.append(alert)

        except Exception as e:
            logger.error(f"Error checking loitering alerts: {e}")

        return alerts

    def get_analysis_statistics(self, camera_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get analysis statistics for a camera"""
        try:
            # This would typically query a database
            # For now, return mock statistics

            return {
                'camera_id': camera_id,
                'time_period_hours': hours,
                'total_detections': 150,
                'object_counts': {
                    'person': 45,
                    'car': 32,
                    'truck': 8,
                    'bicycle': 12
                },
                'alert_counts': {
                    'high': 3,
                    'medium': 12,
                    'low': 8
                },
                'average_confidence': 0.78,
                'frames_analyzed': 1024
            }

        except Exception as e:
            logger.error(f"Error getting analysis statistics: {e}")
            return {}