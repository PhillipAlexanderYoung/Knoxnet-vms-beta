"""
Production-Ready Object Detector with SORT Tracking
Supports MobileNet SSD (default), YOLO, and custom models
"""

import sys
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os

from .paths import get_models_dir
from .utils.detector_device import (
    DetectorDeviceConfig,
    resolve_detector_device,
)

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Unified object detector supporting multiple models with SORT tracking integration.
    MobileNet SSD is used as the fast, lightweight default detector.
    """
    
    def __init__(
        self,
        model_type: str = "mobilenet",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the object detector.
        
        Args:
            model_type: Type of model to use ("mobilenet", "yolo", or "custom")
            model_path: Path to custom model file (optional)
            device: Optional device preference ("auto", "cpu", "gpu")
        """
        self.model_type = model_type.lower()
        self.model = None
        self.model_path = model_path
        self._device_config: DetectorDeviceConfig = resolve_detector_device(device)
        
        # COCO class names for YOLO models
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
        
        # MobileNet SSD class names (VOC dataset)
        self.mobilenet_classes = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        
        # Detection parameters
        self.conf_threshold = 0.25  # Confidence threshold (can be overridden per call)
        self.nms_threshold = 0.45   # NMS IoU threshold
        self.input_size = 640       # Input size for YOLO
        # Allow a healthy number of detections to avoid dropping vehicles in wide scenes
        self.max_detections = 50
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the detection model based on type."""
        try:
            if self.model_type == "mobilenet":
                self._load_mobilenet()
            elif self.model_type == "yolo":
                self._load_yolo()
            elif self.model_type == "custom":
                self._load_custom()
            else:
                logger.warning(f"Unknown model type: {self.model_type}, falling back to MobileNet")
                self.model_type = "mobilenet"
                self._load_mobilenet()
        except Exception as e:
            logger.error(f"Failed to load {self.model_type} model: {e}")
            # Fallback to MobileNet
            if self.model_type != "mobilenet":
                logger.info("Falling back to MobileNet SSD")
                self.model_type = "mobilenet"
                self._load_mobilenet()

        self._apply_device_to_model()

    def _apply_device_to_model(self):
        """Configure the underlying model with the resolved device settings."""

        if self.model is None:
            return

        # OpenCV DNN based models (MobileNet / Caffe custom)
        if hasattr(self.model, "setPreferableBackend"):
            self._configure_opencv_backend()

        # PyTorch-based models (Ultralytics YOLO / custom PT)
        if hasattr(self.model, "to"):
            self._configure_yolo_device()

    def _configure_opencv_backend(self):
        try:
            backend = int(self._device_config.opencv_backend)
            target = int(self._device_config.opencv_target)
            self.model.setPreferableBackend(backend)
            self.model.setPreferableTarget(target)
        except Exception as exc:
            logger.debug(f"Unable to set OpenCV backend/target: {exc}")

    def _configure_yolo_device(self):
        try:
            device = self._device_config.torch_device
            if device == "cuda" and not self._device_config.capabilities.torch_cuda:
                device = "cpu"
            # Some Ultralytics models expose model.model for the underlying torch nn.Module
            target_module = getattr(self.model, "model", self.model)
            if hasattr(target_module, "to"):
                target_module.to(device)
        except Exception as exc:
            logger.debug(f"Unable to move YOLO model to device {self._device_config.torch_device}: {exc}")

    def set_device(self, preference: Optional[str] = None) -> bool:
        """Resolve and apply a new device preference.

        Returns True when the underlying configuration changed in a way that
        warrants downstream components (e.g., trackers) to refresh.
        """

        new_config = resolve_detector_device(preference or self._device_config.preference)
        changed = new_config != self._device_config
        self._device_config = new_config
        self._apply_device_to_model()
        if changed:
            logger.info("Detector moved to device: %s", self._device_config.summary())
        return changed

    @staticmethod
    def _build_detection(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        frame_w: int,
        frame_h: int,
        class_name: str,
        confidence: float,
        class_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a sanitized detection dictionary or None if invalid."""

        if not np.isfinite([x1, y1, x2, y2]).all():
            return None

        frame_w = max(1.0, float(frame_w))
        frame_h = max(1.0, float(frame_h))

        x1 = float(max(0.0, min(x1, frame_w - 1.0)))
        y1 = float(max(0.0, min(y1, frame_h - 1.0)))
        x2 = float(max(x1 + 1e-2, min(x2, frame_w)))
        y2 = float(max(y1 + 1e-2, min(y2, frame_h)))

        width = float(x2 - x1)
        height = float(y2 - y1)

        if width < 1.0 or height < 1.0:
            return None

        confidence = float(max(0.0, min(confidence, 1.0)))

        return {
            'bbox': {
                'x': round(x1, 2),
                'y': round(y1, 2),
                'w': round(width, 2),
                'h': round(height, 2),
            },
            'class': class_name,
            'confidence': confidence,
            'class_id': class_id if class_id is not None else -1,
        }
    
    def _load_mobilenet(self):
        """Load MobileNet SSD model - lightweight and fast default."""
        try:
            models_dir = get_models_dir()
            
            caffemodel_path = models_dir / "mobilenet_iter_73000.caffemodel"
            
            # Use deploy1.prototxt for 300x300 input (correct one for this model)
            prototxt_path = models_dir / "deploy1.prototxt"
            
            # Fallback to MobileNetSSD_deploy.prototxt if deploy1.prototxt doesn't exist
            if not prototxt_path.exists():
                prototxt_path = models_dir / "MobileNetSSD_deploy.prototxt"
                logger.warning("Using MobileNetSSD_deploy.prototxt (224x224) - may have issues")
            
            if not caffemodel_path.exists() or not prototxt_path.exists():
                logger.error(f"MobileNet model files not found in {models_dir}")
                raise FileNotFoundError("MobileNet model files not found")
            
            self.model = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
            
            # Set backend and target for optimization
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            logger.info(f"✓ Loaded MobileNet SSD model from {caffemodel_path} with {prototxt_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to load MobileNet SSD: {e}")
            raise
    
    def _load_yolo(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            
            if self.model_path and Path(self.model_path).exists():
                model_file = Path(self.model_path)
            else:
                models_dir = get_models_dir()
                
                # Try yolov8n.pt first (smallest/fastest)
                for model_name in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
                    model_file = models_dir / model_name
                    if model_file.exists():
                        break
                else:
                    raise FileNotFoundError("No YOLO model files found")
            
            self.model = YOLO(str(model_file))
            logger.info(f"✓ Loaded YOLO model from {model_file}")
            
        except ImportError:
            logger.error("Ultralytics package not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def _load_custom(self):
        """Load a custom model."""
        if not self.model_path:
            raise ValueError("Custom model path not provided")
        
        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Custom model not found: {self.model_path}")
        
        # Determine model type by extension
        if model_file.suffix == '.pt':
            # Assume YOLO format
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(model_file))
                logger.info(f"✓ Loaded custom YOLO model from {model_file}")
            except ImportError:
                logger.error("Ultralytics package required for .pt models")
                raise
        elif model_file.suffix == '.caffemodel':
            # Assume Caffe format
            prototxt = model_file.with_suffix('.prototxt')
            if not prototxt.exists():
                # Try MobileNetSSD_deploy.prototxt as fallback
                prototxt = model_file.parent / "MobileNetSSD_deploy.prototxt"
            
            if not prototxt.exists():
                raise FileNotFoundError(f"Prototxt file not found for {model_file}")
            
            self.model = cv2.dnn.readNetFromCaffe(str(prototxt), str(model_file))
            logger.info(f"✓ Loaded custom Caffe model from {model_file}")
        else:
            raise ValueError(f"Unsupported model format: {model_file.suffix}")
    
    def detect(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame (BGR format)
            conf_threshold: Confidence threshold override
            
        Returns:
            List of detections with format:
            [{
                'bbox': {'x': int, 'y': int, 'w': int, 'h': int},
                'class': str,
                'confidence': float,
                'class_id': int
            }, ...]
        """
        if self.model is None:
            logger.error("No model loaded")
            return []
        
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        try:
            if self.model_type == "mobilenet":
                return self._detect_mobilenet(frame, threshold)
            elif self.model_type in ["yolo", "custom"]:
                return self._detect_yolo(frame, threshold)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return []
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _detect_mobilenet(self, frame: np.ndarray, conf_threshold: float) -> List[Dict[str, Any]]:
        """Detect objects using MobileNet SSD - matches reference implementation exactly."""
        detections: List[Dict[str, Any]] = []
        h, w = frame.shape[:2]

        # Create blob from frame (exact reference settings)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.model.setInput(blob)
        outputs = self.model.forward()

        # Collect raw boxes for NMS
        boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []
        classes: List[int] = []

        # Process detections exactly like reference implementation
        for i in range(outputs.shape[2]):
            conf = float(outputs[0, 0, i, 2])
            # Guard against extreme/garbage confidences sometimes seen in Caffe models
            if not (0.01 < conf < 0.999):
                continue
            if conf < float(conf_threshold):
                continue

            class_id = int(outputs[0, 0, i, 1])
            if not (0 <= class_id < len(self.mobilenet_classes)):
                continue

            box = outputs[0, 0, i, 3:7] * np.array([w, h, w, h])
            if not np.all(np.isfinite(box)):
                continue
            x1, y1, x2, y2 = box.astype(float)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append((x1, y1, x2, y2))
            scores.append(conf)
            classes.append(class_id)

        # Apply simple NMS per class to reduce duplicates and keep strongest boxes
        keep_indices: List[int] = []
        if boxes:
            boxes_np = np.array(boxes, dtype=np.float32)
            scores_np = np.array(scores, dtype=np.float32)
            classes_np = np.array(classes, dtype=np.int32)

            for cls in np.unique(classes_np):
                idxs = np.where(classes_np == cls)[0]
                if idxs.size == 0:
                    continue
                picked = self._nms(boxes_np[idxs], scores_np[idxs], self.nms_threshold)
                keep_indices.extend(list(idxs[picked]))

        # Build final detections list
        for i in keep_indices:
            x1, y1, x2, y2 = boxes[i]
            class_id = int(classes[i])
            class_name = self.mobilenet_classes[class_id]
            det = self._build_detection(x1, y1, x2, y2, w, h, class_name, float(scores[i]), class_id)
            if det:
                detections.append(det)

        detections.sort(key=lambda d: float(d.get('confidence', 0.0)), reverse=True)
        return detections[: self.max_detections]

    def _nms(self, boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        """A simple class-agnostic NMS for xyxy boxes.
        Returns indices of boxes to keep (relative to the input arrays).
        """
        if boxes_xyxy.size == 0:
            return []
        x1 = boxes_xyxy[:, 0]
        y1 = boxes_xyxy[:, 1]
        x2 = boxes_xyxy[:, 2]
        y2 = boxes_xyxy[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = np.where(union > 0, inter / union, 0.0)
            inds = np.where(iou <= float(iou_thresh))[0]
            order = order[inds + 1]
        return keep
    
    def _detect_yolo(self, frame: np.ndarray, conf_threshold: float) -> List[Dict[str, Any]]:
        """Detect objects using YOLO."""
        detections = []
        frame_h, frame_w = frame.shape[:2]

        # Run inference
        results = self.model(
            frame,
            conf=conf_threshold,
            iou=self.nms_threshold,
            imgsz=self.input_size,
            max_det=int(self.max_detections or 100),
            verbose=False
        )
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id < len(self.coco_classes):
                        class_name = self.coco_classes[class_id]
                        det = self._build_detection(x1, y1, x2, y2, frame_w, frame_h, class_name, confidence, class_id)
                        if det:
                            detections.append(det)
        
        detections.sort(key=lambda d: float(d.get('confidence', 0.0)), reverse=True)
        return detections[: self.max_detections]
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold. Only log when the value actually changes."""
        new_val = max(0.0, min(1.0, threshold))
        if float(new_val) != float(self.conf_threshold):
            self.conf_threshold = new_val
            logger.info(f"Confidence threshold set to {self.conf_threshold}")
        else:
            self.conf_threshold = new_val
    
    def set_nms_threshold(self, threshold: float):
        """Set NMS threshold. Only log when the value actually changes."""
        new_val = max(0.0, min(1.0, threshold))
        if float(new_val) != float(self.nms_threshold):
            self.nms_threshold = new_val
            logger.info(f"NMS threshold set to {self.nms_threshold}")
        else:
            self.nms_threshold = new_val
    
    def set_max_detections(self, value: int):
        """Set maximum number of detections kept after NMS. Only log changes."""
        try:
            new_val = max(1, int(value))
        except Exception:
            new_val = 50
        if int(new_val) != int(self.max_detections):
            self.max_detections = new_val
            logger.info(f"Max detections set to {self.max_detections}")
        else:
            self.max_detections = new_val
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'type': self.model_type,
            'path': str(self.model_path) if self.model_path else None,
            'conf_threshold': self.conf_threshold,
            'nms_threshold': self.nms_threshold,
            'loaded': self.model is not None,
            'device': self._device_config.kind,
            'torch_device': self._device_config.torch_device,
            'device_preference': self._device_config.preference,
        }
