"""
Production vision service using lightweight Hugging Face models.
Models download automatically on first run (~500MB total).
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("vision_local")

def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _resolve_cache_dir() -> Path:
    """
    Resolve a writable cache directory.

    - Source/dev: keep current behavior (repo-relative models/vision/cache).
    - Frozen (PyInstaller/AppImage): use per-user XDG cache, never _internal.
    - Override: KNOXNET_VISION_CACHE_DIR.
    """
    env = (os.environ.get("KNOXNET_VISION_CACHE_DIR") or "").strip()
    if env:
        p = Path(env).expanduser()
        return p.resolve()

    if _is_frozen():
        xdg_cache = (os.environ.get("XDG_CACHE_HOME") or "").strip()
        base = Path(xdg_cache).expanduser() if xdg_cache else (Path.home() / ".cache")
        return (base / "KnoxnetVMS" / "models" / "vision" / "cache").resolve()

    # services/vision_local/app_production.py -> repo root is parents[2]
    return (Path(__file__).resolve().parents[2] / "models" / "vision" / "cache").resolve()


# Configuration (cache must be set before HF/Transformers do any filesystem work)
CACHE_DIR = _resolve_cache_dir()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In frozen builds, force HF/Transformers/Torch caches to land under the writable cache dir.
# (Only set defaults; allow user overrides.)
if _is_frozen():
    hf_home = (CACHE_DIR / "huggingface").resolve()
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str((hf_home / "hub").resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((hf_home / "hub").resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str((hf_home / "transformers").resolve()))
    os.environ.setdefault("TORCH_HOME", str((CACHE_DIR / "torch").resolve()))


import torch
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
    pipeline,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOGGER.info(f"🖥️  Using device: {DEVICE}")
LOGGER.info(f"📁 Cache directory: {CACHE_DIR}")

app = FastAPI(title="Vision Local Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
_blip_model = None
_blip_processor = None
_git_model = None
_git_processor = None
_object_detector = None


class ObjectTag(BaseModel):
    label: str
    confidence: float
    source: str = "model"


class DescribeResult(BaseModel):
    caption: str
    objects: List[ObjectTag]
    model: str


class DescribeResponse(BaseModel):
    results: List[DescribeResult]
    cached: bool = False


class MatchCriteriaResponse(BaseModel):
    verdict: str
    confidence: float
    reason: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]


def load_blip():
    """Load BLIP model (Salesforce/blip-image-captioning-base, ~990MB)"""
    global _blip_model, _blip_processor
    if _blip_model is not None:
        return _blip_model, _blip_processor
    
    LOGGER.info("📥 Loading BLIP model (first run will download ~990MB)...")
    _blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        cache_dir=str(CACHE_DIR)
    )
    # Prefer safetensors weights to avoid torch.load restrictions present in some environments.
    torch_dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
    _blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        cache_dir=str(CACHE_DIR),
        torch_dtype=torch_dtype,
        use_safetensors=True,
    ).to(DEVICE)
    _blip_model.eval()
    LOGGER.info("✅ BLIP model loaded successfully")
    return _blip_model, _blip_processor


def load_git():
    """Load GIT model (microsoft/git-base, smaller alternative ~500MB)"""
    global _git_model, _git_processor
    if _git_model is not None:
        return _git_model, _git_processor
    
    LOGGER.info("📥 Loading GIT model (first run will download ~500MB)...")
    _git_processor = AutoProcessor.from_pretrained(
        "microsoft/git-base",
        cache_dir=str(CACHE_DIR)
    )
    torch_dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
    _git_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/git-base",
        cache_dir=str(CACHE_DIR),
        torch_dtype=torch_dtype,
        use_safetensors=True,
    ).to(DEVICE)
    _git_model.eval()
    LOGGER.info("✅ GIT model loaded successfully")
    return _git_model, _git_processor


def decode_image(image_b64: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    if image_b64.startswith("data:image/"):
        image_b64 = image_b64.split(",", 1)[1]
    
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def generate_caption_blip(image: Image.Image) -> str:
    """Generate caption using BLIP model."""
    model, processor = load_blip()
    
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def generate_caption_git(image: Image.Image) -> str:
    """Generate caption using GIT model."""
    model, processor = load_git()
    
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(pixel_values=inputs.pixel_values, max_new_tokens=50)
    
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    return caption


def load_object_detector():
    """Load YOLO object detector for detailed object extraction."""
    global _object_detector
    if _object_detector is not None:
        return _object_detector
    
    try:
        from ultralytics import YOLO
        yolo_path = Path(__file__).parent.parent.parent / "yolov8n.pt"
        if yolo_path.exists():
            LOGGER.info(f"📥 Loading YOLO detector from {yolo_path}...")
            _object_detector = YOLO(str(yolo_path))
            LOGGER.info("✅ YOLO detector loaded")
            return _object_detector
    except Exception as e:
        LOGGER.warning(f"YOLO detector unavailable: {e}")
    return None


def detect_objects_yolo(image: Image.Image) -> List[ObjectTag]:
    """Detect objects using YOLO model."""
    detector = load_object_detector()
    if detector is None:
        return []
    
    try:
        results = detector(image, conf=0.25, verbose=False)
        objects = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = result.names[class_id]
                objects.append(ObjectTag(
                    label=label,
                    confidence=confidence,
                    source="yolo"
                ))
        return objects
    except Exception as e:
        LOGGER.warning(f"YOLO detection failed: {e}")
        return []


def analyze_image_attributes(image: Image.Image) -> Dict[str, Any]:
    """Analyze image for lighting, time of day, weather conditions."""
    # Convert to numpy for analysis
    img_array = np.array(image)
    
    # Calculate brightness
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    
    # Determine time of day based on brightness
    if brightness < 60:
        time_of_day = "nighttime"
        lighting = "dark"
    elif brightness < 120:
        time_of_day = "dusk/dawn"
        lighting = "low light"
    else:
        time_of_day = "daytime"
        lighting = "bright" if brightness > 180 else "moderate"
    
    # Analyze color temperature (rough weather indicator)
    b_mean = np.mean(img_array[:, :, 2])  # Blue channel
    r_mean = np.mean(img_array[:, :, 0])  # Red channel
    
    if b_mean > r_mean + 15:
        weather = "overcast/cloudy"
    elif r_mean > b_mean + 15:
        weather = "clear/sunny"
    else:
        weather = "neutral"
    
    return {
        "brightness": float(brightness),
        "time_of_day": time_of_day,
        "lighting": lighting,
        "weather": weather,
    }


def generate_detailed_description(
    caption: str,
    detected_objects: List[ObjectTag],
    attributes: Dict[str, Any]
) -> str:
    """Generate a detailed scene description combining caption, objects, and attributes."""
    parts = []
    
    # Time of day
    parts.append(f"{attributes['time_of_day'].title()}")
    
    # Weather if relevant
    if attributes['weather'] != 'neutral':
        parts.append(f"{attributes['weather']}")
    
    # Main caption (cleaned up)
    caption_clean = caption.strip().rstrip('.')
    if caption_clean and caption_clean not in ['a', 'an', 'the']:
        parts.append(f"showing {caption_clean}")
    
    # Count and list objects by type
    if detected_objects:
        object_counts = {}
        for obj in detected_objects:
            label = obj.label
            object_counts[label] = object_counts.get(label, 0) + 1
        
        # Prioritize security-relevant objects
        priority_order = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'dog']
        other_objects = []
        
        for label in priority_order:
            if label in object_counts:
                count = object_counts[label]
                if count == 1:
                    parts.append(f"1 {label}")
                else:
                    plural = label + 's' if not label.endswith('s') else label
                    parts.append(f"{count} {plural}")
                del object_counts[label]
        
        # Add remaining objects
        for label, count in object_counts.items():
            if count == 1:
                other_objects.append(label)
            else:
                plural = label + 's' if not label.endswith('s') else label
                other_objects.append(f"{count} {plural}")
        
        if other_objects:
            parts.extend(other_objects[:3])  # Limit to top 3 other objects
    
    # If no objects detected, mention it
    if not detected_objects or len(detected_objects) == 0:
        parts.append("no people or vehicles detected")
    
    # Combine into natural sentence
    description = ", ".join(parts)
    
    # Capitalize first letter and add period
    if description:
        description = description[0].upper() + description[1:] + "."
    
    return description


def extract_objects_from_caption(caption: str) -> List[ObjectTag]:
    """Extract potential objects from caption text."""
    common_objects = [
        'person', 'people', 'man', 'woman', 'child',
        'car', 'truck', 'bus', 'vehicle', 'motorcycle', 'bicycle',
        'dog', 'cat', 'animal',
        'house', 'building', 'door', 'window',
        'tree', 'grass', 'sky', 'road'
    ]
    
    objects = []
    caption_lower = caption.lower()
    
    for obj in common_objects:
        if obj in caption_lower:
            objects.append(ObjectTag(
                label=obj if obj not in ['man', 'woman', 'child', 'people'] else 'person',
                confidence=0.6,
                source="caption"
            ))
    
    # Remove duplicates
    seen = set()
    unique_objects = []
    for obj in objects:
        if obj.label not in seen:
            seen.add(obj.label)
            unique_objects.append(obj)
    
    return unique_objects


@app.post("/describe")
async def describe(request: dict) -> DescribeResponse:
    """
    Generate image captions and extract objects.
    """
    start_time = time.time()
    
    try:
        image_b64 = request.get("image")
        if not image_b64:
            raise HTTPException(status_code=400, detail="Image required")
        
        model = request.get("model", "blip")
        use_cache = request.get("use_cache", True)
        
        # Check cache
        digest = hashlib.sha256(image_b64[:1000].encode()).hexdigest()[:16]
        
        LOGGER.info(f"Processing describe request with model={model}, digest={digest}")
        
        # Decode image
        pil_image = decode_image(image_b64)
        
        # Analyze image attributes (time of day, weather, lighting)
        attributes = analyze_image_attributes(pil_image)
        
        # Generate base caption from vision model
        if model == "git":
            base_caption = generate_caption_git(pil_image)
        else:  # default to BLIP
            base_caption = generate_caption_blip(pil_image)
        
        # Detect objects with YOLO (if available and requested)
        include_detections = request.get("include_detections", True)
        yolo_objects = detect_objects_yolo(pil_image) if include_detections else []
        
        # Extract objects from caption as backup
        caption_objects = extract_objects_from_caption(base_caption)
        
        # Merge: Prefer YOLO for accuracy, fall back to caption-based
        all_objects = yolo_objects if yolo_objects else caption_objects
        
        # Generate detailed description
        detailed_caption = generate_detailed_description(base_caption, all_objects, attributes)
        
        result = DescribeResult(
            caption=detailed_caption,  # Use enhanced caption
            objects=all_objects,
            model=model
        )
        
        elapsed = time.time() - start_time
        LOGGER.info(f"✅ Generated detailed caption in {elapsed:.2f}s: {detailed_caption}")
        LOGGER.info(f"   Base: {base_caption}")
        LOGGER.info(f"   Objects: {len(all_objects)}, Time: {attributes['time_of_day']}")
        
        return DescribeResponse(results=[result], cached=False)
        
    except Exception as e:
        LOGGER.error(f"Describe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match_criteria")
async def match_criteria(request: dict) -> MatchCriteriaResponse:
    """
    Match image against criteria using vision model.
    """
    start_time = time.time()
    
    try:
        image_b64 = request.get("image")
        criteria = request.get("criteria", "")
        
        if not image_b64:
            raise HTTPException(status_code=400, detail="Image required")
        if not criteria:
            raise HTTPException(status_code=400, detail="Criteria required")
        
        LOGGER.info(f"Matching criteria: {criteria}")
        
        # Decode image and generate caption
        pil_image = decode_image(image_b64)
        caption = generate_caption_blip(pil_image)
        
        # Simple keyword matching
        criteria_lower = criteria.lower()
        caption_lower = caption.lower()
        
        # Extract keywords from criteria
        keywords = [word.strip() for word in criteria_lower.split() if len(word.strip()) > 3]
        
        # Check for matches
        matches = sum(1 for keyword in keywords if keyword in caption_lower)
        confidence = min(0.95, (matches / max(len(keywords), 1)) * 1.2)
        
        if matches > 0:
            verdict = "present"
            reason = f"Caption: '{caption}'. Matched {matches}/{len(keywords)} criteria keywords."
        else:
            verdict = "absent"
            reason = f"Caption: '{caption}'. No criteria keywords matched."
        
        elapsed = time.time() - start_time
        LOGGER.info(f"✅ Criteria match in {elapsed:.2f}s: verdict={verdict}, conf={confidence:.2f}")
        
        return MatchCriteriaResponse(
            verdict=verdict,
            confidence=confidence,
            reason=reason
        )
        
    except Exception as e:
        LOGGER.error(f"Match criteria error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> HealthResponse:
    """Health check with model status."""
    models_loaded = {
        "blip": _blip_model is not None,
        "git": _git_model is not None,
    }
    
    status = "ok" if any(models_loaded.values()) or DEVICE == "cpu" else "degraded"
    
    return HealthResponse(status=status, models_loaded=models_loaded)


@app.get("/models")
async def models() -> Dict[str, Any]:
    """List available models."""
    models_loaded = {
        "blip": _blip_model is not None,
        "git": _git_model is not None,
    }
    
    return {
        "default_model": "blip",
        "default_composition": "blip",
        "models": [
            {"name": "blip", "available": True, "description": "Salesforce BLIP (990MB, accurate)"},
            {"name": "git", "available": True, "description": "Microsoft GIT (500MB, fast)"},
        ],
        "device": DEVICE,
        "models_loaded": models_loaded,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return """# HELP vision_requests_total Total requests
# TYPE vision_requests_total counter
vision_requests_total{endpoint="describe"} 0
vision_requests_total{endpoint="match_criteria"} 0
"""


if __name__ == "__main__":
    import uvicorn
    LOGGER.info("🚀 Starting vision service on http://127.0.0.1:8101")
    LOGGER.info(f"🖥️  Device: {DEVICE}")
    LOGGER.info("📥 Models will download automatically on first use (~500-990MB each)")
    uvicorn.run(app, host="0.0.0.0", port=8101, log_level="info")

