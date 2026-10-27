from __future__ import annotations

import base64
import hashlib
import io
import logging
import time
from typing import Dict, List, Optional

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Histogram, Counter, Gauge, generate_latest
from starlette.responses import Response

try:
    from .cache import ResultCache
    from .config import ServiceSettings, get_settings
    from .detector import Detection, load_detector
    from .runners import (
        Blip2Runner,
        DescribeOutput,
        MatchOutput,
        MiniCpmRunner,
        RunContext,
        VisionRunner,
        LlavaRunner,
    )
    from .schemas import (
        CriteriaVerdict,
        DescribeRequest,
        DescribeResponse,
        DescribeResult,
        HealthResponse,
        MatchCriteriaRequest,
        ObjectTag,
    )
except ImportError:
    from cache import ResultCache
    from config import ServiceSettings, get_settings
    from detector import Detection, load_detector
    from runners import (
        Blip2Runner,
        DescribeOutput,
        MatchOutput,
        MiniCpmRunner,
        RunContext,
        VisionRunner,
        LlavaRunner,
    )
    from schemas import (
        CriteriaVerdict,
        DescribeRequest,
        DescribeResponse,
        DescribeResult,
        HealthResponse,
        MatchCriteriaRequest,
        ObjectTag,
    )

LOGGER = logging.getLogger("vision_local")


class RunnerRegistry:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.runners: Dict[str, VisionRunner] = {}
        # Lazy load runners on demand
        # self._load_runners()

    def _ensure_catalog_install(self, name: str) -> None:
        try:
            from core.model_library.catalog import ensure_model_installed
        except Exception:
            return
        try:
            ensure_model_installed(name)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Model library install failed for %s: %s", name, exc)

    def _load_runner(self, name: str) -> Optional[VisionRunner]:
        try:
            if name == "blip2":
                if not self.settings.model_paths["blip2"].exists():
                    self._ensure_catalog_install("blip2")
                return Blip2Runner(
                    model_dir=self.settings.model_paths["blip2"],
                    device=self.settings.device,
                    max_new_tokens=self.settings.max_new_tokens,
                )
            elif name == "llava":
                if not self.settings.model_paths["llava"].exists():
                    self._ensure_catalog_install("llava")
                return LlavaRunner(
                    model_dir=self.settings.model_paths["llava"],
                    device=self.settings.device,
                    max_new_tokens=self.settings.max_new_tokens,
                )
            elif name == "minicpm":
                if not self.settings.model_paths["minicpm"].exists():
                    self._ensure_catalog_install("minicpm")
                return MiniCpmRunner(
                    model_dir=self.settings.model_paths["minicpm"],
                    device=self.settings.device,
                    max_new_tokens=self.settings.max_new_tokens,
                )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Failed to load %s: %s", name, exc)
        return None

    def get(self, name: str) -> VisionRunner:
        runner = self.runners.get(name)
        if runner is None:
            # Attempt to load
            LOGGER.info("Lazy loading runner: %s", name)
            runner = self._load_runner(name)
            if runner:
                self.runners[name] = runner
            else:
                raise KeyError(f"Runner {name} unavailable (failed to load)")
        return runner

    def availability(self) -> Dict[str, bool]:
        # Check loaded runners or model existence
        status = {}
        for name in ["blip2", "llava", "minicpm"]:
            if name in self.runners and self.runners[name] is not None:
                status[name] = True
            else:
                # Check if model path exists
                import os
                path = self.settings.model_paths.get(name)
                status[name] = os.path.exists(path) if path else False
        return status


def decode_image(image_b64: str) -> Image.Image:
    try:
        data = base64.b64decode(image_b64)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail="Invalid base64 payload") from exc
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail="Unsupported image format") from exc


def create_app(settings: Optional[ServiceSettings] = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Local Vision Service", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    histogram = Histogram(
        f"{settings.prometheus_namespace}_latency_ms",
        "Model inference latency in milliseconds",
        ["model", "action", "cached"],
        buckets=[bucket / 1000.0 for bucket in settings.metrics_bucket_ms],
    )
    counter = Counter(
        f"{settings.prometheus_namespace}_requests_total",
        "Total vision inference requests",
        ["model", "action", "cached"],
    )
    queue_depth = Gauge(
        f"{settings.prometheus_namespace}_queue_depth",
        "Approximate request queue depth",
    )

    cache = ResultCache(settings.cache_dir)
    registry = RunnerRegistry(settings)
    install_hint = (
        "Install services/vision_local/requirements.gpu.txt or run scripts/install-local-ai-extras.ps1 "
        "to enable local vision models."
    )

    @app.on_event("startup")
    async def _startup() -> None:
        LOGGER.info("Local vision service starting at %s:%s", settings.host, settings.port)

    @app.middleware("http")
    async def _queue_middleware(request, call_next):  # type: ignore
        queue_depth.inc()
        try:
            return await call_next(request)
        finally:
            queue_depth.dec()

    def build_context(detections: List[Detection], request_timeout: Optional[float]) -> RunContext:
        return RunContext(
            detections=[det.label for det in detections],
            timeout_s=request_timeout or settings.timeout_s,
            device=settings.device,
            metadata={"max_new_tokens": settings.max_new_tokens},
        )

    def run_describe(model: str, image: Image.Image, digest: str, context: RunContext, use_cache: bool) -> DescribeResult:
        cache_key = f"{digest}"
        cached_payload = cache.get(model, cache_key, "describe") if use_cache else None
        if cached_payload:
            counter.labels(model=model, action="describe", cached="true").inc()
            histogram.labels(model=model, action="describe", cached="true").observe(0.0)
            return DescribeResult(
                model=model,
                caption=cached_payload["caption"],
                objects=[ObjectTag(**obj) for obj in cached_payload["objects"]],
                latency_ms=cached_payload.get("latency_ms", 0.0),
                cached=True,
                device=cached_payload.get("device"),
                diagnostics=cached_payload.get("diagnostics", {}),
            )

        start = time.perf_counter()
        try:
            runner = registry.get(model)
        except KeyError as exc:
            raise HTTPException(status_code=503, detail=f"Model '{model}' is unavailable. {install_hint}") from exc
        output: DescribeOutput = runner.describe(image=image, context=context)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        histogram.labels(model=model, action="describe", cached="false").observe(elapsed_ms / 1000.0)
        counter.labels(model=model, action="describe", cached="false").inc()

        objects = [ObjectTag(label=tag, confidence=None, source="model") for tag in output.tags]
        result = DescribeResult(
            model=model,
            caption=output.caption,
            objects=objects,
            latency_ms=elapsed_ms,
            cached=False,
            device=settings.device,
            diagnostics={"cpu_percent": psutil.Process().cpu_percent(interval=None)},
        )
        cache.set(
            model,
            cache_key,
            "describe",
            {
                "caption": result.caption,
                "objects": [obj.dict() for obj in result.objects],
                "latency_ms": result.latency_ms,
                "device": result.device,
                "diagnostics": result.diagnostics,
            },
        )
        return result

    def run_match(model: str, image: Image.Image, digest: str, criteria: List[str], context: RunContext, use_cache: bool) -> CriteriaVerdict:
        cache_key = f"{digest}:{hashlib.sha256('||'.join(criteria).encode()).hexdigest()}"
        cached_payload = cache.get(model, cache_key, "match") if use_cache else None
        if cached_payload:
            counter.labels(model=model, action="match", cached="true").inc()
            histogram.labels(model=model, action="match", cached="true").observe(0.0)
            return CriteriaVerdict(
                model=model,
                verdict=cached_payload["verdict"],
                reasoning=cached_payload["reasoning"],
                latency_ms=cached_payload.get("latency_ms", 0.0),
                cached=True,
                device=cached_payload.get("device"),
            )

        start = time.perf_counter()
        try:
            runner = registry.get(model)
        except KeyError as exc:
            raise HTTPException(status_code=503, detail=f"Model '{model}' is unavailable. {install_hint}") from exc
        output: MatchOutput = runner.match_criteria(image=image, criteria=criteria, context=context)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        histogram.labels(model=model, action="match", cached="false").observe(elapsed_ms / 1000.0)
        counter.labels(model=model, action="match", cached="false").inc()

        verdict = CriteriaVerdict(
            model=model,
            verdict=output.verdict,
            reasoning=output.reasoning,
            latency_ms=elapsed_ms,
            cached=False,
            device=settings.device,
        )
        cache.set(
            model,
            cache_key,
            "match",
            {
                "verdict": verdict.verdict,
                "reasoning": verdict.reasoning,
                "latency_ms": verdict.latency_ms,
                "device": verdict.device,
            },
        )
        return verdict

    @app.post("/describe", response_model=DescribeResponse)
    async def describe(request: DescribeRequest) -> DescribeResponse:
        image = decode_image(request.image_b64)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        models = request.composition or ([request.model] if request.model else settings.default_composition)
        detections: List[Detection] = []
        if request.include_detections and settings.enable_detector:
            try:
                detector = load_detector(str(settings.detector_model), settings.device)
                detections = detector.detect(image)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Detector failed: %s", exc)
        context = build_context(detections, request.timeout_s)
        results = [run_describe(model, image, digest, context, request.use_cache) for model in models]
        aggregate_caption = results[0].caption if results else ""
        aggregate_objects = merge_objects(detections, results)
        return DescribeResponse(results=results, aggregate_caption=aggregate_caption, aggregate_objects=aggregate_objects)

    @app.post("/match_criteria", response_model=CriteriaVerdict)
    async def match_criteria(request: MatchCriteriaRequest) -> CriteriaVerdict:
        if not request.criteria:
            raise HTTPException(status_code=400, detail="Criteria list required")
        image = decode_image(request.image_b64)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        model = request.model or settings.default_model
        detections: List[Detection] = []
        if request.include_detections and settings.enable_detector:
            try:
                detector = load_detector(str(settings.detector_model), settings.device)
                detections = detector.detect(image)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Detector failed: %s", exc)
        context = build_context(detections, request.timeout_s)
        return run_match(model, image, digest, request.criteria, context, request.use_cache)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        availability = registry.availability()
        status = "ok" if all(availability.values()) else ("degraded" if any(availability.values()) else "error")
        return HealthResponse(status=status, models_loaded=availability)

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/models")
    async def models() -> Dict[str, Any]:
        availability = registry.availability()
        models = []
        for name, available in availability.items():
            models.append(
                {
                    "name": name,
                    "available": available,
                }
            )
        catalog_entries = []
        try:
            from core.model_library.catalog import list_catalog_entries

            for entry in list_catalog_entries(modality="image_to_text"):
                catalog_entries.append(
                    {
                        "id": entry.id,
                        "name": entry.display_name,
                        "installed": entry.is_installed(),
                        "requires_gpu": entry.requires_gpu,
                        "min_vram_gb": entry.min_vram_gb,
                        "size_gb": entry.size_gb,
                        "license_url": entry.license_url,
                    }
                )
        except Exception:
            catalog_entries = []
        return {
            "default_model": settings.default_model,
            "default_composition": settings.default_composition,
            "models": models,
            "catalog": catalog_entries,
        }

    return app


def merge_objects(detections: List[Detection], results: List[DescribeResult]) -> List[ObjectTag]:
    label_map: Dict[str, ObjectTag] = {}
    for detection in detections:
        label_map.setdefault(
            detection.label,
            ObjectTag(label=detection.label, confidence=detection.confidence, source="detector"),
        )
    for result in results:
        for obj in result.objects:
            label_map.setdefault(obj.label, ObjectTag(label=obj.label, confidence=obj.confidence, source="model"))
    return list(label_map.values())

