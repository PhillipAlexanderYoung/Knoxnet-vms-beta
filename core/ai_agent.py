import asyncio
import json
import logging
import time
import hashlib
import base64
import math
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import os
import requests
from abc import ABC, abstractmethod
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import IntentLearner (optional feature)
try:
    from core.llm_intent_learner import IntentLearner
    INTENT_LEARNER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"IntentLearner not available: {e}")
    IntentLearner = None
    INTENT_LEARNER_AVAILABLE = False


class AIProviderType(Enum):
    """Supported AI provider types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class AIContext:
    """Context sent to the AI agent"""
    devices: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    layout: List[Dict[str, Any]]
    overlays: Optional[List[Dict[str, Any]]] = None
    system_time: str = None
    user_intent: Optional[Dict[str, Any]] = None
    permissions: Optional[List[str]] = None
    # Per-request overrides (Desktop terminal / UI). Kept generic to stay forward-compatible.
    llm: Optional[Dict[str, Any]] = None
    vision: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.system_time is None:
            self.system_time = datetime.now().isoformat()


@dataclass
class AIAction:
    """Action that the AI agent can perform"""
    kind: str  # create_widget, create_camera_widget, create_all_cameras_grid, move_widget, resize_widget, update_widget, create_rule, camera_snapshot, reorganize_dashboard, detect_objects, setup_monitoring, interpret_zone_rules, recall_context
    widget_type: Optional[str] = None
    widget_id: Optional[str] = None
    props: Optional[Dict[str, Any]] = None
    position: Optional[Dict[str, int]] = None
    size: Optional[Dict[str, int]] = None
    rule_name: Optional[str] = None
    rule_when: Optional[str] = None
    rule_actions: Optional[List[Dict[str, Any]]] = None
    # New fields for enhanced functionality
    detection_model: Optional[str] = None  # 'yolov8' or 'mobilenet'
    camera_id: Optional[str] = None
    zone_id: Optional[str] = None
    natural_language_rule: Optional[str] = None
    object_types: Optional[List[str]] = None
    confidence_threshold: Optional[float] = None
    monitoring_enabled: Optional[bool] = None
    context_query: Optional[str] = None
    car_counting_camera_id: Optional[str] = None
    # Generic tool execution support
    tool_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class AIResponse:
    """Response from the AI agent"""
    message: str
    actions: List[AIAction] = None
    vision_analysis: Optional[str] = None
    vision_analysis_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    provider: Optional[str] = None  # Track which provider generated this response (e.g., 'openai', 'huggingface_local')
    model: Optional[str] = None     # Track which model was used (e.g., 'gpt-3.5-turbo', 'TinyLlama')

    def __post_init__(self):
        if self.actions is None:
            self.actions = []


@dataclass
class AITelemetry:
    """Telemetry data for AI operations"""
    prompt_hash: str
    provider_name: str
    model: str
    latency: float
    action_count: int
    success: bool
    vision_used: bool
    token_count: Optional[int] = None
    image_size: Optional[int] = None
    error: Optional[str] = None


class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], model: str, opts: Dict[str, Any]) -> Dict[str, Any]:
        """Send chat messages to the AI provider"""
        pass
    
    @abstractmethod
    async def vision(self, image: str, prompt: str, model: str, opts: Dict[str, Any]) -> Dict[str, Any]:
        """Send vision request to the AI provider"""
        pass


class HuggingFaceLocalProvider(AIProvider):
    """Local HuggingFace model provider via vLLM service"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8102"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
    
    async def chat(self, messages: List[Dict[str, str]], model: str, opts: Dict[str, Any]) -> Dict[str, Any]:
        """Send chat request to local LLM service"""
        try:
            logger.info(f"HuggingFace Local chat called with {len(messages)} messages")
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": opts.get("max_tokens", 512),
                "temperature": opts.get("temperature", 0.7),
                "top_p": opts.get("top_p", 0.9),
                "stream": False
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=opts.get("timeout", 60)
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Return in OpenAI-compatible format with content extraction
            return {
                "content": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {}),
                "model": result["model"],
                "choices": result["choices"]  # Keep original for compatibility
            }
            
        except Exception as e:
            logger.error(f"HuggingFace Local chat failed: {e}")
            raise
    
    async def vision(self, image: str, prompt: str, model: str, opts: Dict[str, Any]) -> Dict[str, Any]:
        """Vision is not supported by most local LLMs (except LLaVA)"""
        logger.warning("Vision not yet supported for local HuggingFace models")
        raise NotImplementedError("Vision analysis not supported by this provider")


class OpenAIProvider(AIProvider):
    """OpenAI-compatible provider"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        # Local vision service integration
        self.vision_endpoint = os.environ.get("LOCAL_VISION_ENDPOINT", "http://127.0.0.1:8101")
        self.use_local_vision = os.environ.get("USE_LOCAL_VISION", "true").lower() in {"true", "1", "yes"}
    
    async def chat(self, messages: List[Dict[str, str]], model: str, opts: Dict[str, Any]) -> Dict[str, Any]:
        """Send chat request to OpenAI"""
        try:
            logger.info(f"OpenAI chat called with {len(messages)} messages")

            # Debug: Check message structure
            for i, msg in enumerate(messages):
                try:
                    logger.info(
                        f"Message {i}: role={msg.get('role')}, content_length={len(str(msg.get('content', '')))}"
                    )
                except Exception:
                    logger.info("Message %d content length check failed", i)

            # Create a safe copy of messages to avoid any non-serializable content
            safe_messages: List[Dict[str, str]] = []
            for msg in messages:
                safe_messages.append({
                    "role": str(msg.get("role", "user")),
                    "content": str(msg.get("content", ""))
                })

            payload = {
                "model": model,
                "messages": safe_messages,
                "max_tokens": opts.get("max_tokens", 1000),
                "temperature": opts.get("temperature", 0.7),
                "stream": False
            }

            # Serialize payload explicitly to isolate JSON issues
            try:
                payload_json = json.dumps(payload, ensure_ascii=False)
                logger.info("Payload JSON size: %d chars", len(payload_json))
            except Exception as ser_err:
                import traceback as _tb
                logger.error("Failed to serialize OpenAI payload: %s\n%s", ser_err, _tb.format_exc())
                raise

            # Perform the HTTP request using pre-serialized JSON to avoid internal json recursion
            import eventlet
            real_threading = eventlet.patcher.original('threading')
            result_holder = {'response': None, 'error': None}
            
            def run_request():
                try:
                    result_holder['response'] = self.session.post(
                        f"{self.base_url}/chat/completions",
                        data=payload_json,
                        timeout=opts.get("timeout", 30)
                    )
                except Exception as req_err:
                    result_holder['error'] = req_err
            
            t = real_threading.Thread(target=run_request)
            t.start()
            t.join()
            
            if result_holder['error']:
                import traceback as _tb
                logger.error("OpenAI HTTP request failed: %s\n%s", result_holder['error'], _tb.format_exc())
                raise result_holder['error']
                
            response = result_holder['response']

            try:
                response.raise_for_status()
            except Exception as status_err:
                import traceback as _tb
                logger.error("OpenAI status error: %s, body=%s\n%s", status_err, response.text, _tb.format_exc())
                raise

            try:
                result = response.json()
            except Exception as parse_err:
                import traceback as _tb
                logger.error("Failed to parse OpenAI response JSON: %s, body=%s\n%s", parse_err, response.text[:500], _tb.format_exc())
                raise

            return {
                "content": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {}),
                "model": result["model"]
            }

        except Exception as e:
            import traceback as _tb
            logger.error("OpenAI chat error: %s\n%s", e, _tb.format_exc())
            raise
    
    async def vision(self, image: str, prompt: str, model: str, opts: Dict[str, Any]) -> Dict[str, Any]:
        """Send vision request to local service or cloud API based on source parameter"""
        try:
            # Remove data URL prefix if present
            image_b64 = image.split(",", 1)[1] if image.startswith("data:image/") else image
            
            # Allow user overrides for endpoints
            local_endpoint = opts.get("local_endpoint") or self.vision_endpoint
            cloud_endpoint = opts.get("cloud_endpoint") or self.base_url
            cloud_api_key = opts.get("cloud_api_key") or self.api_key
            
            # Check requested source (user override or default)
            requested_source = opts.get("source", "local" if self.use_local_vision else "cloud")
            
            logger.info(f"Vision request: source={requested_source}, model={model}")
            
            # Try local vision service if requested
            if requested_source in ("local", "dual"):
                try:
                    logger.info(f"Trying local vision service at {local_endpoint}")

                    # Local vision service uses its own model slugs (e.g., blip2/llava/minicpm),
                    # not OpenAI model ids like "gpt-4o". Prefer explicit local override or env.
                    local_model = (
                        opts.get("local_model")
                        or os.environ.get("LOCAL_VISION_MODEL")
                        or "blip2"
                    )
                    try:
                        from core.model_library.catalog import resolve_vision_model_slug

                        local_model = resolve_vision_model_slug(str(local_model))
                    except Exception:
                        local_model = str(local_model)
                    composition = opts.get("composition")
                    if isinstance(composition, str):
                        composition = [item.strip() for item in composition.split(",") if item.strip()]
                    
                    # Run in a separate thread to prevent blocking the event loop
                    import eventlet
                    real_threading = eventlet.patcher.original('threading')
                    result_holder = {'response': None, 'error': None}
                    
                    def run_local_request():
                        try:
                            result_holder['response'] = self.session.post(
                                f"{local_endpoint}/describe",
                                json={
                                    # Match services/vision_local/schemas.py: DescribeRequest
                                    "image_b64": image_b64,
                                    "model": local_model,
                                    "composition": composition,
                                    "include_detections": opts.get("include_detections", True),
                                    "use_cache": opts.get("use_cache", True),
                                    "timeout_s": opts.get("timeout_s") or opts.get("timeout", 60),
                                },
                                timeout=opts.get("timeout", 60)
                            )
                        except Exception as e:
                            result_holder['error'] = e
                    
                    t = real_threading.Thread(target=run_local_request)
                    t.start()
                    t.join()
                    
                    if result_holder['error']:
                        raise result_holder['error']
                        
                    local_response = result_holder['response']
                    
                    if local_response.status_code == 200:
                        local_data = local_response.json()
                        results = local_data.get("results", [])
                        if results and requested_source == "local":
                            # Local-only mode: return immediately
                            caption = local_data.get("aggregate_caption") or results[0].get("caption", "")
                            objects = local_data.get("aggregate_objects") or results[0].get("objects", [])
                            logger.info(f"✅ Local vision service responded: {str(caption)[:100]}...")
                            
                            return {
                                "content": caption,
                                "analysis": {
                                    "caption": caption,
                                    "objects": objects,
                                    "model": results[0].get("model", local_model),
                                    "source": "local"
                                },
                                "model": "local-blip",
                                "usage": {}
                            }
                        elif results and requested_source == "dual":
                            # Dual mode: local succeeded, skip cloud
                            caption = local_data.get("aggregate_caption") or results[0].get("caption", "")
                            objects = local_data.get("aggregate_objects") or results[0].get("objects", [])
                            logger.info(f"✅ Local (dual) responded: {str(caption)[:100]}...")
                            return {
                                "content": caption,
                                "analysis": {
                                    "caption": caption,
                                    "objects": objects,
                                    "model": results[0].get("model", local_model),
                                    "source": "local"
                                },
                                "model": "local-blip",
                                "usage": {}
                            }
                except Exception as local_err:
                    logger.warning(f"Local vision service failed: {local_err}")
                    # Only fallback if dual mode
                    if requested_source != "dual":
                        raise  # Re-raise if local was explicitly requested
            
            # Use cloud API (either requested directly or dual fallback)
            if requested_source in ("cloud", "dual"):
                logger.info(f"Using cloud vision API: {cloud_endpoint}, model: {model}")
                
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": opts.get("max_tokens", 800),
                    "temperature": opts.get("temperature", 0.3)
                }
                
                # Use custom headers if API key provided
                headers = {
                    "Authorization": f"Bearer {cloud_api_key}",
                    "Content-Type": "application/json"
                }
                
                # Run in a separate thread to prevent blocking the event loop
                import eventlet
                real_threading = eventlet.patcher.original('threading')
                result_holder = {'response': None, 'error': None}
                
                def run_cloud_request():
                    try:
                        result_holder['response'] = requests.post(
                            f"{cloud_endpoint}/chat/completions",
                            json=payload,
                            headers=headers,
                            timeout=opts.get("timeout", 60)
                        )
                    except Exception as e:
                        result_holder['error'] = e
                
                t = real_threading.Thread(target=run_cloud_request)
                t.start()
                t.join()
                
                if result_holder['error']:
                     # Re-raise with context, but ensure we unwrap exception if needed
                     logger.error(f"Cloud request failed in thread: {result_holder['error']}")
                     raise result_holder['error']
                     
                response = result_holder['response']
                
                # Handle API errors gracefully
                if response.status_code != 200:
                    error_body = response.text
                    if response.status_code == 429:
                        error_msg = "Cloud API rate limit exceeded or quota depleted. Please check your API plan and billing."
                    elif response.status_code == 401:
                        error_msg = "Cloud API authentication failed. Check your API key."
                    elif response.status_code == 400:
                        error_msg = f"Cloud API request error: {error_body[:200]}"
                    else:
                        error_msg = f"Cloud API error ({response.status_code}): {error_body[:200]}"
                    
                    logger.error(f"Cloud API error: {error_msg}")
                    
                    # Return structured error instead of crashing
                    return {
                        "content": f"Cloud API Error: {error_msg}",
                        "analysis": {
                            "caption": f"⚠️ {error_msg}",
                            "objects": [],
                            "model": model,
                            "source": "cloud",
                            "error": True
                        },
                        "usage": {},
                        "model": model,
                        "error": error_msg
                    }
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                logger.info(f"✅ Cloud vision API responded: {content[:100]}...")
                
                return {
                    "content": content,
                    "analysis": {
                        "caption": content,
                        "objects": [],  # Cloud doesn't provide structured objects
                        "model": model,
                        "source": "cloud"
                    },
                    "usage": result.get("usage", {}),
                    "model": result["model"]
                }
            
            # If we get here, no valid source was processed
            logger.error(f"No valid source processed (requested: {requested_source})")
            raise ValueError(f"No valid source processed (requested: {requested_source})")
            
        except Exception as e:
            logger.error(f"Vision error: {e}")
            raise


class AnthropicProvider(AIProvider):
    """Anthropic Claude provider"""

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        )

    @staticmethod
    def _convert_messages(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Anthropic expects user/assistant roles with text content arrays."""
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            converted.append({"role": role, "content": [{"type": "text", "text": str(content)}]})
        return converted

    async def chat(self, messages: List[Dict[str, str]], model: str, opts: Dict[str, Any]) -> Dict[str, Any]:
        try:
            payload = {
                "model": model,
                "messages": self._convert_messages(messages),
                "max_tokens": opts.get("max_tokens", 1000),
                "temperature": opts.get("temperature", 0.7),
                "stream": False,
            }
            response = self.session.post(
                f"{self.base_url}/messages", json=payload, timeout=opts.get("timeout", 60)
            )
            response.raise_for_status()
            data = response.json()
            content_blocks = data.get("content", [])
            text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
            combined = "\n".join([p for p in text_parts if p])
            return {"content": combined or data, "usage": data.get("usage", {}), "model": data.get("model", model)}
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            raise

    async def vision(self, image: str, prompt: str, model: str, opts: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("Anthropic vision not implemented; falling back to chat without image.")
        return await self.chat(
            messages=[{"role": "user", "content": f"{prompt}\n\n[vision omitted]"}],
            model=model,
            opts=opts,
        )


class LocalProvider(AIProvider):
    """Local/OSS AI provider (e.g., Ollama, LM Studio, local vision microservice)"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        self.vision_endpoint = os.environ.get("LOCAL_VISION_ENDPOINT", "http://127.0.0.1:8101")
        self.vision_model = os.environ.get("LOCAL_VISION_MODEL", "blip2")
        composition = os.environ.get("LOCAL_VISION_COMPOSITION", "blip2,llava,minicpm")
        self.vision_composition = [item.strip() for item in composition.split(",") if item.strip()]
        self.vision_source = os.environ.get("LOCAL_VISION_SOURCE", "local")

    async def chat(self, messages: List[Dict[str, str]], model: str, opts: Dict[str, Any]) -> Dict[str, Any]:
        """Send chat request to local provider"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": opts.get("temperature", 0.7),
                    "num_predict": opts.get("max_tokens", 1000),
                },
            }

            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=opts.get("timeout", 30),
            )
            response.raise_for_status()

            result = response.json()
            return {
                "content": result["message"]["content"],
                "usage": result.get("usage", {}),
                "model": result["model"],
            }

        except Exception as e:
            logger.error(f"Local provider chat error: {e}")
            raise

    def _should_use_local_vision(self, source: Optional[str]) -> bool:
        desired = (source or self.vision_source) or "local"
        return bool(self.vision_endpoint) and desired in {"local", "dual"}

    def _invoke_local_vision(
        self,
        image_b64: str,
        model: Optional[str],
        composition: Optional[List[str]],
        opts: Dict[str, Any],
    ) -> Dict[str, Any]:
        timeout = opts.get("timeout", 60)
        payload = {
            "image_b64": image_b64,
            "model": model or self.vision_model,
            "composition": composition or self.vision_composition,
            "include_detections": opts.get("include_detections", True),
            "use_cache": opts.get("use_cache", True),
            "timeout_s": timeout,
        }
        
        # Run in a separate thread to prevent blocking the event loop (since this is a sync request)
        import eventlet
        real_threading = eventlet.patcher.original('threading')
        result_holder = {'response': None, 'error': None}
        
        def run_request():
            try:
                result_holder['response'] = self.session.post(
                    f"{self.vision_endpoint}/describe",
                    json=payload,
                    timeout=timeout,
                )
            except Exception as e:
                result_holder['error'] = e
                
        t = real_threading.Thread(target=run_request)
        t.start()
        t.join()
        
        if result_holder['error']:
             logger.error(f"Local vision describe request failed in thread: {result_holder['error']}")
             raise result_holder['error']
             
        response = result_holder['response']
        response.raise_for_status()
        return response.json()

    def _invoke_local_match(
        self,
        image_b64: str,
        model: str,
        criteria: List[str],
        opts: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not criteria:
            return None
        timeout = opts.get("timeout", 60)
        payload = {
            "image_b64": image_b64,
            "criteria": criteria,
            "model": model,
            "include_detections": opts.get("include_detections", True),
            "use_cache": opts.get("use_cache", True),
            "timeout_s": timeout,
        }
        
        import eventlet
        real_threading = eventlet.patcher.original('threading')
        result_holder = {'response': None, 'error': None}
        
        def run_request():
            try:
                result_holder['response'] = self.session.post(
                    f"{self.vision_endpoint}/match_criteria",
                    json=payload,
                    timeout=timeout,
                )
            except Exception as e:
                result_holder['error'] = e
                
        t = real_threading.Thread(target=run_request)
        t.start()
        t.join()
        
        if result_holder['error']:
             logger.error(f"Local vision match request failed in thread: {result_holder['error']}")
             raise result_holder['error']

        response = result_holder['response']
        response.raise_for_status()
        return response.json()

    async def vision(self, image: str, prompt: str, model: str, opts: Dict[str, Any]) -> Dict[str, Any]:
        """Send vision request to local provider or the dedicated vision service."""
        try:
            source = opts.get("source")
            image_payload = image.split(",", 1)[1] if image.startswith("data:image/") else image

            composition = opts.get("composition")
            if isinstance(composition, str):
                composition = [item.strip() for item in composition.split(",") if item.strip()]

            criteria = opts.get("criteria") or []

            if self._should_use_local_vision(source):
                try:
                    describe_payload = self._invoke_local_vision(
                        image_b64=image_payload,
                        model=model,
                        composition=composition,
                        opts=opts,
                    )
                    aggregate_caption = describe_payload.get("aggregate_caption") or ""
                    aggregate_objects = describe_payload.get("aggregate_objects") or []
                    results = describe_payload.get("results") or []

                    verdict_payload = None
                    if criteria:
                        verdict_payload = self._invoke_local_match(
                            image_b64=image_payload,
                            model=results[0]["model"] if results else (model or self.vision_model),
                            criteria=criteria,
                            opts=opts,
                        )

                    summary = aggregate_caption or prompt or "Vision analysis completed."
                    return {
                        "content": summary,
                        "analysis": {
                            "caption": aggregate_caption,
                            "objects": aggregate_objects,
                            "by_model": results,
                            "verdict": verdict_payload,
                        },
                        "model": ",".join([result["model"] for result in results]) if results else model,
                    }
                except Exception as exc:
                    logger.warning(f"Local vision service failed, falling back to base provider: {exc}")

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "data": image_payload},
                        ],
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": opts.get("temperature", 0.3),
                    "num_predict": opts.get("max_tokens", 500),
                },
            }

            # Run in a separate thread to prevent blocking the event loop
            import eventlet
            real_threading = eventlet.patcher.original('threading')
            result_holder = {'response': None, 'error': None}
            
            def run_request():
                try:
                    result_holder['response'] = self.session.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                        timeout=opts.get("timeout", 60),
                    )
                except Exception as e:
                    result_holder['error'] = e
            
            t = real_threading.Thread(target=run_request)
            t.start()
            t.join()
            
            if result_holder['error']:
                 logger.error(f"Local chat request failed in thread: {result_holder['error']}")
                 raise result_holder['error']
                 
            response = result_holder['response']
            response.raise_for_status()

            result = response.json()
            return {
                "content": result["message"]["content"],
                "usage": result.get("usage", {}),
                "model": result["model"],
            }

        except Exception as e:
            logger.error(f"Local provider vision error: {e}")
            raise





class ObjectDetectionTool:
    """Handles YOLOv8 (.pt) and MobileNet SSD (.caffemodel) detection models.

    Uses the production detector implementation in `core.object_detector.ObjectDetector`.
    """
    
    def __init__(self, model_name: str = "mobilenet_iter_73000.caffemodel"):
        self.model_name = model_name
        # Look for models in the project root models directory
        self.model_path = Path(__file__).parent.parent / "models" / model_name
        self.model = None
        # Separate class sets for YOLO (COCO) and MobileNet SSD (VOC)
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.mobilenet_voc_classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the YOLOv8 (.pt) or MobileNet SSD (.caffemodel) model from models folder."""
        if not self.model_path.exists():
            logger.warning(f"Model file not found at {self.model_path}. Please ensure it's in the 'models' directory.")
            return
        
        try:
            from core.object_detector import ObjectDetector

            if self.model_name.endswith(".pt"):
                self.model = ObjectDetector(model_type="yolo", model_path=str(self.model_path))
                logger.info(f"✓ Loaded YOLO detector from {self.model_path}")
            elif self.model_name.endswith(".caffemodel"):
                # MobileNet SSD is selected by model_type; model_path is not required.
                self.model = ObjectDetector(model_type="mobilenet")
                logger.info("✓ Loaded MobileNet SSD detector")
            else:
                logger.warning(f"Unsupported model type: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using the loaded model.
        Returns a list of dictionaries, each containing 'class_name', 'confidence', 'bbox', and 'label'.
        """
        if self.model is None:
            logger.warning("Object detection model not loaded. Cannot perform detection.")
            return []
        
        try:
            # core.object_detector returns a normalized dict schema; adapt to AI Agent's legacy schema.
            dets = self.model.detect(image)  # type: ignore[attr-defined]
            out: List[Dict[str, Any]] = []
            for d in dets or []:
                bbox = d.get("bbox") or {}
                # Convert bbox dict -> [x, y, w, h] list
                try:
                    x = int(float(bbox.get("x", 0)))
                    y = int(float(bbox.get("y", 0)))
                    w = int(float(bbox.get("w", 0)))
                    h = int(float(bbox.get("h", 0)))
                except Exception:
                    x, y, w, h = 0, 0, 0, 0
                label = d.get("class") or d.get("label") or d.get("class_name")
                out.append(
                    {
                        "class_name": str(label) if label is not None else "",
                        "label": str(label) if label is not None else "",
                        "confidence": float(d.get("confidence", 0.0) or 0.0),
                        "bbox": [x, y, w, h],
                        "class_id": int(d.get("class_id", -1) or -1),
                    }
                )
            return out
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []


class AIAgent:
    """Main AI agent service"""
    
    def __init__(self):
        self.provider = None
        self.fallback_provider = None  # Local fallback when primary fails
        self.provider_chain: List[Dict[str, Any]] = []  # Ordered providers with metadata
        self.fallback_chain: List[Any] = []  # Provider objects after primary
        self.user_keys: Dict[str, Dict[str, Any]] = {}
        self.chat_model = os.environ.get("AI_CHAT_MODEL", "gpt-3.5-turbo")
        self.vision_model = os.environ.get("AI_VISION_MODEL", "gpt-4-vision-preview")
        self.timeout = int(os.environ.get("AI_TIMEOUT", "30"))
        self.rate_limit_requests = int(os.environ.get("AI_RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.environ.get("AI_RATE_LIMIT_WINDOW", "3600"))
        self.request_count = 0
        self.last_reset = time.time()
        self.provider_failures = 0  # Track consecutive failures
        self.last_failure_time = 0
        
        # Persistent context for better environment understanding
        self.environment_context = []
        self.max_context_entries = 10  # Keep last 10 vision analyses
        
        # Initialize object detection tools (default to MobileNet SSD)
        self.object_detection_tools = {
            'mobilenet': ObjectDetectionTool('mobilenet_iter_73000.caffemodel'),
            'yolov8': ObjectDetectionTool('yolov8n.pt'),
        }
        
        # Natural language rules storage
        self.natural_language_rules = {}
        self.zone_monitoring = {}
        self.context_memory = []
        
        # Conversation context tracking for natural references
        self.conversation_context = {
            'cameras': [],  # Recently mentioned cameras
            'objects': [],  # Recently mentioned objects
            'actions': [],  # Recently performed actions
            'last_camera': None,  # Most recent camera reference
            'last_objects': [],  # Most recent object search
            'session_start': time.time()
        }
        self.max_context_history = 10  # Keep last 10 items per category
        
        # Intent learner for pattern recognition (helps small/offline models)
        if INTENT_LEARNER_AVAILABLE and IntentLearner:
            try:
                self.intent_learner = IntentLearner()
                logger.info(f"Intent learner initialized: {self.intent_learner.get_stats()}")
            except Exception as e:
                logger.error(f"Failed to initialize intent learner: {e}")
                self.intent_learner = None
        else:
            logger.warning("Intent learner not available, pattern learning disabled")
            self.intent_learner = None
        
        # Stream server reference for detection queries
        self.stream_server = None
        
        self._initialize_provider()
    
    def set_stream_server(self, stream_server):
        """Set reference to stream server for detection queries"""
        self.stream_server = stream_server
        logger.info("Stream server reference set for AI agent detection queries")
    
    def _load_user_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load user-provided API keys for providers."""
        try:
            path = Path("data/llm_user_keys.json")
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception as e:
            logger.warning(f"Could not load user keys: {e}")
        return {}

    def _resolve_key_candidates(self, provider_id: str) -> List[Dict[str, str]]:
        """Return ordered key candidates (user first, then env) for a provider."""
        candidates: List[Dict[str, str]] = []
        user_key = self.user_keys.get(provider_id, {}).get("api_key")
        if user_key:
            candidates.append({"source": "user", "api_key": user_key})

        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "grok": "GROK_API_KEY",
            "xai": "GROK_API_KEY",
            "azure_openai": "AZURE_OPENAI_KEY",
        }
        env_var = env_var_map.get(provider_id, "")
        if env_var:
            env_key = os.environ.get(env_var)
            if env_key and env_key != user_key:
                candidates.append({"source": "env", "api_key": env_key})
        return candidates

    def _create_provider_instance(self, provider_id: str, api_key: Optional[str], cfg: Dict[str, Any]):
        """Instantiate provider by id and key."""
        try:
            provider_id = provider_id.lower()
            if provider_id == "openai":
                if not api_key:
                    return None
                base_url = cfg.get("base_url") or "https://api.openai.com/v1"
                return OpenAIProvider(api_key=api_key, base_url=base_url)
            if provider_id in {"grok", "xai"}:
                if not api_key:
                    return None
                base_url = cfg.get("base_url") or "https://api.x.ai/v1"
                return OpenAIProvider(api_key=api_key, base_url=base_url)
            if provider_id == "anthropic":
                if not api_key:
                    return None
                base_url = cfg.get("base_url") or "https://api.anthropic.com/v1"
                return AnthropicProvider(api_key=api_key, base_url=base_url)
            if provider_id == "azure_openai":
                if not api_key:
                    return None
                base_url = cfg.get("base_url") or os.environ.get("AZURE_OPENAI_ENDPOINT")
                return OpenAIProvider(api_key=api_key, base_url=base_url)
            if provider_id in {"huggingface_local", "local_llm"}:
                service_url = cfg.get("service_url") or cfg.get("base_url")
                if not service_url:
                    llm_service_host = os.environ.get("LLM_SERVICE_HOST", "127.0.0.1")
                    llm_service_port = os.environ.get("LLM_SERVICE_PORT", "8102")
                    service_url = f"http://{llm_service_host}:{llm_service_port}"
                return HuggingFaceLocalProvider(base_url=service_url)
            if provider_id == "local":
                base_url = cfg.get("base_url") or "http://localhost:11434"
                return LocalProvider(base_url=base_url)
        except Exception as e:
            logger.error(f"Failed to create provider {provider_id}: {e}")
            return None
        return None

    def _apply_provider_chain(self, chain: List[Dict[str, Any]]):
        """Apply provider chain to agent state."""
        self.provider_chain = chain
        self.provider = chain[0]["provider"] if chain else None
        self.fallback_chain = [entry["provider"] for entry in chain[1:]] if len(chain) > 1 else []
        self.fallback_provider = self.fallback_chain[0] if self.fallback_chain else None

    def reload_providers(self) -> Dict[str, Any]:
        """Reload providers from config and user keys."""
        try:
            self.provider = None
            self.fallback_provider = None
            self.provider_chain = []
            self.fallback_chain = []
            self._initialize_provider()
            return self.get_status()
        except Exception as e:
            logger.error(f"Failed to reload providers: {e}")
            return {"error": str(e)}

    def _initialize_provider(self):
        """Initialize the AI provider based on environment variables with fallback support"""
        # Load fallback mode from config file if available
        fallback_mode = "hybrid"  # default
        provider_priority = None
        providers_cfg: Dict[str, Any] = {}
        try:
            from pathlib import Path
            config_path = Path('data/llm_config.json')
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    fallback_mode = config.get('fallback_mode', 'hybrid')
                    self.config = config
                    # Load provider priority if present
                    provider_priority = config.get("provider_priority")
                    providers_cfg = config.get("providers", {})
                    logger.info(f"Loaded fallback_mode from config: {fallback_mode}")
        except Exception as e:
            logger.warning(f"Could not load fallback mode from config: {e}")
        
        provider_type = os.environ.get("AI_PROVIDER", "openai").lower()
        base_url = os.environ.get("AI_BASE_URL")
        
        # Determine fallback behavior based on fallback_mode
        if fallback_mode == "local_only":
            enable_fallback = False  # Don't try API at all
            provider_type = "huggingface_local"  # Force local
            logger.info("Fallback mode: LOCAL ONLY - using local LLM exclusively")
        elif fallback_mode == "api_only":
            enable_fallback = False  # Don't fall back to local
            logger.info("Fallback mode: API ONLY - no local fallback")
        else:  # hybrid
            enable_fallback = os.environ.get("ENABLE_LOCAL_FALLBACK", "true").lower() in {"true", "1", "yes"}
            logger.info(f"Fallback mode: HYBRID - API with local fallback (enabled={enable_fallback})")

        # --- New provider priority chain support ---
        self.user_keys = self._load_user_keys()
        provider_chain: List[Dict[str, Any]] = []
        priority_list = provider_priority or [provider_type]

        for pid in priority_list:
            pid_lower = str(pid).lower()
            cfg_entry = providers_cfg.get(pid_lower, {})
            if cfg_entry is not None and not cfg_entry.get("enabled", True):
                continue

            # API-based providers: try user key then env key
            key_candidates = self._resolve_key_candidates(pid_lower)

            if pid_lower in {"openai", "grok", "xai", "anthropic", "azure_openai"}:
                for kc in key_candidates:
                    provider_obj = self._create_provider_instance(pid_lower, kc.get("api_key"), cfg_entry)
                    if provider_obj:
                        provider_chain.append(
                            {"id": pid_lower, "provider": provider_obj, "source": kc.get("source", "env")}
                        )
                # If no key worked, skip to next
                continue

            # Keyless providers (local)
            provider_obj = self._create_provider_instance(pid_lower, None, cfg_entry)
            if provider_obj:
                provider_chain.append({"id": pid_lower, "provider": provider_obj, "source": "local"})

        # Add legacy fallback defaults when no explicit priority provided
        if provider_chain and not provider_priority and enable_fallback:
            try:
                if provider_type == AIProviderType.OPENAI.value:
                    llm_service_host = os.environ.get("LLM_SERVICE_HOST", "127.0.0.1")
                    llm_service_port = os.environ.get("LLM_SERVICE_PORT", "8102")
                    llm_service_url = f"http://{llm_service_host}:{llm_service_port}"
                    try:
                        health_check = requests.get(f"{llm_service_url}/health", timeout=2)
                        if health_check.status_code == 200:
                            fb_provider = HuggingFaceLocalProvider(base_url=llm_service_url)
                            provider_chain.append({"id": "huggingface_local", "provider": fb_provider, "source": "local"})
                            logger.info(f"✓ Added local LLM fallback: {llm_service_url}")
                    except Exception:
                        fb_provider = LocalProvider(base_url="http://localhost:11434")
                        provider_chain.append({"id": "local", "provider": fb_provider, "source": "local"})
                        logger.info("Added Ollama local fallback provider")
                elif provider_type == "huggingface_local":
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if api_key:
                        fb_provider = OpenAIProvider(api_key=api_key, base_url="https://api.openai.com/v1")
                        provider_chain.append({"id": "openai", "provider": fb_provider, "source": "env"})
                        logger.info("Added OpenAI fallback provider for local LLM")
            except Exception as fb_err:
                logger.warning(f"Could not add legacy fallback: {fb_err}")

        if provider_chain:
            self._apply_provider_chain(provider_chain)
            logger.info(
                "AI agent initialized with provider chain: %s",
                " -> ".join([f"{p['id']}({p['source']})" for p in provider_chain]),
            )
            return
        
        if provider_type == "huggingface_local":
            # Local HuggingFace via vLLM service
            llm_service_host = os.environ.get("LLM_SERVICE_HOST", "127.0.0.1")
            llm_service_port = os.environ.get("LLM_SERVICE_PORT", "8102")
            service_url = base_url or f"http://{llm_service_host}:{llm_service_port}"
            
            self.provider = HuggingFaceLocalProvider(base_url=service_url)
            self.chat_model = os.environ.get("LLM_DEFAULT_MODEL", self.chat_model)
            logger.info(f"AI agent initialized with local HuggingFace provider: {service_url}")
            
            # Initialize OpenAI fallback if enabled and key is present
            if enable_fallback:
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    self.fallback_provider = OpenAIProvider(
                        api_key=api_key,
                        base_url="https://api.openai.com/v1"
                    )
                    logger.info("✓ OpenAI fallback initialized for local provider")
            
        elif provider_type == AIProviderType.OPENAI.value:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found, AI agent will be disabled")
                return
            
            self.provider = OpenAIProvider(
                api_key=api_key,
                base_url=base_url or "https://api.openai.com/v1"
            )
            
            # Initialize local LLM fallback for when OpenAI fails or is rate limited
            if enable_fallback:
                try:
                    # Check if local LLM service is available
                    llm_service_host = os.environ.get("LLM_SERVICE_HOST", "127.0.0.1")
                    llm_service_port = os.environ.get("LLM_SERVICE_PORT", "8102")
                    llm_service_url = f"http://{llm_service_host}:{llm_service_port}"
                    
                    # Test if service is available
                    try:
                        import requests
                        health_check = requests.get(f"{llm_service_url}/health", timeout=2)
                        if health_check.status_code == 200:
                            self.fallback_provider = HuggingFaceLocalProvider(base_url=llm_service_url)
                            logger.info(f"✓ Local LLM fallback initialized: {llm_service_url}")
                        else:
                            raise Exception("Health check failed")
                    except:
                        # Fall back to Ollama if LLM service not available
                        self.fallback_provider = LocalProvider(base_url="http://localhost:11434")
                        logger.info("Local Ollama fallback provider initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize fallback provider: {e}")
            
        elif provider_type == AIProviderType.LOCAL.value:
            self.provider = LocalProvider(
                base_url=base_url or "http://localhost:11434"
            )
            self.vision_model = os.environ.get("LOCAL_VISION_MODEL", self.vision_model)
            
            # Initialize OpenAI fallback if enabled and key is present
            if enable_fallback:
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    self.fallback_provider = OpenAIProvider(
                        api_key=api_key,
                        base_url="https://api.openai.com/v1"
                    )
                    logger.info("✓ OpenAI fallback initialized for local provider")
        
        elif provider_type == "azure_openai":
            # Azure OpenAI support
            api_key = os.environ.get("AZURE_OPENAI_KEY")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            
            if not api_key or not endpoint:
                logger.warning("Azure OpenAI credentials not found, AI agent will be disabled")
                return
            
            self.provider = OpenAIProvider(
                api_key=api_key,
                base_url=endpoint
            )
            logger.info(f"AI agent initialized with Azure OpenAI provider")
        
        elif provider_type == "anthropic":
            # Anthropic Claude support (future implementation)
            logger.warning("Anthropic provider not yet implemented, falling back to OpenAI")
            provider_type = "openai"
            # Recursively initialize with fallback
            os.environ["AI_PROVIDER"] = "openai"
            return self._initialize_provider()
            
        else:
            logger.warning(f"Unsupported AI provider: {provider_type}")
            return
        
        logger.info(f"AI agent initialized with provider: {provider_type}")

    def _build_provider_chain_from_priority(self, priority_list: List[str]) -> List[Dict[str, Any]]:
        """
        Build an ordered provider chain from a priority list without mutating agent global state.
        Best-effort: skips providers without keys or that fail to instantiate.
        """
        providers_cfg: Dict[str, Any] = {}
        try:
            providers_cfg = (getattr(self, "config", {}) or {}).get("providers", {}) or {}
        except Exception:
            providers_cfg = {}

        chain: List[Dict[str, Any]] = []
        for pid in priority_list or []:
            pid_lower = str(pid or "").strip().lower()
            if not pid_lower:
                continue

            cfg_entry = providers_cfg.get(pid_lower, {})
            if isinstance(cfg_entry, dict) and cfg_entry.get("enabled") is False:
                continue

            if pid_lower in {"openai", "grok", "xai", "anthropic", "azure_openai"}:
                for kc in self._resolve_key_candidates(pid_lower):
                    provider_obj = self._create_provider_instance(pid_lower, kc.get("api_key"), cfg_entry if isinstance(cfg_entry, dict) else {})
                    if provider_obj:
                        chain.append({"id": pid_lower, "provider": provider_obj, "source": kc.get("source", "env")})
                        break
                continue

            provider_obj = self._create_provider_instance(pid_lower, None, cfg_entry if isinstance(cfg_entry, dict) else {})
            if provider_obj:
                chain.append({"id": pid_lower, "provider": provider_obj, "source": "local"})

        return chain

    def _resolve_request_overrides(self, context: AIContext) -> Dict[str, Any]:
        """
        Resolve per-request overrides from context.llm / context.vision.
        Returns dict with:
          - provider, provider_id, fallback(list), chat_model, vision_model
        """
        provider_for_request = self.provider
        provider_id_for_request = None
        try:
            if self.provider_chain:
                provider_id_for_request = self.provider_chain[0].get("id")
        except Exception:
            provider_id_for_request = None

        fallback_for_request: List[Any] = []
        try:
            fallback_for_request = list(self.fallback_chain or [])
            if not fallback_for_request and self.fallback_provider:
                fallback_for_request = [self.fallback_provider]
        except Exception:
            fallback_for_request = []

        chat_model = self.chat_model
        vision_model = self.vision_model

        llm_over = getattr(context, "llm", None) if context else None
        if isinstance(llm_over, dict):
            model_override = llm_over.get("model")
            if isinstance(model_override, str) and model_override.strip():
                chat_model = model_override.strip()

            provider_first = llm_over.get("provider")
            priority = llm_over.get("provider_priority") or llm_over.get("providerPriority")
            priority_list: List[str] = []
            if isinstance(priority, list):
                priority_list = [str(p).strip().lower() for p in priority if isinstance(p, (str, int, float)) and str(p).strip()]
            if isinstance(provider_first, str) and provider_first.strip():
                pf = provider_first.strip().lower()
                priority_list = [pf] + [p for p in priority_list if p != pf]
            if priority_list:
                chain = self._build_provider_chain_from_priority(priority_list)
                if chain:
                    provider_for_request = chain[0]["provider"]
                    provider_id_for_request = chain[0].get("id")
                    fallback_for_request = [entry["provider"] for entry in chain[1:]]

        vis_over = getattr(context, "vision", None) if context else None
        if isinstance(vis_over, dict):
            vm = vis_over.get("model")
            if isinstance(vm, str) and vm.strip():
                vision_model = vm.strip()

        return {
            "provider": provider_for_request,
            "provider_id": provider_id_for_request,
            "fallback": fallback_for_request,
            "chat_model": chat_model,
            "vision_model": vision_model,
        }
    
    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.last_reset > self.rate_limit_window:
            self.request_count = 0
            self.last_reset = current_time
        
        if self.request_count >= self.rate_limit_requests:
            return False
        
        self.request_count += 1
        return True
    
    def _create_prompt_hash(self, messages: List[Dict[str, str]], context: Optional[AIContext] = None) -> str:
        """Create a hash of the prompt for telemetry"""
        try:
            # Create a simplified context for hashing to avoid recursion issues
            context_data = None
            if context:
                context_data = {
                    "device_count": len(context.devices) if context.devices else 0,
                    "connection_count": len(context.connections) if context.connections else 0,
                    "layout_count": len(context.layout) if context.layout else 0,
                    "system_time": context.system_time,
                    "has_user_intent": bool(context.user_intent),
                    "permission_count": len(context.permissions) if context.permissions else 0
                }
            
            # Create a simplified version of messages to avoid recursion
            simplified_messages = []
            for msg in messages:
                simplified_msg = {
                    "role": msg.get("role", "unknown"),
                    "content_length": len(str(msg.get("content", "")))
                }
                simplified_messages.append(simplified_msg)
            
            prompt_data = {
                "messages": simplified_messages,
                "context": context_data
            }
            prompt_str = json.dumps(prompt_data, sort_keys=True)
            return hashlib.sha256(prompt_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.error(f"Error creating prompt hash: {e}")
            # Fallback to a simple hash
            try:
                # Just hash the message count and context summary
                summary = f"messages:{len(messages)},context:{bool(context)}"
                return hashlib.sha256(summary.encode()).hexdigest()[:16]
            except:
                return hashlib.sha256("fallback".encode()).hexdigest()[:16]
    
    def _parse_ai_response(self, content: str) -> AIResponse:
        """Parse AI response and extract actions"""
        try:
            logger.info(f"Parsing AI response: {content[:200]}...")
            
            # Try to parse as JSON first - handle both raw JSON and markdown-wrapped JSON
            cleaned_content = content.strip()
            
            # Check if it's markdown-wrapped JSON
            if cleaned_content.startswith("```json") and cleaned_content.endswith("```"):
                logger.info("Content appears to be markdown-wrapped JSON, extracting...")
                # Extract JSON from markdown code block
                json_start = cleaned_content.find("```json") + 7
                json_end = cleaned_content.rfind("```")
                cleaned_content = cleaned_content[json_start:json_end].strip()
                logger.info(f"Extracted JSON: {cleaned_content[:200]}...")
            
            if cleaned_content.startswith("{"):
                logger.info("Content appears to be JSON, attempting to parse...")
                data = json.loads(cleaned_content)
                logger.info(f"Parsed JSON data: {data}")
                
                message = data.get("message", content)
                actions_data = data.get("actions", [])
                logger.info(f"Found message: {message}")
                logger.info(f"Found actions data: {actions_data}")
                
                actions = []
                for action_data in actions_data:
                    logger.info(f"Processing action: {action_data}")
                    
                    # Extract kind/tool - handle multiple formats
                    kind = action_data.get("kind") or action_data.get("tool") or action_data.get("action")
                    params = action_data.get("parameters") or action_data.get("params") or action_data.get("props") or {}

                    # --- Guardrails: resolve ambiguous camera references using conversation context
                    # The LLM sometimes omits cameraRef on follow-ups like "enable motion boxes".
                    # If we have a last_camera from the conversation, fill it in so tool executors
                    # (PyQt / React) can proceed deterministically.
                    try:
                        if isinstance(params, dict):
                            # Normalize camera ref variants -> cameraRef
                            if "cameraRef" not in params:
                                cam = params.get("camera") or params.get("camera_id") or params.get("cameraId")
                                if cam:
                                    params["cameraRef"] = cam

                            # Normalize object classes variants -> objectClasses
                            if "objectClasses" not in params:
                                oc = params.get("object_classes") or params.get("classes")
                                if oc:
                                    params["objectClasses"] = oc

                            kind_l = (kind or "").lower().strip()
                            tool_id_l = str(action_data.get("tool_id") or action_data.get("toolId") or "").lower().strip()
                            is_execute_tool = kind_l == "execute_tool" or tool_id_l in {"snapshot_detect", "motion_detector_watch", "stop_motion_detector_watch", "take_camera_snapshot"}

                            if is_execute_tool and not params.get("cameraRef"):
                                last_cam = self.conversation_context.get("last_camera")
                                if last_cam:
                                    params["cameraRef"] = last_cam
                    except Exception:
                        pass
                    
                    # If it's EXECUTE_TOOL, extract the actual tool name
                    if kind == "EXECUTE_TOOL" or not kind:
                        kind = action_data.get("tool") or action_data.get("tool_name")
                        
                        # Infer tool from parameters if still not found
                        if not kind or kind == "EXECUTE_TOOL":
                            if 'cameraRef' in params or 'camera_id' in params:
                                kind = 'create_camera_widget'
                                logger.info(f"  Inferred tool 'create_camera_widget' from cameraRef parameter")
                            elif 'script' in params or 'script_content' in params:
                                kind = 'create_python_script'
                                logger.info(f"  Inferred tool 'create_python_script' from script parameter")
                            elif 'all_cameras' in params or action_data.get('all_cameras'):
                                kind = 'create_all_cameras_grid'
                                logger.info(f"  Inferred tool 'create_all_cameras_grid' from all_cameras parameter")
                            elif 'analysisPrompt' in params:
                                # This is likely a camera snapshot with analysis
                                kind = 'take_camera_snapshot'
                                logger.info(f"  Inferred tool 'take_camera_snapshot' from analysisPrompt parameter")
                            else:
                                kind = action_data.get("kind", "unknown_tool")
                                logger.warning(f"  Could not infer tool from parameters: {list(params.keys())}")
                    
                    action = AIAction(
                        kind=kind,
                        widget_type=action_data.get("widget_type"),
                        widget_id=action_data.get("widget_id"),
                        props=action_data.get("props"),
                        position=action_data.get("position"),
                        size=action_data.get("size"),
                        rule_name=action_data.get("rule_name"),
                        rule_when=action_data.get("rule_when"),
                        rule_actions=action_data.get("rule_actions"),
                        # Extended fields mapped through
                        detection_model=action_data.get("detection_model"),
                        camera_id=action_data.get("camera_id") or action_data.get("cameraId"),
                        zone_id=action_data.get("zone_id") or action_data.get("zoneId"),
                        natural_language_rule=action_data.get("natural_language_rule"),
                        object_types=action_data.get("object_types"),
                        confidence_threshold=action_data.get("confidence_threshold"),
                        monitoring_enabled=action_data.get("monitoring_enabled") if action_data.get("monitoring_enabled") is not None else action_data.get("enabled"),
                        context_query=action_data.get("context_query"),
                        car_counting_camera_id=action_data.get("car_counting_camera_id"),
                        tool_id=action_data.get("tool_id") or action_data.get("toolId") or kind,
                        parameters=params
                    )
                    actions.append(action)
                    logger.info(f"Created action: kind={kind}, tool_id={action.tool_id}")
                
                # Get provider name for feedback tracking
                provider_name = type(self.provider).__name__.replace('Provider', '').lower() if self.provider else 'unknown'
                if provider_name == 'openai':
                    provider_name = 'OpenAI'
                elif provider_name == 'huggingfacelocal':
                    provider_name = 'Local LLM'
                    
                result = AIResponse(
                    message=message, 
                    actions=actions,
                    provider=provider_name,
                    model=self.chat_model
                )
                logger.info(f"Final parsed response: message='{result.message}', actions={len(result.actions)}, provider={provider_name}")
                return result
            
            # Additional fallback: try to find JSON anywhere in the content
            logger.info("Content doesn't start with {, looking for JSON anywhere in content...")
            
            # Try to find JSON object anywhere in the text
            import re
            json_match = re.search(r'\{.*?"message".*?"actions".*?\}', content, re.DOTALL)
            if json_match:
                logger.info("Found JSON pattern in content, attempting to parse...")
                try:
                    potential_json = json_match.group()
                    data = json.loads(potential_json)
                    logger.info(f"Successfully parsed embedded JSON: {data}")
                    
                    message = data.get("message", content)
                    actions_data = data.get("actions", [])
                    
                    actions = []
                    for action_data in actions_data:
                        action = AIAction(
                            kind=action_data["kind"],
                            widget_type=action_data.get("widget_type"),
                            props=action_data.get("props"),
                            position=action_data.get("position"),
                            size=action_data.get("size"),
                            rule_name=action_data.get("rule_name"),
                            rule_when=action_data.get("rule_when"),
                            rule_actions=action_data.get("rule_actions")
                        )
                        actions.append(action)
                    
                    return AIResponse(message=message, actions=actions)
                except Exception as e:
                    logger.error(f"Failed to parse embedded JSON: {e}")
            
            # Final fallback: treat as plain text
            logger.info("No valid JSON found, treating as plain text")
            return AIResponse(message=content)
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return AIResponse(message=content, error="Failed to parse AI response")
    
    def _extract_camera_references(self, message: str, available_cameras: List[Dict]) -> List[str]:
        """Extract camera name references from user message."""
        message_lower = message.lower()
        referenced_cameras = []
        
        for camera in available_cameras:
            camera_name = camera.get('name', '')
            if not camera_name:
                continue
            
            # Check for direct name match (case insensitive)
            if camera_name.lower() in message_lower:
                referenced_cameras.append(camera_name)
                continue
            
            # Check for partial matches (e.g., "primary entrance" matches "Primary Entrance")
            name_words = camera_name.lower().split()
            if any(word in message_lower for word in name_words if len(word) > 3):
                referenced_cameras.append(camera_name)
        
        return referenced_cameras
    
    def _extract_object_references(self, message: str) -> List[str]:
        """Extract object type references from user message."""
        import re
        
        message_lower = message.lower()
        found_objects = []
        
        # Pattern 1: "how many X" or "count X" or "look for X" or "detect X"
        patterns = [
            r'how many (\w+)',
            r'count (\w+)',
            r'look for (\w+)',
            r'detect (\w+)',
            r'find (\w+)',
            r'check for (\w+)',
            r'any (\w+)',
            r'are there (\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, message_lower)
            for match in matches:
                # Clean up common noise words
                if match not in ['the', 'a', 'an', 'at', 'in', 'on', 'of', 'for', 'to', 'from', 'with', 'camera', 'cameras']:
                    # Normalize plural to singular for common cases
                    obj = match.rstrip('s') if match.endswith('s') and len(match) > 4 else match
                    found_objects.append(obj)
        
        # Also check for common object keywords for better coverage
        common_objects = [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle',
            'person', 'people', 'human',
            'dog', 'cat', 'bird',
            'bench', 'chair', 'table',
            'laptop', 'phone', 'computer',
            'wheelbarrow', 'shovel', 'tool'
        ]
        
        for obj in common_objects:
            if obj in message_lower:
                normalized = 'person' if obj in ['people', 'human'] else obj
                if normalized not in found_objects:
                    found_objects.append(normalized)
        
        return list(set(found_objects))  # Remove duplicates
    
    def _update_conversation_context(self, message: str, context: AIContext, actions: List[AIAction]):
        """Update conversation context with extracted information."""
        # Extract camera references from available cameras
        available_cameras = context.devices if context.devices else []
        camera_refs = self._extract_camera_references(message, available_cameras)
        
        # Extract object references
        object_refs = self._extract_object_references(message)
        
        # Update camera context
        for camera in camera_refs:
            if camera not in self.conversation_context['cameras']:
                self.conversation_context['cameras'].append(camera)
        
        # Keep only recent cameras
        if len(self.conversation_context['cameras']) > self.max_context_history:
            self.conversation_context['cameras'] = self.conversation_context['cameras'][-self.max_context_history:]
        
        # Update last camera if any found
        if camera_refs:
            self.conversation_context['last_camera'] = camera_refs[-1]

        # Also learn camera context from tool actions (important for follow-ups when the user omits camera name)
        # Example: user says "how many cars" → we choose Active camera internally; next message "behind the tree"
        # should inherit that camera even though it wasn't mentioned in text.
        action_cam_refs: List[str] = []
        try:
            for a in actions or []:
                if not a:
                    continue
                # Prefer explicit action camera fields
                cam = getattr(a, "camera_id", None) or getattr(a, "car_counting_camera_id", None)
                if isinstance(cam, str) and cam.strip():
                    action_cam_refs.append(cam.strip())
                    continue
                # Fall back to parameters
                params = getattr(a, "parameters", None)
                if isinstance(params, dict):
                    cam2 = params.get("cameraRef") or params.get("camera") or params.get("camera_id") or params.get("cameraId")
                    if isinstance(cam2, str) and cam2.strip():
                        action_cam_refs.append(cam2.strip())
        except Exception:
            action_cam_refs = []

        for cam in action_cam_refs:
            if cam and cam not in self.conversation_context['cameras']:
                self.conversation_context['cameras'].append(cam)
        if action_cam_refs:
            self.conversation_context['last_camera'] = action_cam_refs[-1]
        
        # Update object context
        for obj in object_refs:
            if obj not in self.conversation_context['objects']:
                self.conversation_context['objects'].append(obj)
        
        # Keep only recent objects
        if len(self.conversation_context['objects']) > self.max_context_history:
            self.conversation_context['objects'] = self.conversation_context['objects'][-self.max_context_history:]
        
        # Update last objects if any found
        if object_refs:
            self.conversation_context['last_objects'] = object_refs
        
        # Track actions performed
        for action in actions:
            action_summary = {
                'kind': action.kind,
                'camera': getattr(action, 'camera_id', None),
                'timestamp': time.time()
            }
            self.conversation_context['actions'].append(action_summary)
        
        # Keep only recent actions
        if len(self.conversation_context['actions']) > self.max_context_history:
            self.conversation_context['actions'] = self.conversation_context['actions'][-self.max_context_history:]
        
        logger.info(f"Conversation context updated - Last camera: {self.conversation_context['last_camera']}, Objects: {self.conversation_context['last_objects']}")
    
    def _get_conversation_context_summary(self) -> str:
        """Get a summary of the conversation context for the LLM."""
        summary_parts = []
        
        if self.conversation_context['last_camera']:
            summary_parts.append(f"Recently discussed camera: {self.conversation_context['last_camera']}")
        
        if self.conversation_context['cameras']:
            recent_cameras = ', '.join(self.conversation_context['cameras'][-3:])
            summary_parts.append(f"Recent cameras mentioned: {recent_cameras}")
        
        if self.conversation_context['last_objects']:
            objects = ', '.join(self.conversation_context['last_objects'])
            summary_parts.append(f"Recently mentioned objects: {objects}")
        
        if self.conversation_context['actions']:
            recent_actions = self.conversation_context['actions'][-3:]
            action_types = ', '.join([a['kind'] for a in recent_actions])
            summary_parts.append(f"Recent actions: {action_types}")
        
        return '\n'.join(summary_parts) if summary_parts else 'No recent conversation context'
    
    async def snapshot_detect(self, camera_id: str, camera_name: str, 
                             object_classes: List[str], model: str = 'auto',
                             confidence: float = 0.25) -> Dict[str, Any]:
        """
        Robust multi-detector snapshot detection with vision API fallback.
        
        Args:
            camera_id: Camera identifier
            camera_name: Human-readable camera name
            object_classes: List of object classes to detect (e.g., ['car', 'person'])
            model: Detection model ('auto', 'mobilenet', 'yolov8', or specific like 'yolov8n')
            confidence: Initial confidence threshold (default 0.25)
            
        Returns:
            Dictionary with detection results including bounding boxes and confidence
        """
        try:
            from .detector_manager import get_detector_manager
            import requests
            
            logger.info(f"🔍 Multi-detector snapshot detection: {camera_name} for {object_classes}")
            
            # Take snapshot via HTTP endpoint
            base_url = os.environ.get('PUBLIC_BASE_URL') or os.environ.get('API_BASE_URL') or 'http://localhost:5000'
            
            try:
                resp = await asyncio.to_thread(
                    requests.get,
                    f"{base_url}/api/cameras/{camera_id}/snapshot",
                    timeout=10
                )
                
                if not resp.ok:
                    return {
                        'success': False,
                        'error': f'Failed to capture snapshot: HTTP {resp.status_code}',
                        'camera': camera_name
                    }
                
                # Decode image from response
                image_bytes = resp.content
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return {
                        'success': False,
                        'error': 'Failed to decode snapshot image',
                        'camera': camera_name
                    }
                    
            except Exception as snap_error:
                return {
                    'success': False,
                    'error': f'Snapshot capture failed: {str(snap_error)}',
                    'camera': camera_name
                }
            
            # Get detector manager
            detector_mgr = get_detector_manager()
            
            # Check if requested classes are valid for onboard detectors
            coco_classes = detector_mgr._coco_classes if hasattr(detector_mgr, '_coco_classes') else []
            voc_classes = detector_mgr._voc_classes if hasattr(detector_mgr, '_voc_classes') else []
            
            # Normalize requested classes for checking
            normalized_requests = [obj.lower().strip() for obj in object_classes]
            
            # Check which classes are valid for onboard detection
            valid_for_coco = []
            valid_for_voc = []
            invalid_classes = []
            
            for obj in normalized_requests:
                # Check COCO (YOLOv8)
                if obj in coco_classes:
                    valid_for_coco.append(obj)
                # Check VOC (MobileNet)
                voc_variants = {
                    'airplane': 'aeroplane',
                    'motorcycle': 'motorbike', 
                    'dining table': 'diningtable',
                    'potted plant': 'pottedplant',
                    'tv': 'tvmonitor'
                }
                voc_obj = voc_variants.get(obj, obj)
                if voc_obj in voc_classes:
                    valid_for_voc.append(voc_obj)
                
                # If not in either, it's invalid for onboard detectors
                if obj not in coco_classes and voc_obj not in voc_classes:
                    invalid_classes.append(obj)
            
            # If ALL requested classes are invalid, skip onboard detection and go straight to Vision API
            if invalid_classes and not valid_for_coco and not valid_for_voc:
                logger.info(f"⚠️ Classes {invalid_classes} not supported by onboard detectors, skipping to Vision API")
                all_detections = []
                detection_attempts = []
            else:
                # Strategy 1: Try multiple detectors with various confidence levels
                all_detections = []
                detection_attempts = []
                
                # Define detection strategies (model, confidence)
                if model == 'auto':
                    strategies = [
                        ('mobilenet', 0.25),
                        ('mobilenet', 0.20),
                        ('yolo', 0.25),
                        ('yolo', 0.20),
                    ]
                else:
                    # User specified model - try with varying confidence
                    strategies = [
                        (model, confidence),
                        (model, max(0.15, confidence - 0.1)),
                        (model, max(0.10, confidence - 0.15)),
                    ]
                
                logger.info(f"Trying {len(strategies)} detection strategies...")
                
                for idx, (det_model, conf_threshold) in enumerate(strategies):
                    try:
                        # Set detector
                        # Configure (IMPORTANT: include `model` here, otherwise DetectorManager defaults to "default"
                        # and overrides any prior set_camera_detector() call.)
                        model_type = 'yolo' if 'yolo' in det_model.lower() else 'mobilenet'
                        detector_mgr.set_detection_config(f"{camera_id}_temp_{idx}", {
                            'enabled': True,
                            'confidence': conf_threshold,
                            'classes': object_classes,
                            'model': model_type,
                        })
                        
                        # Detect
                        result = detector_mgr.detect_and_track(
                            f"{camera_id}_temp_{idx}", 
                            frame, 
                            conf_threshold=conf_threshold, 
                            force_detection=True
                        )
                        
                        detections = result.get('tracks', [])
                        if detections:
                            all_detections.extend(detections)
                            detection_attempts.append({
                                'model': det_model,
                                'confidence': conf_threshold,
                                'found': len(detections)
                            })
                            logger.info(f"  ✓ {det_model} @ {conf_threshold}: {len(detections)} objects")
                        else:
                            logger.info(f"  ✗ {det_model} @ {conf_threshold}: 0 objects")
                            
                    except Exception as e:
                        logger.warning(f"  ⚠ {det_model} @ {conf_threshold} failed: {e}")
                        continue
            
            # Aggregate detections using NMS-like approach
            aggregated_detections = self._aggregate_detections(all_detections, iou_threshold=0.3)
            
            # If we found objects, return them
            if aggregated_detections:
                detection_summary = {}
                detailed_results = []
                
                for det in aggregated_detections:
                    class_name = det.get('class', 'unknown')
                    bbox = det.get('bbox', {})
                    conf = det.get('confidence', 0.0)
                    
                    if class_name not in detection_summary:
                        detection_summary[class_name] = 0
                    detection_summary[class_name] += 1
                    
                    detailed_results.append({
                        'class': class_name,
                        'confidence': round(conf, 3),
                        'bbox': {
                            'x': bbox.get('x', 0),
                            'y': bbox.get('y', 0),
                            'width': bbox.get('w', 0),
                            'height': bbox.get('h', 0)
                        }
                    })
                
                logger.info(f"✓ Multi-detector success: {detection_summary}")
                
                return {
                    'success': True,
                    'camera': camera_name,
                    'camera_id': camera_id,
                    'method': 'multi_detector',
                    'detections': detailed_results,
                    'summary': detection_summary,
                    'total_objects': len(aggregated_detections),
                    'classes_found': list(detection_summary.keys()),
                    'detection_attempts': detection_attempts
                }
            
            # Strategy 2: Try with stricter settings (lower confidence, more aggressive NMS)
            # Only try if we have valid classes for onboard detectors
            if not (invalid_classes and not valid_for_coco and not valid_for_voc):
                logger.info("⚙ No detections found, trying stricter settings...")
                
                strict_strategies = [
                    ('yolo', 0.15),
                    ('mobilenet', 0.15),
                    ('yolo', 0.10),
                ]
                
                for idx, (det_model, conf_threshold) in enumerate(strict_strategies):
                    try:
                        model_type = 'yolo' if 'yolo' in det_model.lower() else 'mobilenet'
                        detector_mgr.set_detection_config(f"{camera_id}_strict_{idx}", {
                            'enabled': True,
                            'confidence': conf_threshold,
                            'classes': object_classes,
                            'nms': 0.3,  # More aggressive NMS
                            'model': model_type,
                        })
                        
                        result = detector_mgr.detect_and_track(
                            f"{camera_id}_strict_{idx}", 
                            frame, 
                            conf_threshold=conf_threshold, 
                            force_detection=True
                        )
                        
                        detections = result.get('tracks', [])
                        if detections:
                            all_detections.extend(detections)
                            logger.info(f"  ✓ Strict {det_model} @ {conf_threshold}: {len(detections)} objects")
                            
                    except Exception as e:
                        logger.warning(f"  ⚠ Strict {det_model} failed: {e}")
                        continue
            
            # Re-aggregate with strict detections
            aggregated_detections = self._aggregate_detections(all_detections, iou_threshold=0.3)
            
            if aggregated_detections:
                detection_summary = {}
                detailed_results = []
                
                for det in aggregated_detections:
                    class_name = det.get('class', 'unknown')
                    bbox = det.get('bbox', {})
                    conf = det.get('confidence', 0.0)
                    
                    if class_name not in detection_summary:
                        detection_summary[class_name] = 0
                    detection_summary[class_name] += 1
                    
                    detailed_results.append({
                        'class': class_name,
                        'confidence': round(conf, 3),
                        'bbox': {
                            'x': bbox.get('x', 0),
                            'y': bbox.get('y', 0),
                            'width': bbox.get('w', 0),
                            'height': bbox.get('h', 0)
                        }
                    })
                
                logger.info(f"✓ Strict detection success: {detection_summary}")
                
                return {
                    'success': True,
                    'camera': camera_name,
                    'camera_id': camera_id,
                    'method': 'strict_detector',
                    'detections': detailed_results,
                    'summary': detection_summary,
                    'total_objects': len(aggregated_detections),
                    'classes_found': list(detection_summary.keys()),
                    'detection_attempts': detection_attempts
                }
            
            # Strategy 3: Fallback to LLM Vision API
            # Either because all detectors failed OR because classes aren't supported
            if invalid_classes and not valid_for_coco and not valid_for_voc:
                logger.info(f"🤖 Skipping onboard detectors - {invalid_classes} not in COCO/VOC classes. Using Vision API...")
            else:
                logger.info("🤖 All detectors failed, falling back to LLM Vision API...")
            
            if not self.provider:
                logger.warning("No LLM provider available for vision fallback")
                invalid_msg = f" (Note: {', '.join(invalid_classes)} not supported by onboard detectors)" if invalid_classes else ""
                return {
                    'success': False,
                    'camera': camera_name,
                    'error': f'No {", ".join(object_classes)} detected by any method{invalid_msg}',
                    'method': 'all_failed',
                    'detection_attempts': detection_attempts,
                    'invalid_classes': invalid_classes
                }
            
            # Encode image for vision API
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            image_data_url = f"data:image/jpeg;base64,{image_base64}"
            
            # Create vision prompt
            object_list = ', '.join(object_classes)
            
            # Special note if onboard detectors don't support these classes
            unsupported_note = ""
            if invalid_classes:
                unsupported_note = f"\n\nNOTE: {', '.join(invalid_classes)} are not in standard detection datasets, so look extra carefully for these objects."
            
            vision_prompt = f"""Analyze this camera image from {camera_name}.

Count and locate ALL {object_list} in the image.{unsupported_note}

For EACH object found, return:
1. Object type (class)
2. Your confidence (0.0-1.0)
3. Bounding box as PIXEL coordinates: {{"x": pixels_from_left, "y": pixels_from_top, "width": box_width_px, "height": box_height_px}}

Return ONLY a valid JSON array (NO extra text):
[
  {{"class": "car", "confidence": 0.95, "bbox": {{"x": 120, "y": 45, "width": 180, "height": 120}}}},
  {{"class": "wheelbarrow", "confidence": 0.87, "bbox": {{"x": 300, "y": 200, "width": 85, "height": 60}}}}
]

If no {object_list} are visible, return: []

Be thorough:
- Check entire image, including edges
- Look for partially visible objects
- Check shadowy areas
- Small/distant objects count too
- If unsure, estimate confidence lower (0.5-0.7)"""

            try:
                vision_result = await self.provider.vision(
                    image=image_data_url,
                    prompt=vision_prompt,
                    model=self.vision_model,
                    # Prefer local vision, but allow cloud fallback if local service is unavailable.
                    opts={"temperature": 0.1, "max_tokens": 500, "timeout": self.timeout, "source": "dual"}
                )
                
                vision_content = vision_result.get('content', '')
                logger.info(f"Vision API response: {vision_content[:200]}")
                
                # Parse vision response
                import re
                json_match = re.search(r'\[.*\]', vision_content, re.DOTALL)
                if json_match:
                    vision_detections = json.loads(json_match.group())
                    
                    if vision_detections and isinstance(vision_detections, list):
                        detection_summary = {}
                        for det in vision_detections:
                            class_name = det.get('class', 'unknown')
                            if class_name not in detection_summary:
                                detection_summary[class_name] = 0
                            detection_summary[class_name] += 1
                        
                        logger.info(f"✓ Vision API success: {detection_summary}")
                        
                        # Add note about why Vision API was used
                        note = 'Detected using LLM Vision API'
                        if invalid_classes:
                            note += f' ({", ".join(invalid_classes)} not in standard detection classes)'
                        
                        return {
                            'success': True,
                            'camera': camera_name,
                            'camera_id': camera_id,
                            'method': 'vision_api',
                            'detections': vision_detections,
                            'summary': detection_summary,
                            'total_objects': len(vision_detections),
                            'classes_found': list(detection_summary.keys()),
                            'note': note,
                            'invalid_classes': invalid_classes
                        }
                
            except Exception as vision_error:
                logger.error(f"Vision API fallback failed: {vision_error}")
            
            # Complete failure
            invalid_msg = ""
            if invalid_classes:
                invalid_msg = f" (Note: {', '.join(invalid_classes)} not supported by standard detectors)"
            
            return {
                'success': False,
                'camera': camera_name,
                'error': f'No {", ".join(object_classes)} detected by any method{invalid_msg}',
                'method': 'all_failed',
                'detection_attempts': detection_attempts,
                'invalid_classes': invalid_classes,
                'note': 'Tried multiple detectors and vision API'
            }
            
        except Exception as e:
            logger.error(f"Snapshot detection failed for {camera_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'camera': camera_name
            }
    
    def _aggregate_detections(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """
        Aggregate multiple detections using NMS-like approach to remove duplicates.
        Keeps highest confidence detection for overlapping boxes.
        """
        if not detections:
            return []
        
        # Group by class
        by_class = {}
        for det in detections:
            class_name = det.get('class', 'unknown')
            if class_name not in by_class:
                by_class[class_name] = []
            by_class[class_name].append(det)
        
        # Apply NMS per class
        final_detections = []
        
        for class_name, class_detections in by_class.items():
            # Sort by confidence
            sorted_dets = sorted(class_detections, key=lambda d: d.get('confidence', 0.0), reverse=True)
            
            keep = []
            while sorted_dets:
                # Take highest confidence detection
                best = sorted_dets.pop(0)
                keep.append(best)
                
                # Remove overlapping detections
                best_bbox = best.get('bbox', {})
                filtered = []
                
                for det in sorted_dets:
                    det_bbox = det.get('bbox', {})
                    iou = self._calculate_iou(best_bbox, det_bbox)
                    
                    if iou < iou_threshold:
                        filtered.append(det)
                
                sorted_dets = filtered
            
            final_detections.extend(keep)
        
        return final_detections
    
    def _calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        try:
            x1 = bbox1.get('x', 0)
            y1 = bbox1.get('y', 0)
            w1 = bbox1.get('w', 0) or bbox1.get('width', 0)
            h1 = bbox1.get('h', 0) or bbox1.get('height', 0)
            
            x2 = bbox2.get('x', 0)
            y2 = bbox2.get('y', 0)
            w2 = bbox2.get('w', 0) or bbox2.get('width', 0)
            h2 = bbox2.get('h', 0) or bbox2.get('height', 0)
            
            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate union
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception:
            return 0.0
    
    async def chat(self, message: str, context: AIContext, image: Optional[str] = None) -> AIResponse:
        """Send a chat message to the AI agent"""
        logger.info("AI chat method called - starting")

        # --- Fast-path commands (no LLM / no provider required) -----------------
        # Keep the system responsive for deterministic UI actions like opening a camera.
        try:
            open_cam = self._try_handle_open_camera_command(message, context)
            if open_cam is not None:
                return open_cam
        except Exception as _fast_err:
            logger.debug("Fast-path open camera handler failed: %s", _fast_err)
        
        req = self._resolve_request_overrides(context)
        provider_for_request = req.get("provider")
        provider_id_for_request = req.get("provider_id")
        fallback_candidates = req.get("fallback") or []
        chat_model_for_request = req.get("chat_model") or self.chat_model
        vision_model_for_request = req.get("vision_model") or self.vision_model

        if not provider_for_request:
            logger.info("No provider available")
            return AIResponse(
                message="AI agent is not configured. Please check your AI provider settings.",
                error="AI provider not available"
            )
        
        if not self._check_rate_limit():
            logger.info("Rate limit exceeded")
            return AIResponse(
                message="Rate limit exceeded. Please try again later.",
                error="Rate limit exceeded"
            )
        
        start_time = time.time()
        vision_used = image is not None
        image_size = len(image.encode()) if image else None
        
        # Debug logging
        logger.info(f"AI chat called with message: {message[:100]}...")
        logger.info(f"Vision enabled: {vision_used}")
        logger.info(f"Image size: {image_size} bytes" if image_size else "No image")
        if image:
            logger.info(f"Image starts with: {image[:100]}...")
        
        # Initialize messages variable to avoid scope issues in error handling
        messages = []
        vision_result = None
        vision_caption = ""
        vision_data = None
        
        try:
            # Lightweight intent handling for natural questions
            direct_resp = self._try_handle_natural_query(message, context)
            if direct_resp is not None:
                # Keep conversation context consistent even on fast-path returns
                try:
                    self._update_conversation_context(message, context, direct_resp.actions or [])
                except Exception:
                    pass
                return direct_resp

            logger.info("Starting chat processing")
            
            # Try to find learned pattern first (fast path for offline models)
            learned_pattern = None
            if self.intent_learner:
                learned_pattern = self.intent_learner.find_matching_pattern(message, threshold=0.75)
            
            if learned_pattern and not image:
                logger.info(f"✓ Found learned pattern: {learned_pattern.id} (used {learned_pattern.success_count} times)")
                
                # Execute directly from learned pattern
                action = AIAction(
                    tool=learned_pattern.tool,
                    parameters=learned_pattern.parameters.copy()
                )
                
                # Update pattern usage
                self.intent_learner.learn_from_execution(
                    user_query=message,
                    tool=learned_pattern.tool,
                    parameters=learned_pattern.parameters,
                    success=True,
                    feedback_score=1.0
                )
                
                return AIResponse(
                    message=f"Executing learned action: {learned_pattern.tool}",
                    actions=[action],
                    error=None
                )
            
            # Get environment context summary
            environment_context = self._get_environment_context_summary()
            
            # Get conversation context summary
            conversation_context_summary = self._get_conversation_context_summary()
            
            # Detect if we'll use local LLM (for context optimization)
            using_local_llm = False
            try:
                if isinstance(provider_for_request, HuggingFaceLocalProvider):
                    using_local_llm = True
                elif fallback_candidates and any(isinstance(p, HuggingFaceLocalProvider) for p in fallback_candidates):
                    # Might use fallback, prepare condensed context
                    using_local_llm = True
            except:
                pass
            
            # Create system prompt using a safe template (avoid f-string brace parsing)
            prompt_template = """You are an AI Security Assistant for Knoxnet VMS Beta, a surveillance and automation dashboard. You can see the dashboard layout and camera feeds to help operators monitor their security system.

ENVIRONMENT CONTEXT:
<<ENV_CTX>>

CONVERSATION CONTEXT (Use this to resolve ambiguous references):
<<CONV_CTX>>

CONVERSATIONAL UNDERSTANDING:
- **CRITICAL**: Use conversation context to resolve ambiguous references
- If user says "how many cars" without specifying camera, use the last mentioned camera
- If user asks "are there any" after mentioning objects, check for those objects
- Track the flow: "show primary entrance" → "how many cars" means "how many cars at primary entrance"
- Be smart about pronouns: "it", "there", "that camera" all refer to recently discussed cameras
- Remember recent object searches: if user asked about cars, follow-up questions likely relate to cars

INTENT CLASSIFICATION DECISION TREE:
┌─────────────────────────────────────────────────────────────────┐
│ User Question Type          │ Tool to Use                       │
├─────────────────────────────────────────────────────────────────┤
│ "how many X"                │ snapshot_detect                   │
│ "count X"                   │ snapshot_detect                   │
│ "are there any X"           │ snapshot_detect                   │
│ "find all X"                │ snapshot_detect                   │
│ "show me events" / "timeline"│ events_search                    │
├─────────────────────────────────────────────────────────────────┤
│ "what color is X"           │ take_camera_snapshot + analysisPrompt │
│ "describe X"                │ take_camera_snapshot + analysisPrompt │
│ "is X open/closed"          │ take_camera_snapshot + analysisPrompt │
│ "what does X look like"     │ take_camera_snapshot + analysisPrompt │
│ "what's happening"          │ take_camera_snapshot + analysisPrompt │
│ "tell me about X"           │ take_camera_snapshot + analysisPrompt │
└─────────────────────────────────────────────────────────────────┘

DETECTION TOOL SELECTION:
- **"how many X"** → Use EXECUTE_TOOL snapshot_detect (for COUNTING objects)
- **"count X"** → Use EXECUTE_TOOL snapshot_detect for instant counts
- **"are there any X"** → Use EXECUTE_TOOL snapshot_detect

DESCRIPTIVE/ANALYSIS QUESTIONS (DO NOT USE snapshot_detect):
- **"what color is X"** → Use EXECUTE_TOOL take_camera_snapshot with analysisPrompt about color
- **"is the door open/closed"** → Use take_camera_snapshot with analysisPrompt about door state
- **"what's happening"** → Use take_camera_snapshot with analysisPrompt about activity
- **"describe X"** → Use take_camera_snapshot with analysisPrompt about description
- **Color, state, activity, description questions** → Use take_camera_snapshot (NOT snapshot_detect!)
- snapshot_detect is ONLY for counting/detecting objects - NOT for describing them

CRITICAL INTENT UNDERSTANDING:
- User asks "what color is the house" → They want COLOR not detection! Use take_camera_snapshot with analysisPrompt="What color is the house?"
- User asks "how many cars" → They want COUNT! Use snapshot_detect with objectClasses=["car"]
- User asks "is door closed" → They want STATE! Use take_camera_snapshot with analysisPrompt="Is the door open or closed?"
- User asks "what color is the tractor" → They want COLOR! Use take_camera_snapshot with analysisPrompt="What color is the tractor?"
- **UNDERSTAND THE QUESTION TYPE** - counting vs describing are different intents!

CONVERSATION FLOW EXAMPLES:
Example 1 - Descriptive Questions:
  User: "look at the workshop"
  AI: create_camera_widget("Workshop")
  
  User: "what color is the house"
  AI: take_camera_snapshot(cameraRef="Workshop", analysisPrompt="What color is the house?")
  Result: "The house appears to be white with blue trim"
  
  User: "what color is the tractor"
  AI: take_camera_snapshot(cameraRef="Workshop", analysisPrompt="What color is the tractor?")
  Result: "The tractor is green"

Example 2 - Counting Questions:
  User: "show operations hub"
  AI: create_camera_widget("Operations Hub")
  
  User: "how many cars"
  AI: snapshot_detect(cameraRef="Operations Hub", objectClasses=["car"])
  Result: "2 cars detected"
  
  User: "any people"
  AI: snapshot_detect(cameraRef="Operations Hub", objectClasses=["person"])
  Result: "1 person detected"

Example 3 - Mixed Questions:
  User: "check the workshop"
  AI: create_camera_widget("Workshop")
  
  User: "is there a wheelbarrow and what color is it"
  AI: [Single action - can answer both in one snapshot]
    take_camera_snapshot(cameraRef="Workshop", analysisPrompt="Is there a wheelbarrow? If yes, what color is it?")
  Result: "Yes, there is 1 wheelbarrow. It's orange/red colored."

Example 4 - Real User Questions (from conversation):
  User: "look at the workshop what color is the house and is the door closed"
  AI: take_camera_snapshot(cameraRef="Workshop", analysisPrompt="What color is the house? Is the door open or closed?")
  Result: "The house is white with blue trim. The door appears to be closed."
  
  User: "what color is the tractor at the workshop"
  AI: take_camera_snapshot(cameraRef="Workshop", analysisPrompt="What color is the tractor?")
  Result: "The tractor is green with yellow wheels."
  
  User: "how many cars at operations hub"
  AI: snapshot_detect(cameraRef="Operations Hub", objectClasses=["car"])
  Result: "4 cars detected"

IMPORTANT VISION GUIDELINES:
- When analyzing images, describe what you ACTUALLY see, not what you expect to see
- Reference cameras strictly by their DISPLAY NAMES as shown in the system (never fabricate names), not widget IDs
- If you see a dashboard screenshot, describe the actual widgets, their positions, and any visible camera feeds
- If you see a camera feed, describe the scene, objects, people, vehicles, and any activity
- Be specific about what's visible vs. what might be hidden or unclear
- If you can't see something clearly, say so rather than making assumptions
- Use your environment context to understand patterns and changes over time
- Reference previous observations when relevant to provide continuity

COMPOSITE IMAGE ANALYSIS:
- When analyzing composite images (multiple camera feeds stitched together), focus on the most relevant or concerning activity
- Identify which camera each section belongs to using the labels
- Provide a concise summary of the overall security situation across all cameras
- Highlight any unusual activity, people, vehicles, or security concerns
- If multiple cameras show the same area from different angles, note this for better coverage

ZONE AND LINE DETECTION:
- **CRITICAL**: Pay attention to detection zones (polygonal areas), laser lines, and tags drawn on camera feeds
- These visual elements are overlaid on the actual camera feeds and represent active security monitoring areas
- **Zones**: Colored polygonal areas for motion detection - mention if people/objects are inside these areas
- **Laser Lines**: Animated gradient lines for line crossing detection - mention if anyone is crossing or near these lines
- **Tags**: Cross-shaped markers for specific points of interest - describe what's happening at these locations
- When describing activity, ALWAYS mention if objects/people are within zones or crossing laser lines
- These detection areas are the primary security monitoring tools - they indicate what the system is actively watching
- Reference these detection areas when describing security-relevant activity and explain their purpose

RESPONSE STYLE:
- **BE EXTREMELY CONCISE** - Keep responses under 50 words unless critical details require more
- Focus ONLY on security-relevant details: people, vehicles, unusual activity, potential threats
- Use bullet points or very short sentences
- Prioritize information by security importance
- If everything looks normal, say "All clear" or "No activity"
- **NEVER write long paragraphs or detailed descriptions**
- **CRITICAL**: If you see zones, lines, or tags, mention them first

USER INTENT PHILOSOPHY:
- **FOCUS ON INTENT, NOT EXACT NAMES** - Understand what the user wants to achieve
- **BE ACTION-ORIENTED** - Execute actions, don't just acknowledge requests
- **USE CONTEXT** - If only one camera visible, that's the camera they mean
- **BE FLEXIBLE** - "the camera", "that cam", "it" all mean the visible camera when only one is shown
- **INFER SMARTLY** - User says "zoom in" → they mean the visible camera
- **DON'T BE PEDANTIC** - User intent > exact camera names
- **JUST DO IT** - Execute what the user wants, don't ask for clarification unless truly ambiguous

NATURAL LANGUAGE UNDERSTANDING:
You should understand vague and casual requests like:
- "show all cams" / "show me all the cameras" / "display all cameras" → CRITICAL: Check dashboard_state to count how many cameras are currently displayed vs total available cameras. ONLY say "already displayed" if ALL cameras are visible (e.g., 8 out of 8). If ANY cameras missing (e.g., only 2 of 8 shown), create camera grid with clearExisting=true to show all cameras properly.
- "show all cameras" / "all cams" → Same as above - verify ALL cameras are shown, not just some
- "reorganize cameras" / "organize camera layout" / "fix camera layout" → Use EXECUTE_TOOL reorganize_camera_widgets to PRESERVE existing widgets but reorganize them into optimal grid. DO NOT remove widgets!
- "clean up layout" / "tidy up cameras" → Use reorganize_camera_widgets to preserve and reorganize
- "create 2x2 grid" → create 2x2 camera grid layout (use EXECUTE_TOOL create_camera_grid_layout with layout="2x2")
- "show 3x3 layout" → create 3x3 camera grid (use EXECUTE_TOOL create_camera_grid_layout with layout="3x3")
- "create 4x4 grid" → create 4x4 camera grid (use EXECUTE_TOOL create_camera_grid_layout with layout="4x4")
- "1+5 layout" → create 1 large + 5 small cameras layout (use EXECUTE_TOOL create_1plus5_layout)
- "1+7 layout" → create 1 large + 7 small cameras layout (use EXECUTE_TOOL create_1plus7_layout)
- "split view" → create 2 cameras side by side (use EXECUTE_TOOL create_split_view_layout)
- "clear cameras" → remove all camera widgets (use EXECUTE_TOOL clear_all_camera_widgets)
- "create terminal" → create terminal widget (use EXECUTE_TOOL open_terminal_widget)
- "open weather widget" → create weather widget (use EXECUTE_TOOL open_weather_widget)
- "organize widgets" → auto-arrange widgets (use EXECUTE_TOOL organize_widgets)
- "list all widgets" → show all widgets (use EXECUTE_TOOL list_widgets)
- "open north entrance cam" → create widget for north entrance camera

SPECIALIZED WIDGET COMMANDS (Context-Aware):
- "show me point cloud in bone" / "show point cloud bone" → Open depth map for visible camera with bone color scheme (use EXECUTE_TOOL open_depth_map_widget with colorScheme="bone", mode="pointcloud")
- "open depth map for SCOUT in jet" → Open depth map for SCOUT with jet colors (use EXECUTE_TOOL open_depth_map_widget with cameraRef="SCOUT", colorScheme="jet")
- "show depth in viridis" → Open depth map with viridis color scheme (use EXECUTE_TOOL open_depth_map_widget with colorScheme="viridis")
- "point cloud turbo" / "turbo point cloud" → Open point cloud with turbo color scheme (use EXECUTE_TOOL open_depth_map_widget with colorScheme="turbo")
- "SLAM for scout" → Open SLAM mode depth map (use EXECUTE_TOOL open_depth_map_widget with mode="slam")
- "open audio for primary entrance" / "audio primary entrance" → Open audio widget for Primary Entrance (use EXECUTE_TOOL open_audio_widget with cameraRef="Primary Entrance")
- "listen to north entrance" → Open audio in listen mode (use EXECUTE_TOOL open_audio_widget with mode="listen")
- "two way audio workshop" → Open audio in two-way mode for Workshop (use EXECUTE_TOOL open_audio_widget with mode="twoWay")
- "talk to camera" → Open audio in talk mode for visible camera (use EXECUTE_TOOL open_audio_widget with mode="talk")
- "record audio north entrance" → Open audio in record mode (use EXECUTE_TOOL open_audio_widget with mode="record")
- "open ptz for primary entrance" / "ptz primary entrance" → Open PTZ control (use EXECUTE_TOOL open_ptz_widget with cameraRef="Primary Entrance")
- "ptz control" → Open PTZ for visible camera (use EXECUTE_TOOL open_ptz_widget)
- "open AI assistant" / "ai assistant" / "assistant" → Open AI assistant widget (use EXECUTE_TOOL open_ai_assistant_widget)
- "open terminal" / "terminal" / "command line" → Open terminal widget (use EXECUTE_TOOL open_terminal_widget)
- "show weather" / "weather widget" → Open weather widget (use EXECUTE_TOOL open_weather_widget)

CAMERA CONTROL COMMANDS (Full Widget Control):
- "use webrtc for all cameras" / "switch all to webrtc" → Set all camera widgets to WebRTC (use EXECUTE_TOOL set_camera_transport_mode with applyToAll=true, transportMode="webrtc")
- "use hls for North Entrance" / "switch North Entrance to hls" → Set specific camera to HLS (use EXECUTE_TOOL set_camera_transport_mode with transportMode="hls")
- "add zone to North Entrance camera" → Add detection zone to camera widget (use EXECUTE_TOOL add_shapes_to_camera with zones array)
- "create North Entrance camera with entry zone" → Create camera widget with pre-configured zone (use EXECUTE_TOOL create_camera_with_shapes)
- "start recording North Entrance" / "record Primary Entrance" → Start recording camera (use EXECUTE_TOOL toggle_camera_recording with record=true)
- "stop recording North Entrance" → Stop recording (use EXECUTE_TOOL toggle_camera_recording with record=false)
- "take snapshot of North Entrance" / "capture North Entrance" → Take snapshot (use EXECUTE_TOOL take_camera_snapshot)
- "set overlay transparency to 80%" / "overlay alpha 0.8" → Set overlay transparency (use EXECUTE_TOOL set_camera_display_settings with overlayAlpha=0.8)
- "lock aspect ratio" / "unlock aspect" → Control aspect ratio (use EXECUTE_TOOL set_camera_display_settings)

OBJECT DETECTION COMMANDS (Full Detection Control):
- "turn on object detection on scout and look for benches" / "detect benches on scout" → Enable detection with specific classes (use EXECUTE_TOOL enable_object_detection with cameraRef="SCOUT", classes=["bench"])
- "look for laptops on the scout" / "detect laptops on scout" → Enable detection for laptops (use EXECUTE_TOOL enable_object_detection with classes=["laptop"])
- "detect people and cars on Primary Entrance" → Enable for multiple classes (use EXECUTE_TOOL enable_object_detection with classes=["person", "car"])
- "turn off object detection" / "disable detection" → Disable detection (use EXECUTE_TOOL disable_object_detection)
- "use yolov8 for detection" / "switch to yolov8" → Change detection model (use EXECUTE_TOOL set_detection_model with model="yolov8")
- "use mobilenet" → Switch to MobileNet (use EXECUTE_TOOL set_detection_model with model="mobilenet")
- "set detection confidence to 70%" → Adjust confidence (use EXECUTE_TOOL set_detection_confidence with confidence=0.7)
- "look for dogs and cats" → Set detection classes (use EXECUTE_TOOL set_detection_classes with classes=["dog", "cat"])
- "detect everything" / "detect all objects" → Enable with all common classes (use EXECUTE_TOOL enable_object_detection with classes=["person","car","dog","cat","bicycle","motorcycle"])
- "only detect in zones" → Enable zone-only detection (use EXECUTE_TOOL enable_object_detection with detectInZonesOnly=true)
- "what can you detect" / "list detection classes" → Show available classes (use EXECUTE_TOOL get_detection_classes)

- "what's happening" → take a camera snapshot of the relevant feed and analyze it
- "show me the front" → find and display front-related cameras
- "show the workshop" → create widget for workshop camera
- "what changed" → compare current view with previous observations
- "is it still there" → reference previous observations about specific objects/people
- "count cars" → set up car counting for visible cameras
- "count cars in zone 1" → set up car counting for the camera with zone 1
- "start counting vehicles" → set up car counting for visible cameras
- "place all camera overlays" → place overlay widgets for every camera (use EXECUTE_TOOL place_all_camera_overlays)
- "place a camera overlay for Workshop cam purple overlay" → create overlay for camera "Workshop" and select purple shapes (use EXECUTE_TOOL create_camera_overlay, then select shapes by color)
- "show me overlay for Primary Entrance, zone 3" → create overlay for "Primary Entrance" and select the zone labeled/ID for zone 3 (use EXECUTE_TOOL create_camera_overlay with zoneIds)

AVAILABLE TOOLS AND ACTIONS:
You can execute actions by including them in your JSON response. Available actions:

1. CREATE_CAMERA_WIDGET - Create a camera widget on the dashboard
   {
     "kind": "create_camera_widget",
     "props": {
       "camera_name": "exact or partial camera name",
       "camera_id": "optional camera ID if known"
     }
   }

2. CREATE_ALL_CAMERAS_GRID - Create a grid layout with all cameras
   {
     "kind": "create_all_cameras_grid"
   }

3. CREATE_WIDGET - Create any type of widget
   {
     "kind": "create_widget",
     "widget_type": "camera|ai|ptz|terminal|weather|etc",
     "position": {"x": 100, "y": 100},
     "size": {"width": 400, "height": 300},
     "props": {"title": "Widget Title", "camera_id": "optional"}
   }

4. MOVE_WIDGET - Move an existing widget
   {
     "kind": "move_widget", 
     "widget_id": "widget-id",
     "position": {"x": 200, "y": 150}
   }

5. CAMERA_SNAPSHOT - Take a snapshot of a camera
   {
     "kind": "camera_snapshot",
     "props": {"camera_id": "camera-id"}
   }

6. REORGANIZE_DASHBOARD - Reorganize dashboard layout to avoid overlaps
   {
     "kind": "reorganize_dashboard"
   }

7. DETECT_OBJECTS - Perform object detection on an image using YOLOv8 or MobileNet
   {
     "kind": "detect_objects",
     "detection_model": "yolov8|mobilenet",
     "props": {
       "image": "base64_image_data",
       "confidence_threshold": 0.5
     }
   }

8. SETUP_MONITORING - Setup monitoring for a specific zone
   {
     "kind": "setup_monitoring",
     "zone_id": "zone-id",
     "camera_id": "camera-id",
     "monitoring_enabled": true
   }

9. INTERPRET_ZONE_RULES - Interpret natural language rules for a zone
   {
     "kind": "interpret_zone_rules",
     "zone_id": "zone-id",
     "natural_language_rule": "if any person goes into the zone it should open a certain camera widget and log the output"
   }

10. RECALL_CONTEXT - Recall relevant context from memory
    {
      "kind": "recall_context",
      "context_query": "what happened with the primary entrance camera yesterday"
    }

11. ADD_NATURAL_LANGUAGE_RULE - Add a natural language rule for a zone
    {
      "kind": "add_natural_language_rule",
      "zone_id": "zone-id",
      "natural_language_rule": "if any person goes into the zone it should open a certain camera widget and log the output",
      "camera_id": "camera-id"
    }

12. EXECUTE_TOOL - Count roadway vehicles for a camera (ONLY when the user explicitly asks to "count cars" or "vehicle counting". Do NOT use for "watch for cars" or alert-style criteria.)
    {
      "kind": "execute_tool",
      "tool_id": "count_roadway_vehicles",
      "parameters": {
        "cameraRef": "camera name or ID",
        "durationSeconds": 10,
        "sampleFps": 5,
        "minConfidence": 0.4,
        "useHighPrecision": true
      }
    }

13. EXECUTE_TOOL - Execute a frontend AI tool by id with parameters (preferred for motion watch)
    {
      "kind": "execute_tool",
      "tool_id": "motion_detector_watch",
      "parameters": {
        "cameraRef": "Primary Entrance",
        "durationMinutes": 15,
        "zonesOnly": true
      }
    }

14. EXECUTE_TOOL - Watch Screenshot Analysis (criteria-based analysis during an active motion watch)
    {
      "kind": "execute_tool",
      "tool_id": "watch_screenshot_analysis",
      "parameters": {
        "criteria": "e.g., red cars | white cars | package at door",
        "cameraRef": "Primary Entrance (optional)",
        "durationMinutes": 10,
        "notifyOnMatchOnly": true
      }
    }

14b. EXECUTE_TOOL - Search Capture Events Timeline (indexed Motion Watch / captures)
    {
      "kind": "execute_tool",
      "tool_id": "events_search",
      "parameters": {
        "query": "e.g., white truck near produce stand",
        "cameraRef": "camera name (optional)",
        "start": "ISO timestamp or unix seconds (optional)",
        "end": "ISO timestamp or unix seconds (optional)",
        "limit": 25
      }
    }

15. EXECUTE_TOOL - Place all camera overlays on the dashboard
    {
      "kind": "execute_tool",
      "tool_id": "place_all_camera_overlays",
      "parameters": {
        "size": { "width": 320, "height": 180 },
        "rotationDeg": 0,
        "scale": 1,
        "showLabels": true
      }
    }

16. EXECUTE_TOOL - Create a camera overlay for a specific camera
    {
      "kind": "execute_tool",
      "tool_id": "create_camera_overlay",
      "parameters": {
        "cameraRef": "Workshop",
        "zoneIds": [],
        "lineIds": [],
        "tagIds": [],
        "rotationDeg": 0,
        "scale": 1
      }
    }

17. EXECUTE_TOOL - Rotate an existing overlay widget
    {
      "kind": "execute_tool",
      "tool_id": "rotate_overlay",
      "parameters": {
        "widgetId": "overlay-widget-id",
        "direction": "right",
        "degrees": 15
      }
    }

18. EXECUTE_TOOL - Expand (resize) an existing overlay widget
    {
      "kind": "execute_tool",
      "tool_id": "expand_overlay",
      "parameters": {
        "widgetId": "overlay-widget-id",
        "scale": 1.2
      }
    }

19. EXECUTE_TOOL - Select overlay shapes (zones/lines/tags) for a widget
    {
      "kind": "execute_tool",
      "tool_id": "select_overlay_shapes",
      "parameters": {
        "widgetId": "overlay-widget-id",
        "zoneIds": ["zone-3-id"],
        "lineIds": [],
        "tagIds": []
      }
    }

20. EXECUTE_TOOL - Create VMS Camera Grid Layout (2x2, 3x3, 4x4, etc.)
    {
      "kind": "execute_tool",
      "tool_id": "create_camera_grid_layout",
      "parameters": {
        "layout": "2x2",
        "cameraIds": ["camera-1", "camera-2", "camera-3", "camera-4"],
        "clearExisting": false
      }
    }

21. EXECUTE_TOOL - Create 1+5 VMS Layout (1 large camera + 5 smaller)
    {
      "kind": "execute_tool",
      "tool_id": "create_1plus5_layout",
      "parameters": {
        "cameraIds": ["main-camera", "cam2", "cam3", "cam4", "cam5", "cam6"],
        "clearExisting": false
      }
    }

22. EXECUTE_TOOL - Create 1+7 VMS Layout (1 large camera + 7 smaller)
    {
      "kind": "execute_tool",
      "tool_id": "create_1plus7_layout",
      "parameters": {
        "cameraIds": ["main-camera", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7", "cam8"],
        "clearExisting": false
      }
    }

23. EXECUTE_TOOL - Create Split View Layout (2 cameras side by side or stacked)
    {
      "kind": "execute_tool",
      "tool_id": "create_split_view_layout",
      "parameters": {
        "cameraIds": ["camera-1", "camera-2"],
        "orientation": "horizontal",
        "clearExisting": false
      }
    }

24. EXECUTE_TOOL - Clear All Camera Widgets
    {
      "kind": "execute_tool",
      "tool_id": "clear_all_camera_widgets",
      "parameters": {}
    }

24b. EXECUTE_TOOL - Reorganize Existing Camera Widgets (PRESERVES widgets)
    {
      "kind": "execute_tool",
      "tool_id": "reorganize_camera_widgets",
      "parameters": {
        "layout": "3x3"
      }
    }

25. EXECUTE_TOOL - Create Any Widget Type
    {
      "kind": "execute_tool",
      "tool_id": "create_widget",
      "parameters": {
        "widgetType": "terminal",
        "title": "Command Terminal",
        "position": {"x": 100, "y": 100},
        "size": {"width": 600, "height": 400}
      }
    }

26. EXECUTE_TOOL - Update Widget Properties
    {
      "kind": "execute_tool",
      "tool_id": "update_widget",
      "parameters": {
        "widgetId": "widget-id",
        "position": {"x": 200, "y": 150},
        "title": "New Title"
      }
    }

27. EXECUTE_TOOL - List All Widgets
    {
      "kind": "execute_tool",
      "tool_id": "list_widgets",
      "parameters": {
        "type": "camera"
      }
    }

28. EXECUTE_TOOL - Organize Widgets (Auto-arrange)
    {
      "kind": "execute_tool",
      "tool_id": "organize_widgets",
      "parameters": {
        "layout": "grid"
      }
    }

29. EXECUTE_TOOL - Clear All Widgets
    {
      "kind": "execute_tool",
      "tool_id": "clear_all_widgets",
      "parameters": {}
    }

30. EXECUTE_TOOL - Open Depth Map Widget (with color scheme and mode)
    {
      "kind": "execute_tool",
      "tool_id": "open_depth_map_widget",
      "parameters": {
        "cameraRef": "SCOUT",
        "colorScheme": "bone",
        "mode": "pointcloud"
      }
    }

31. EXECUTE_TOOL - Open Audio Widget
    {
      "kind": "execute_tool",
      "tool_id": "open_audio_widget",
      "parameters": {
        "cameraRef": "Primary Entrance",
        "mode": "listen",
        "autoStart": true
      }
    }

32. EXECUTE_TOOL - Open PTZ Control Widget
    {
      "kind": "execute_tool",
      "tool_id": "open_ptz_widget",
      "parameters": {
        "cameraRef": "Primary Entrance",
        "showPresets": true
      }
    }

33. EXECUTE_TOOL - Open AI Assistant Widget
    {
      "kind": "execute_tool",
      "tool_id": "open_ai_assistant_widget",
      "parameters": {
        "mode": "chat"
      }
    }

34. EXECUTE_TOOL - Open Terminal Widget
    {
      "kind": "execute_tool",
      "tool_id": "open_terminal_widget",
      "parameters": {}
    }

35. EXECUTE_TOOL - Open Weather Widget
    {
      "kind": "execute_tool",
      "tool_id": "open_weather_widget",
      "parameters": {
        "location": "auto"
      }
    }

36. EXECUTE_TOOL - Set Camera Transport Mode (WebRTC/HLS/RTSP)
    {
      "kind": "execute_tool",
      "tool_id": "set_camera_transport_mode",
      "parameters": {
        "cameraRef": "Primary Entrance",
        "transportMode": "webrtc",
        "applyToAll": false
      }
    }

37. EXECUTE_TOOL - Add Zones/Lines/Tags to Camera Widget
    {
      "kind": "execute_tool",
      "tool_id": "add_shapes_to_camera",
      "parameters": {
        "cameraRef": "North Entrance",
        "zones": [{"name": "Entry Zone", "points": [[0.2,0.3],[0.8,0.3],[0.8,0.7],[0.2,0.7]], "color": "#00ff00"}],
        "lines": [{"name": "Line 1", "p1": {"x": 0.3, "y": 0.5}, "p2": {"x": 0.7, "y": 0.5}}]
      }
    }

38. EXECUTE_TOOL - Create Camera Widget with Pre-Configured Shapes
    {
      "kind": "execute_tool",
      "tool_id": "create_camera_with_shapes",
      "parameters": {
        "cameraRef": "North Entrance",
        "zones": [{"name": "Zone 1", "points": [[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]], "color": "#00ff00"}],
        "transportMode": "hls"
      }
    }

39. EXECUTE_TOOL - Toggle Camera Recording
    {
      "kind": "execute_tool",
      "tool_id": "toggle_camera_recording",
      "parameters": {
        "cameraRef": "Primary Entrance",
        "record": true
      }
    }

40. EXECUTE_TOOL - Take Camera Snapshot (For Visual Analysis)
    {
      "kind": "execute_tool",
      "tool_id": "take_camera_snapshot",
      "parameters": {
        "cameraRef": "North Entrance",
        "includeOverlays": true,
        "analysisPrompt": "What color is the house? Is the door open or closed?"
      }
    }
    - **USE analysisPrompt for descriptive questions!**
    - Examples:
      * "what color is the house" → analysisPrompt: "What color is the house?"
      * "what color is the tractor" → analysisPrompt: "What color is the tractor?"
      * "is the door open" → analysisPrompt: "Is the door open or closed?"
      * "describe the scene" → analysisPrompt: "Describe what you see in detail"
    - The Vision API will analyze the image and answer the question directly
    - DO NOT use snapshot_detect for color/description questions!

41. EXECUTE_TOOL - Set Camera Display Settings
    {
      "kind": "execute_tool",
      "tool_id": "set_camera_display_settings",
      "parameters": {
        "cameraRef": "North Entrance",
        "overlayAlpha": 0.8,
        "aspectRatio": "16:9",
        "aspectLocked": true
      }
    }

42. EXECUTE_TOOL - Enable Object Detection with Classes
    {
      "kind": "execute_tool",
      "tool_id": "enable_object_detection",
      "parameters": {
        "cameraRef": "SCOUT",
        "classes": ["laptop", "bench", "person"],
        "model": "yolov8",
        "confidence": 0.5
      }
    }

43. EXECUTE_TOOL - Disable Object Detection
    {
      "kind": "execute_tool",
      "tool_id": "disable_object_detection",
      "parameters": {
        "cameraRef": "SCOUT"
      }
    }

44. EXECUTE_TOOL - Set Detection Classes Only
    {
      "kind": "execute_tool",
      "tool_id": "set_detection_classes",
      "parameters": {
        "cameraRef": "Primary Entrance",
        "classes": ["person", "car", "truck"]
      }
    }

45. EXECUTE_TOOL - Set Detection Confidence Threshold
    {
      "kind": "execute_tool",
      "tool_id": "set_detection_confidence",
      "parameters": {
        "cameraRef": "North Entrance",
        "confidence": 0.7
      }
    }

46. EXECUTE_TOOL - Set Detection Model
    {
      "kind": "execute_tool",
      "tool_id": "set_detection_model",
      "parameters": {
        "cameraRef": "SCOUT",
        "model": "yolov8"
      }
    }

47. EXECUTE_TOOL - Get Available Detection Classes
    {
      "kind": "execute_tool",
      "tool_id": "get_detection_classes",
      "parameters": {
        "model": "yolov8"
      }
    }

48. EXECUTE_TOOL - Snapshot Detection (Robust Multi-Detector)
    {
      "kind": "execute_tool",
      "tool_id": "snapshot_detect",
      "parameters": {
        "cameraRef": "Primary Entrance",
        "objectClasses": ["car", "person"],
        "model": "auto",
        "confidence": 0.25
      }
    }
    - **ROBUST DETECTION**: Automatically tries multiple detectors with various confidence levels
    - **Strategy 1**: MobileNet + YOLO at 0.25, 0.20 confidence (4 attempts)
    - **Strategy 2**: If no detection, tries stricter settings at 0.15, 0.10 confidence
    - **Strategy 3**: If all fail, falls back to LLM Vision API for human-level detection
    - **Aggregation**: Uses NMS to merge results and remove duplicates
    - **Output**: Returns bounding boxes, confidence scores, and detection method used
    - USE THIS for "how many X" questions - it won't miss objects!
    - Model defaults to 'auto' (tries multiple) or specify 'mobilenet'/'yolov8'
    - Automatically uses last mentioned camera if not specified

NATURAL LANGUAGE RULE EXAMPLES:
- "if any person goes into the zone it should open a certain camera widget and log the output"
- "when motion is detected in this zone, start recording and send an alert"
- "if a car enters this area, notify security and open the north entrance camera"
- "when someone crosses this line, log the event and capture a snapshot"

OBJECT DETECTION CAPABILITIES:
- YOLOv8: Fast and accurate object detection for people, vehicles, and common objects
- MobileNet: Lightweight detection suitable for real-time processing
- Both models can detect: people, cars, trucks, motorcycles, bicycles, and many other objects
- Detection results include: object type, confidence score, and bounding box coordinates

CONTEXT RECALL FEATURES:
- Maintain memory of previous interactions and observations
- Recall relevant information based on natural language queries
- Track zone monitoring history and rule triggers
- Provide continuity across multiple sessions

RESPONSE FORMAT:
Always respond in this EXACT JSON format (NO markdown code blocks, NO ```json wrapper):
{
  "message": "Your response to the user",
  "actions": [
    {
      "kind": "action_type",
      "props": {"...": "..."}
    }
  ]
}

CRITICAL: Return ONLY the raw JSON object. Do NOT wrap it in ```json code blocks or any markdown formatting.

When responding:
1. Understand the user's intent, even from vague language
2. **CRITICAL - Check Dashboard State First & Auto-Create Widgets**: ALWAYS check the provided dashboard_state or layout
   - If user says "show all cams", count BOTH: (a) how many cameras are currently displayed, (b) how many cameras exist in cameras.json
   - ONLY respond "All cameras are already displayed" if the counts MATCH (e.g., 8 of 8 displayed)
   - If counts DON'T match (e.g., only 2 of 8 displayed), CREATE camera grid with clearExisting=true to show ALL cameras
   - **CRITICAL - Auto-Create Missing Widgets**: If user references a specific camera that doesn't have a widget, CREATE IT FIRST
     Example: "look for cars at the primary entrance" but no Primary Entrance widget → create_camera_widget("Primary Entrance") THEN enable_object_detection + motion_detector_watch
   - If asking "what's happening" on an existing camera, use CAMERA_SNAPSHOT instead of creating duplicate widget
3. **Smart Layout Selection**: When user says "show all cams" without specifying layout:
   - For 1-4 cameras: use 2x2 grid
   - For 5-9 cameras: use 3x3 grid
   - For 10-16 cameras: use 4x4 grid
   - For 17+ cameras: use appropriate NxM grid
   - Always maintain 16:9 aspect ratio for camera feeds
4. **Avoid Duplicates**: Before creating any camera widget, verify it doesn't already exist in dashboard_state
5. **Overlay vs Camera Widgets**: If they ask to place an overlay or "camera overlay", use EXECUTE_TOOL with overlay tools (place_all_camera_overlays or create_camera_overlay) instead of creating camera widgets
6. **CRITICAL - Context-Aware Camera Resolution**: 
   - If user says "zoom in", "pan left", "what's happening" without specifying camera name, check dashboard_state
   - If only ONE camera widget visible → That's the camera they mean! Use that camera ID/name
   - If user says "the camera", "that cam", "it" → Use context to resolve
   - Don't ask for clarification when context is obvious (e.g., one camera visible)
7. **Intent > Precision**: Understand what user wants to DO, not just what they SAY
   - "zoom in" → Execute PTZ zoom on visible camera
   - "what's happening" → Take snapshot of visible camera
   - "start recording" → Start recording on visible camera
   - Be ACTION-ORIENTED: execute tools, don't just acknowledge
8. **CRITICAL - Widget Preservation Philosophy**:
   - ALWAYS prefer REORGANIZING over REMOVING widgets
   - If user says "organize cameras", "fix layout", "tidy up" → Use reorganize_camera_widgets (PRESERVES widgets)
   - ONLY use clearExisting=true if user explicitly says "clear", "remove", "start fresh", or "replace all"
   - Preserve non-camera widgets (terminals, AI assistants, etc.) when clearing cameras
   - User's widgets are valuable - don't destroy their work!
9. Reference cameras by their display names from cameras.json, not technical IDs
10. Provide actionable insights based on what you observe
11. Keep responses concise and security-focused
12. **Graceful Loading Policy**: 
    - Tools automatically stagger widget creation (500ms between widgets)
    - Each camera gets 1200ms to establish stream connection before creating next
    - Uses HLS by default for bulk loading (more reliable than WebRTC for many simultaneous streams)
    - Total time for 8 cameras: ~13 seconds (worth it for reliable connections!)
    - Automatically verifies streams are loading and retries if needed
    - This prevents black screens, timeouts, and connection failures
    - Don't rush - patience ensures all cameras load properly
    - Tell user: "Loading cameras gracefully to ensure all connect properly. Using HLS streaming for reliability..."

CAR COUNTING REQUESTS:
- When users ask to "count cars", "count vehicles", "start counting", or similar
- When they mention specific zones like "count cars in zone 1"
- ALWAYS execute the `count_roadway_vehicles` tool (via EXECUTE_TOOL) for the appropriate camera
- Provide cameraRef using the camera name/ID from context or the most relevant visible camera
- Pass optional parameters when appropriate:
  - durationSeconds (2-30, default 10)
  - sampleFps (1-12, default 5)
  - minConfidence (0.1-0.9, default 0.4)
  - useHighPrecision (default true; set false for low-power mode)

Remember: You are an intelligent agent with tools. Use them to make things happen, don't just describe what you could do.

MOTION WATCH POLICY:
- If the user says "detect motion", "watch for motion", "monitor motion" optionally with a camera name, call EXECUTE_TOOL with tool_id="motion_detector_watch" and parameters including cameraRef (use DISPLAY NAME if provided by user - CRITICAL: extract camera name from user message!), durationMinutes (default 15), and zonesOnly=true to prioritize user overlay areas. If they ask to stop, use tool_id="stop_motion_detector_watch".

OBJECT DETECTION WITH MOTION WATCH - INTELLIGENT WORKFLOW:
- When user says "look for [objects] at/on [camera]", follow this COMPLETE workflow:
  
  STEP 1 - CHECK FOR CAMERA WIDGET:
  - Check dashboard_state to see if camera widget exists for the specified camera
  - If NO widget found for that camera → Create it FIRST using create_camera_widget
  
  STEP 2 - ENABLE OBJECT DETECTION:
  - EXECUTE_TOOL enable_object_detection with cameraRef=[camera], classes=[objects]
  
  STEP 3 - START MOTION WATCH:
  - EXECUTE_TOOL motion_detector_watch with cameraRef=[camera]
  
  STEP 4 - EXTRACT CAMERA NAME CORRECTLY (CRITICAL!):
  - Parse the user message to find camera references
  - Common patterns to extract:
    * "at the [camera name]" → Extract [camera name]
    * "on [camera name]" → Extract [camera name]
    * "for [camera name]" → Extract [camera name]
    * "[camera name] camera" → Extract [camera name]
  - Match to actual cameras from cameras.json:
    * "primary entrance" / "the primary entrance" / "at primary entrance" → cameraRef="Primary Entrance"
    * "scout" / "the scout" / "on scout" → cameraRef="SCOUT"
    * "workshop" / "the workshop" / "at workshop" → cameraRef="Workshop"
    * "north entrance" / "the north entrance" → cameraRef="North Entrance"
  - CRITICAL: NEVER pass undefined or null! Extract camera name from message or use visible camera!
  - If you cannot extract a camera name, check dashboard_state for visible cameras and use the first one
  
- Complete Examples with Exact Parameter Extraction:
  
  - "look for cars at the primary entrance":
    * Extract camera: "at the primary entrance" contains "primary entrance" → Match to "Primary Entrance" from cameras.json
    * Check dashboard: No Primary Entrance widget
    * Actions:
      1. create_camera_widget(cameraId="Primary Entrance")
      2. enable_object_detection(cameraRef="Primary Entrance", classes=["car"])
      3. motion_detector_watch(cameraRef="Primary Entrance")  ← NOT undefined!
  
  - "look for laptops on scout":
    * Extract camera: "on scout" contains "scout" → Match to "SCOUT" from cameras.json
    * Check dashboard: SCOUT widget exists
    * Actions:
      1. enable_object_detection(cameraRef="SCOUT", classes=["laptop"])  ← NOT undefined!
      2. motion_detector_watch(cameraRef="SCOUT")  ← NOT undefined!
  
  - "detect benches on workshop":
    * Extract camera: "on workshop" contains "workshop" → Match to "Workshop" from cameras.json
    * Check dashboard: No Workshop widget
    * Actions:
      1. create_camera_widget(cameraId="Workshop")
      2. enable_object_detection(cameraRef="Workshop", classes=["bench"])  ← NOT undefined!
      3. motion_detector_watch(cameraRef="Workshop")  ← NOT undefined!
  
  - "look for people" (single camera visible on dashboard):
    * No camera name in message
    * Check dashboard_state: One camera visible (e.g., "North Entrance")
    * Use visible camera: cameraRef="North Entrance"
    * Actions:
      1. enable_object_detection(cameraRef="North Entrance", classes=["person"])  ← Use visible camera!
      2. motion_detector_watch(cameraRef="North Entrance")  ← Use visible camera!

CRITICAL CAMERA NAME EXTRACTION RULES (MANDATORY!):
1. ALWAYS check the AVAILABLE CAMERAS list provided in this prompt
2. Match user text to actual camera names using fuzzy matching
3. Common camera names (THESE ARE THE REAL NAMES - use exactly):
   - "North Entrance" (NOT "north entrance" - use exact case)
   - "Workshop" (NOT "workshop")
   - "Primary Entrance" (NOT "primary entrance" or "Main north entrance" - exact case!)
   - "Operations Hub"
   - "Front Landing"
   - "Storage Bay 01: right"
   - "Storage Bay 01: left"
   - "SCOUT" (all caps!)
4. Extraction examples (USER MESSAGE → cameraRef value):
   - "at the primary entrance" → cameraRef="Primary Entrance" (match "primary entrance" to "Primary Entrance")
   - "on scout" → cameraRef="SCOUT" (match "scout" to "SCOUT")
   - "at workshop" → cameraRef="Workshop" (match "workshop" to "Workshop")
   - "the north entrance camera" → cameraRef="North Entrance" (match "north entrance" to "North Entrance")
5. NEVER EVER use "undefined", "null", "", or any non-existent camera name
6. If you cannot find a camera name in the message, check dashboard_state for visible camera widgets and use that camera's name
7. When in doubt, use the first camera from the available cameras list rather than undefined
"""

            # For local LLMs with limited context, use condensed prompt
            if using_local_llm:
                logger.info("Using condensed prompt for local LLM (2048 token limit)")
                
                # Get camera list from devices
                camera_list = []
                try:
                    if context and hasattr(context, 'devices'):
                        camera_list = [d.get('name') for d in context.devices if d.get('type') == 'camera' and d.get('name')]
                    if not camera_list and context and hasattr(context, 'cameras') and context.cameras:
                        camera_list = [cam.get('name', '') for cam in context.cameras if cam.get('name')]
                except:
                    pass
                
                logger.info(f"Camera list for local LLM: {camera_list}")
                
                # Get learned patterns to help small model
                learned_examples = ""
                if self.intent_learner:
                    learned_examples = self.intent_learner.get_compact_tool_guide(max_examples=3)
                
                # Ultra-compact system prompt with clear examples
                cameras_str = ', '.join(camera_list) if camera_list else 'Ask user'
                active_cam = self.conversation_context.get("last_camera")
                conv_hint = f"Active camera: {active_cam}" if active_cam else "Active camera: (none)"
                
                system_prompt = f"""You are Knoxnet VMS Beta AI.
Cameras: {cameras_str}
{conv_hint}

Camera defaulting rule:
- If user does NOT specify a camera name, use Active camera.
- If Active camera is (none), use the first camera from Cameras.

Tools: create_camera_widget, take_camera_snapshot, snapshot_detect, motion_detector_watch

{learned_examples if learned_examples else "Examples:"}
"show primary entrance" → {{"message": "Opening", "actions": [{{"action": "EXECUTE_TOOL", "tool": "create_camera_widget", "parameters": {{"cameraRef": "Primary Entrance"}}}}]}}

"what's at X" → {{"message": "Analyzing", "actions": [{{"action": "EXECUTE_TOOL", "tool": "take_camera_snapshot", "parameters": {{"cameraRef": "X", "analysisPrompt": "what do you see"}}}}]}}

"is there a car behind the tree" (no camera name) → {{"message":"Checking","actions":[{{"action":"EXECUTE_TOOL","tool":"take_camera_snapshot","parameters":{{"cameraRef":"<Active camera>","analysisPrompt":"Look specifically for a car behind the tree (even if partially occluded). Answer yes/no and where."}}}}]}}

"how many cars" (no camera name) → {{"message":"Counting","actions":[{{"action":"EXECUTE_TOOL","tool":"snapshot_detect","parameters":{{"cameraRef":"<Active camera>","objectClasses":["car"]}}}}]}}

Return ONLY JSON. No extra text."""
                
                safe_system_content = system_prompt
                user_content = f"User: {message}"
                
            else:
                # Full prompt for cloud LLMs with large context windows
                system_prompt = prompt_template.replace("<<ENV_CTX>>", environment_context)
                system_prompt = system_prompt.replace("<<CONV_CTX>>", conversation_context_summary)

                # Build a list of allowed camera names from context to help with extraction
                available_cameras_text = ""
                if context and hasattr(context, 'cameras') and context.cameras:
                    camera_names = [cam.get('name', '') for cam in context.cameras if cam.get('name')]
                    if camera_names:
                        available_cameras_text = f"\n\nAVAILABLE CAMERAS IN SYSTEM:\n{', '.join(camera_names)}\n\nWhen extracting camera names from user messages, match to these EXACT names (case-sensitive).\n"
                
                system_prompt = system_prompt + available_cameras_text
            
                # Build a list of allowed camera names from context to prevent fabrication
                try:
                    # Prefer provided devices; fallback to any 'cameras' list in context
                    allowed_cameras = [
                        d.get('name') for d in (getattr(context, 'devices', []) or [])
                        if isinstance(d, dict) and d.get('type') == 'camera' and d.get('name')
                    ]
                    if not allowed_cameras:
                        extra_names = []
                        try:
                            # context might be a dataclass; use asdict to access extras
                            ctx_dict = asdict(context)
                            extra_names = [n for n in (ctx_dict.get('cameras') or []) if isinstance(n, str) and n]
                        except Exception:
                            pass
                        allowed_cameras = extra_names
                except Exception:
                    allowed_cameras = []

                # Augment system prompt to forbid fabricated camera names
                if allowed_cameras:
                    allowed_names_clause = (
                        "\nALLOWED CAMERA NAMES (reference ONLY from this list):\n- "
                        + "\n- ".join(allowed_cameras)
                        + "\nSTRICT: If a name is not in this list, respond with 'Unknown camera name' and ask the user to specify. Do NOT suggest or invent alternative names.\n"
                    )
                    prompt_with_policy = system_prompt + allowed_names_clause
                else:
                    prompt_with_policy = system_prompt + "\nIf camera names are not provided, avoid naming cameras; describe them generically.\n"

                # Escape braces in system content to prevent any downstream formatter from treating them as specifiers
                safe_system_content = prompt_with_policy.replace('{', '{{').replace('}', '}}')

                # Build user content without f-strings to avoid brace parsing edge-cases
                try:
                    user_ctx_json = json.dumps(asdict(context), indent=2, default=str)
                except Exception:
                    # Fallback to a minimal context representation
                    user_ctx_json = json.dumps({
                        "devices": getattr(context, 'devices', []),
                        "layout": getattr(context, 'layout', []),
                        "system_time": getattr(context, 'system_time', None)
                    }, indent=2, default=str)

                user_content = "Context: " + user_ctx_json + "\n\nUser message: " + str(message)

            # Create messages
            messages = [
                {"role": "system", "content": safe_system_content},
                {"role": "user", "content": user_content}
            ]
            
            # Use vision if image provided
            if image:
                logger.info("Processing vision request...")
                # Robust, neutral vision prompt that respects initial user intent
                intent_text = str(message).strip()
                vision_prompt = (
                    "You are analyzing an image. Respond directly to the user's request using only what is visible. "
                    "Do not include disclaimers, meta-comments, or references to systems or security. If a requested detail is not visible or not legible, say 'not visible' or 'not readable'. "
                    "If the image contains any overlaid graphics (colored polygons, lines, markers, or labels), mention them first only if they are actually visible and relate them to the answer. "
                    "Be precise and concise (1-2 short sentences).\n\n"
                    f"User request: {intent_text}"
                )
                
                logger.info(f"Vision prompt: {vision_prompt[:200]}...")
                logger.info(f"Using vision model: {vision_model_for_request}")
                
                vision_result = await provider_for_request.vision(
                    image=image,
                    prompt=vision_prompt,
                    model=vision_model_for_request,
                    opts={"temperature": 0.3, "max_tokens": 200, "timeout": self.timeout}
                )
                
                logger.info(f"Vision result: {vision_result.get('content', '')[:200]}...")
                vision_data = vision_result.get("analysis") if isinstance(vision_result, dict) else None
                vision_caption = ""
                if isinstance(vision_data, dict):
                    vision_caption = vision_data.get("caption") or ""
                if not vision_caption:
                    vision_caption = vision_result.get("content", "")
                
                # Add vision analysis to persistent environment context
                # Extract camera name from context if available
                camera_name = None
                for device in context.devices:
                    if device.get('is_visible_on_dashboard') and device.get('screenshot'):
                        camera_name = device.get('name')
                        break
                
                self._add_to_environment_context(vision_caption, camera_name)

                if vision_caption:
                    # Add vision analysis to context
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Vision analysis: {vision_caption}",
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Based on the vision analysis above, respond to: {message}",
                        }
                    )
            else:
                logger.info("No image provided, skipping vision processing")
            
            logger.info("About to call provider.chat")
            # Get AI response
            result = await provider_for_request.chat(
                messages=messages,
                model=chat_model_for_request,
                opts={"temperature": 0.7, "max_tokens": 150, "timeout": self.timeout}
            )
            logger.info("Provider.chat completed successfully")
            self.provider_failures = 0  # Reset failure count on success
            
            logger.info("About to parse AI response")
            # Parse response
            ai_response = self._parse_ai_response(result["content"])
            # Track provider/model used (for UI)
            try:
                if not ai_response.provider and provider_id_for_request:
                    ai_response.provider = str(provider_id_for_request)
                if not ai_response.model and chat_model_for_request:
                    ai_response.model = str(chat_model_for_request)
            except Exception:
                pass
            
            # Enforce camera name policy in final message by sanitizing unknown names
            try:
                allowed_names: List[str] = []
                try:
                    allowed_names = [
                        d.get('name') for d in (getattr(context, 'devices', []) or [])
                        if isinstance(d, dict) and d.get('type') == 'camera' and d.get('name')
                    ]
                except Exception:
                    pass
                if not allowed_names:
                    try:
                        ctx_dict = asdict(context)
                        allowed_names = [n for n in (ctx_dict.get('cameras') or []) if isinstance(n, str) and n]
                    except Exception:
                        pass
                ai_response.message = self._sanitize_ai_camera_references(ai_response.message, allowed_names)
            except Exception as _san_e:
                logger.warning(f"Camera name sanitization skipped: {_san_e}")
            logger.info("AI response parsed successfully")
            
            # Always preserve the vision analysis if we have it
            if image and vision_result:
                content_text = vision_caption or vision_result.get("content", "")
                if content_text:
                    ai_response.vision_analysis = content_text
                    ai_response.vision_analysis_data = vision_data
                    logger.info(f"Preserved vision analysis: {len(content_text)} chars")
            else:
                ai_response.vision_analysis = None
                ai_response.vision_analysis_data = None
            
            # Update conversation context with this interaction
            self._update_conversation_context(message, context, ai_response.actions)
            
            # Learn from successful tool executions (for future pattern matching)
            if self.intent_learner and ai_response.actions and not image:  # Only learn from non-vision queries
                for action in ai_response.actions:
                    try:
                        self.intent_learner.learn_from_execution(
                            user_query=message,
                            tool=action.tool,
                            parameters=action.parameters,
                            success=True,
                            feedback_score=1.0
                        )
                        logger.debug(f"Learned: {message[:50]} → {action.tool}")
                    except Exception as learn_err:
                        logger.debug(f"Could not learn from execution: {learn_err}")
            
            # Log telemetry (simplified to avoid recursion)
            latency = time.time() - start_time
            logger.info(f"AI chat completed: latency={latency:.2f}s, actions={len(ai_response.actions)}, vision_used={vision_used}")
            
            return ai_response
            
        except Exception as e:
            self.provider_failures += 1
            logger.error(f"AI chat error (failure count: {self.provider_failures}): {e}")
            
            # If we have a successful vision analysis, use it to provide a meaningful response
            if image and vision_result:
                logger.info("Using successful vision analysis despite chat completion error")
                vision_content = vision_caption or vision_result.get("content", "")
                
                # Add to environment context even in error case
                camera_name = None
                for device in context.devices:
                    if device.get('is_visible_on_dashboard') and device.get('screenshot'):
                        camera_name = device.get('name')
                        break
                
                self._add_to_environment_context(vision_content, camera_name)
                
                # Create a response based on the vision analysis
                if "camera feed" in vision_content.lower() or "primary entrance" in vision_content.lower():
                    response_message = f"Based on the camera feed analysis: {vision_content}"
                else:
                    response_message = f"Here's what I can see: {vision_content}"
                
                # Log telemetry for partial success (simplified)
                latency = time.time() - start_time
                logger.info(f"AI chat completed with vision fallback: latency={latency:.2f}s, vision_used={vision_used}")
                
                return AIResponse(
                    message=response_message,
                    vision_analysis=vision_content,
                    vision_analysis_data=vision_data,
                    error=None,  # Don't show error to user since we have useful response
                )
            
            # Check if this is a rate limit error
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "rate limit" in error_str or "quota" in error_str
            
            # Try fallback providers in order when primary fails (prefer per-request chain if provided)
            if not fallback_candidates:
                fallback_candidates = self.fallback_chain if self.fallback_chain else (
                    [self.fallback_provider] if self.fallback_provider else []
                )
            if fallback_candidates and (is_rate_limit or self.provider_failures > 0):
                logger.warning(f"Primary provider failed (rate_limit={is_rate_limit}), attempting fallback chain")
                for fb_idx, fb_provider in enumerate(fallback_candidates):
                    if not fb_provider:
                        continue
                    try:
                        fallback_response = await fb_provider.chat(
                            messages=messages,
                            model=chat_model_for_request,
                            opts={"max_tokens": 1000, "temperature": 0.7, "timeout": 60}
                        )

                        # Extract content from fallback response
                        if isinstance(fallback_response, dict):
                            if "choices" in fallback_response and len(fallback_response["choices"]) > 0:
                                content = fallback_response["choices"][0]["message"]["content"]
                            elif "content" in fallback_response:
                                content = fallback_response["content"]
                            else:
                                content = str(fallback_response)
                        else:
                            content = str(fallback_response)
                        
                        logger.info(f"✓ Fallback provider #{fb_idx+1} succeeded! Response length: {len(content)}")
                        
                        # Parse JSON from local LLM response (may be wrapped in text)
                        actions_to_execute = []
                        message_to_show = content
                        
                        try:
                            # Extract JSON - handle nested braces properly
                            json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', content, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                logger.info(f"Found JSON in response: {json_str[:200]}...")
                                parsed = json.loads(json_str)
                                
                                if isinstance(parsed, dict):
                                    # Standard format: {"message": "...", "actions": [...]}
                                    if 'message' in parsed and 'actions' in parsed:
                                        message_to_show = parsed.get('message', content)
                                        actions_to_execute = parsed.get('actions', [])
                                        logger.info(f"✓ Standard format: {len(actions_to_execute)} actions")
                                    
                                    # Alternative format: Single action object {"action": "EXECUTE_TOOL", "tool": "...", "parameters": {...}}
                                    elif 'action' in parsed and parsed.get('action') == 'EXECUTE_TOOL':
                                        logger.info("✓ Single action format detected, wrapping it")
                                        actions_to_execute = [parsed]
                                        message_to_show = f"Executing {parsed.get('tool', 'action')}"
                                    
                                    # Array of actions: [{"action": "EXECUTE_TOOL", ...}]
                                    elif 'actions' in parsed:
                                        actions_to_execute = parsed.get('actions', [])
                                        message_to_show = "Executing actions"
                                        logger.info(f"✓ Actions-only format: {len(actions_to_execute)} actions")
                                    
                                    else:
                                        logger.warning(f"JSON found but unknown format: {list(parsed.keys())}")
                                        
                                    for i, action in enumerate(actions_to_execute):
                                        logger.info(f"  Action {i}: {action.get('tool', 'unknown')} - {action.get('parameters', {})}")
                            else:
                                # Try parsing entire content as JSON
                                parsed = json.loads(content)
                                if isinstance(parsed, dict):
                                    message_to_show = parsed.get('message', content)
                                    actions_to_execute = parsed.get('actions', [])
                                    logger.info(f"✓ Full parse: {len(actions_to_execute)} actions")
                        except Exception as parse_err:
                            logger.warning(f"Could not parse JSON from local LLM: {parse_err}")
                            logger.warning(f"Response content: {content[:500]}")
                            # Try to extract action hints from text
                            if 'create_camera_widget' in content.lower() and 'primary entrance' in content.lower():
                                logger.info("Detected camera widget intent, creating manual action")
                                actions_to_execute = [{
                                    "action": "EXECUTE_TOOL",
                                    "tool": "create_camera_widget",
                                    "parameters": {"cameraRef": "Primary Entrance"}
                                }]
                                message_to_show = "Opening Primary Entrance camera"
                            else:
                                message_to_show = content
                                actions_to_execute = []
                        
                        # Convert actions to AIAction format
                        ai_actions = []
                        for action in actions_to_execute:
                            if isinstance(action, dict):
                                # Extract tool name from 'tool' or 'action' field
                                tool_name = action.get('tool') or action.get('action')
                                params = action.get('parameters', {})
                                
                                # If it's EXECUTE_TOOL, extract the actual tool from parameters or infer it
                                if tool_name == 'EXECUTE_TOOL' or not tool_name:
                                    tool_name = action.get('tool_name') or action.get('tool')
                                    
                                    # Infer tool from parameters if still not found
                                    if not tool_name or tool_name == 'EXECUTE_TOOL':
                                        if 'cameraRef' in params or 'camera_id' in params:
                                            tool_name = 'create_camera_widget'
                                            logger.info(f"  Inferred tool 'create_camera_widget' from cameraRef parameter")
                                        elif 'script' in params or 'script_content' in params:
                                            tool_name = 'create_python_script'
                                            logger.info(f"  Inferred tool 'create_python_script' from script parameter")
                                        elif 'all_cameras' in params or action.get('all_cameras'):
                                            tool_name = 'create_all_cameras_grid'
                                            logger.info(f"  Inferred tool 'create_all_cameras_grid' from all_cameras parameter")
                                        else:
                                            tool_name = 'unknown_tool'
                                            logger.warning(f"  Could not infer tool from parameters: {list(params.keys())}")
                                
                                # Map tool name to kind (for backwards compatibility)
                                kind = tool_name
                                
                                # Create AIAction with proper fields
                                ai_actions.append(AIAction(
                                    kind=kind,
                                    tool_id=tool_name,
                                    parameters=params
                                ))
                                logger.info(f"  Converted action: kind={kind}, tool_id={tool_name}, params={list(params.keys())}")
                        
                        logger.info(f"Returning {len(ai_actions)} AI actions to execute")
                        
                        # Get provider name for feedback tracking (from fallback provider)
                        fallback_provider_name = type(fb_provider).__name__.replace('Provider', '').lower()
                        if fallback_provider_name == 'huggingfacelocal':
                            fallback_provider_name = 'Local LLM'
                        
                        return AIResponse(
                            message=message_to_show,
                            actions=ai_actions,
                            error=None,
                            provider=fallback_provider_name,
                            model=os.environ.get("LLM_DEFAULT_MODEL", "TinyLlama")
                        )
                        
                    except Exception as fallback_error:
                        logger.error(f"Fallback provider #{fb_idx+1} failed: {fallback_error}")
                        continue
            
            # If rate limited and this looks like a camera query, provide helpful guidance
            if is_rate_limit and not image:
                logger.warning("OpenAI rate limited, checking if this is a camera query")
                
                # Try to extract camera reference from message
                message_lower = message.lower()
                camera_keywords = ["what's at", "whats at", "check", "show", "describe", "look at"]
                
                if any(keyword in message_lower for keyword in camera_keywords):
                    # This looks like a camera query
                    logger.info("Detected camera query while rate limited - suggesting vision alternatives")
                    
                    return AIResponse(
                        message="⚠️ OpenAI API is currently rate limited (quota exceeded). For camera analysis, try:\n\n1. Click the subtitle button (🎞) on any camera widget for instant AI analysis\n2. Right-click camera → Settings → Scene Analysis for full control\n\nThe local vision service (BLIP + YOLO) is running and provides free, detailed scene descriptions.",
                        error=None  # Don't show as error since we're being helpful
                    )
            
            # Log error telemetry (simplified)
            latency = time.time() - start_time
            logger.error(f"AI chat failed: latency={latency:.2f}s, vision_used={vision_used}, error={str(e)}")
            
            # Provide helpful error message based on error type
            if is_rate_limit:
                fallback_msg = " A local LLM fallback is available - check LLM Settings to configure it." if not self.fallback_provider else ""
                return AIResponse(
                    message=f"⚠️ OpenAI API is rate limited (quota exceeded).{fallback_msg} The local vision service is available - click the subtitle button (🎞) on camera widgets for free AI analysis.",
                    error="Rate limit exceeded - Local vision still available"
                )
            
            return AIResponse(
                message="I encountered an error processing your request. Please try again.",
                error=str(e)
            )

    def _try_handle_natural_query(self, message: str, context: AIContext) -> Optional[AIResponse]:
        """Handle simple NL queries directly using existing vision tools.

        Supports:
        - "how many cars at <camera>"
        - "what color are they" (follows a previous vehicle query)
        """
        try:
            raw_message = (message or "").strip()
            # If the UI passes terminal history (including previous assistant output),
            # extract only the last "$ ..." command line so keywords like "yesterday"
            # from earlier output don't override the user's current query.
            user_query = raw_message
            try:
                cmd_matches = list(re.finditer(r"(?m)^\s*(?:\[\d{1,2}:\d{2}:\d{2}\]\s*)?\$\s*(.+?)\s*$", raw_message))
                if not cmd_matches:
                    cmd_matches = list(re.finditer(r"(?m)^\s*\\$\s*(.+?)\s*$", raw_message))
                if cmd_matches:
                    user_query = cmd_matches[-1].group(1).strip()
            except Exception:
                user_query = raw_message

            text = user_query.lower()
            if not text:
                return None

            def _last_day_token(s: str) -> Optional[str]:
                try:
                    m = None
                    for m in re.finditer(r"\b(today|yesterday)\b", s):
                        pass
                    return m.group(1) if m else None
                except Exception:
                    return None

            # --- Events / timeline search intent (Motion Watch index) ---
            # Keep this deterministic so the feature works even when small/offline models are used.
            if any(k in text for k in ["timeline", "event search", "events search", "search events", "show events", "show me events", "motion watch events"]):
                # Try to infer camera name if user mentions one
                camera = self._infer_camera_from_message(user_query, context)
                cam_name = None
                if camera:
                    cam_name = camera.get("name") or camera.get("id") or camera.get("camera_id")
                if not cam_name:
                    cam_name = self.conversation_context.get("last_camera")

                # Query text: strip leading command-ish words but keep the rest
                q = (user_query or "").strip()
                for prefix in ["timeline", "events", "event search", "events search", "search events", "show events", "show me events"]:
                    if q.lower().startswith(prefix):
                        q = q[len(prefix):].strip(" :,-")
                        break

                params: Dict[str, Any] = {"query": q}
                if cam_name:
                    params["cameraRef"] = str(cam_name)

                return AIResponse(
                    message="Searching event timeline…",
                    actions=[AIAction(kind="execute_tool", tool_id="events_search", parameters=params)],
                )

            # --- Security report intent (HTML report with crops + links) ---
            # Trigger on common phrasing so users can say "give me a report for ...".
            if (
                ("report" in text and any(k in text for k in ["yesterday", "today", "passed", "events", "timeline"]))
                or any(k in text for k in ["security report", "incident report", "generate report", "make a report", "report timeline", "report for", "give me a report"])
            ):
                camera = self._infer_camera_from_message(user_query, context)
                cam_name = None
                if camera:
                    cam_name = camera.get("name") or camera.get("id") or camera.get("camera_id")
                if not cam_name:
                    cam_name = self.conversation_context.get("last_camera")

                # Strip command-ish words, keep remaining as query
                q = (user_query or "").strip()
                for prefix in ["give me a report", "report for", "security report", "incident report", "generate report", "make a report", "report timeline", "report"]:
                    if q.lower().startswith(prefix):
                        q = q[len(prefix):].strip(" :,-")
                        break

                # Parse day window + optional class filters so reports return *all* matching events,
                # instead of being constrained by free-text search terms.
                now_ts = int(time.time())
                start_ts = None
                end_ts = None
                try:
                    from datetime import datetime as _dt
                    today = _dt.now()
                    start_today = _dt(year=today.year, month=today.month, day=today.day)
                    day_tok = _last_day_token(text)
                    if day_tok == "yesterday":
                        start_ts = int((start_today.timestamp() - 86400))
                        end_ts = int(start_today.timestamp())
                    elif day_tok == "today":
                        start_ts = int(start_today.timestamp())
                        end_ts = now_ts
                except Exception:
                    start_ts, end_ts = None, None

                # Optional detection class filter from query ("all cars ...", "all trucks ...")
                det_classes: List[str] = []
                if "truck" in text or "trucks" in text:
                    det_classes = ["truck"]
                elif "car" in text or "cars" in text:
                    det_classes = ["car"]
                elif "vehicle" in text or "vehicles" in text:
                    det_classes = ["car", "truck", "bus", "motorcycle"]

                params: Dict[str, Any] = {
                    # Prefer filters over free-text search so we don't accidentally narrow to 1 event.
                    "query": "",
                    "limit": 5000,
                    "auto_refresh": True,
                }
                if cam_name:
                    params["cameraRef"] = str(cam_name)
                if start_ts is not None:
                    params["start_ts"] = int(start_ts)
                if end_ts is not None:
                    params["end_ts"] = int(end_ts)
                if det_classes:
                    params["filters"] = {"detection": {"classes": det_classes, "min_confidence": 0.25}}
                else:
                    # Fall back to the cleaned query if we didn't detect a specific class filter.
                    params["query"] = q or ""

                return AIResponse(
                    message="Opening event timeline report...",
                    actions=[AIAction(kind="execute_tool", tool_id="events_report", parameters=params)],
                )

            # --- "passed ... today" / "how many ... today" vehicle queries via event index ---
            # Example: "how many white trucks passed the produce stand today"
            if ("today" in text or "yesterday" in text or "passed" in text or "pass" in text) and any(
                v in text for v in ["truck", "trucks", "vehicle", "vehicles", "car", "cars"]
            ):
                camera = self._infer_camera_from_message(user_query, context)
                cam_name = None
                if camera:
                    cam_name = camera.get("name") or camera.get("id") or camera.get("camera_id")
                if not cam_name:
                    cam_name = self.conversation_context.get("last_camera")

                # Time window
                now_ts = int(time.time())
                start_ts = None
                end_ts = None
                try:
                    from datetime import datetime as _dt
                    today = _dt.now()
                    start_today = _dt(year=today.year, month=today.month, day=today.day)
                    day_tok = _last_day_token(text)
                    if day_tok == "yesterday":
                        start_ts = int((start_today.timestamp() - 86400))
                        end_ts = int(start_today.timestamp())
                    elif day_tok == "today":
                        start_ts = int(start_today.timestamp())
                        end_ts = now_ts
                except Exception:
                    start_ts, end_ts = None, None

                # Basic color extraction (matches event_index_service dominant_color buckets)
                dom_color = None
                for c in ["white", "black", "gray", "red", "green", "blue", "yellow", "brown"]:
                    if c in text:
                        dom_color = c
                        break

                # Query text (FTS / LIKE) is more robust than strict class filters, since some indexes
                # may have captions but missing detection_classes depending on vision availability.
                q_terms: List[str] = []
                if "truck" in text:
                    q_terms.append("truck")
                elif "car" in text or "cars" in text:
                    q_terms.append("car")
                elif "vehicle" in text or "vehicles" in text:
                    q_terms.append("vehicle")
                q_text = " ".join(q_terms).strip()

                # If the user asked for a report, generate the report instead of counting/searching.
                if "report" in text:
                    params: Dict[str, Any] = {"query": message}
                    if cam_name:
                        params["cameraRef"] = str(cam_name)
                    return AIResponse(
                        message="Opening event timeline report...",
                        actions=[AIAction(kind="execute_tool", tool_id="events_report", parameters=params)],
                    )

                # If the user asked for a total ("how many", "total"), compute it server-side so
                # older desktop clients (without events_count tool support) still get an answer.
                wants_total = any(k in text for k in ["how many", "total", "in total", "count"])

                # Detection-level filters for accuracy (object-level index)
                det_classes: List[str] = []
                if q_terms:
                    if q_terms[0] in {"car", "truck"}:
                        det_classes = [q_terms[0]]
                    elif q_terms[0] == "vehicle":
                        det_classes = ["car", "truck", "bus", "motorcycle"]

                if wants_total:
                    try:
                        t0 = time.time()
                        # Route through the backend endpoint so results are consistent with the
                        # running server's index paths and state (not sensitive to CWD).
                        payload: Dict[str, Any] = {
                            "cameraRef": str(cam_name).strip() if isinstance(cam_name, str) and str(cam_name).strip() else None,
                            "start_ts": int(start_ts) if start_ts is not None else None,
                            "end_ts": int(end_ts) if end_ts is not None else None,
                            "detection_classes": det_classes,
                            "filters": {"detection": {"min_confidence": 0.25}},
                            "auto_refresh": True,
                        }
                        resp = requests.post(f"{self.base_url}/api/events/vehicle_count", json=payload, timeout=10)
                        j = resp.json() if resp.ok else {}
                        result = (j.get("data") if isinstance(j, dict) else {}) or {}
                        ms = int((time.time() - t0) * 1000)

                        coverage_obj = result.get("coverage") if isinstance(result.get("coverage"), dict) else {}
                        det_count = int(coverage_obj.get("detection_count") or 0)
                        ev_count = int(coverage_obj.get("event_count") or 0)
                        total_in_range = int(coverage_obj.get("events_total_in_range") or 0)
                        det_indexed = int(coverage_obj.get("events_detection_indexed_in_range") or 0)
                        det_missing = int(coverage_obj.get("events_detection_not_indexed_in_range") or 0)
                        unique_vehicles = int(result.get("unique_vehicle_count") or 0)

                        cam_label = str(cam_name) if isinstance(cam_name, str) and cam_name.strip() else "all cameras"
                        base_label = "vehicle"
                        if det_classes == ["car"]:
                            base_label = "car"
                        elif det_classes == ["truck"]:
                            base_label = "truck"
                        elif isinstance(det_classes, list) and set(det_classes) == {"car", "truck", "bus", "motorcycle"}:
                            base_label = "vehicle"
                        color_prefix = str(dom_color).strip().lower() if isinstance(dom_color, str) and dom_color.strip() else ""
                        pretty = f"{color_prefix} {base_label}".strip() if color_prefix else base_label
                        det_label = f"{pretty} detection" + ("" if det_count == 1 else "s")
                        ev_label = "capture" + ("" if ev_count == 1 else "s")
                        uniq_label = f"unique {pretty} vehicle" + ("" if unique_vehicles == 1 else "s")

                        coverage = f"{det_indexed}/{total_in_range}" if total_in_range else "0/0"
                        note = ""
                        if det_missing > 0:
                            note = (
                                f"\n\nNote: detections are not indexed for {det_missing} event(s) in this window yet, "
                                f"so this total is a lower bound. Run `events reindex` to improve coverage."
                            )
                        day_tok = _last_day_token(text)
                        if day_tok == "yesterday":
                            window_label = "Yesterday"
                        elif day_tok == "today":
                            window_label = "Today"
                        else:
                            window_label = "In this window"

                        return AIResponse(
                            message=(
                                f"{window_label} on {cam_label}:\n"
                                f"- {uniq_label} (tracks): {unique_vehicles}\n"
                                f"- Matching {ev_label} (>=1 match): {ev_count}\n"
                                f"- Matching {det_label}: {det_count}\n"
                                f"Coverage: {coverage} events indexed (computed in {ms}ms).\n"
                                f"Note: detections can be higher than captures when a single screenshot contains multiple cars. "
                                f"Unique vehicle tracks are best-effort across captures."
                                f"{note}"
                            ),
                            actions=[],
                        )
                    except Exception:
                        # Fall back to the UI tool path if anything goes wrong.
                        pass

                # Non-total queries still show a timeline.
                params: Dict[str, Any] = {"query": q_text}
                if cam_name:
                    params["cameraRef"] = str(cam_name)
                if start_ts is not None:
                    params["start_ts"] = int(start_ts)
                if end_ts is not None:
                    params["end_ts"] = int(end_ts)
                if dom_color:
                    params["dominant_color"] = dom_color

                return AIResponse(
                    message="Searching events...",
                    actions=[AIAction(kind="execute_tool", tool_id="events_search", parameters=params)],
                )

            # --- People/person detection queries ---
            # Example: "any people detected near Shed" / "person at Shed"
            if any(k in text for k in ["person", "people"]) and any(k in text for k in ["detect", "detected", "near", "at", "by", "around", "any"]):
                camera = self._infer_camera_from_message(user_query, context)
                cam_name = None
                if camera:
                    cam_name = camera.get("name") or camera.get("id") or camera.get("camera_id")
                if not cam_name:
                    cam_name = self.conversation_context.get("last_camera")

                # Default window: last 2 hours if user didn't specify a day keyword.
                now_ts = int(time.time())
                start_ts = None
                end_ts = None
                try:
                    from datetime import datetime as _dt
                    today = _dt.now()
                    start_today = _dt(year=today.year, month=today.month, day=today.day)
                    day_tok = _last_day_token(text)
                    if day_tok == "yesterday":
                        start_ts = int((start_today.timestamp() - 86400))
                        end_ts = int(start_today.timestamp())
                    elif day_tok == "today":
                        start_ts = int(start_today.timestamp())
                        end_ts = now_ts
                except Exception:
                    start_ts, end_ts = None, None
                if start_ts is None and end_ts is None:
                    start_ts = now_ts - 7200
                    end_ts = now_ts

                # Use timeline search (and let /events/search auto-refresh recent captures on 0 results)
                params: Dict[str, Any] = {
                    "query": "person",
                    "include_detections": True,
                    "start_ts": int(start_ts),
                    "end_ts": int(end_ts),
                    "limit": 25,
                }
                if cam_name:
                    params["cameraRef"] = str(cam_name)

                return AIResponse(
                    message=f"Checking for any people detected near {cam_name or 'cameras'}...",
                    actions=[AIAction(kind="execute_tool", tool_id="events_search", parameters=params)],
                )

            # --- Motion boxes intent (desktop/overlay feature; NOT motion watch) ---
            # Users often say "turn on motion boxes" meaning the red/green overlay on the camera,
            # not a background motion watch job.
            if ("motion box" in text) or ("motion boxes" in text) or ("show motion boxes" in text):
                camera = self._infer_camera_from_message(message, context)
                cam_name = None
                if camera:
                    cam_name = camera.get("name") or camera.get("id") or camera.get("camera_id")
                if not cam_name:
                    # fall back to conversation context
                    cam_name = self.conversation_context.get("last_camera")
                if not cam_name:
                    return AIResponse(message="Which camera should I enable motion boxes on? (e.g., 'Shed')", actions=[])

                enable = True
                if any(k in text for k in ["turn off", "disable", "stop", "hide"]):
                    enable = False

                return AIResponse(
                    message=f"{'Enabling' if enable else 'Disabling'} motion boxes on {cam_name}.",
                    actions=[
                        AIAction(kind="create_camera_widget", camera_id=str(cam_name)),
                        AIAction(kind="execute_tool", tool_id="set_motion_boxes", parameters={"cameraRef": str(cam_name), "enabled": enable}),
                    ],
                )

            # --- PTZ controls intent ---
            if "ptz" in text or "pan " in text or "tilt " in text or "zoom " in text:
                camera = self._infer_camera_from_message(message, context)
                cam_name = None
                if camera:
                    cam_name = camera.get("name") or camera.get("id") or camera.get("camera_id")
                if not cam_name:
                    cam_name = self.conversation_context.get("last_camera")
                if not cam_name:
                    return AIResponse(message="Which camera should I open PTZ controls for? (e.g., 'Shed')", actions=[])

                # If user asks to move PTZ (left/right/up/down/zoom), we still open the PTZ widget first.
                return AIResponse(
                    message=f"Opening PTZ controls for {cam_name}.",
                    actions=[
                        AIAction(kind="create_camera_widget", camera_id=str(cam_name)),
                        AIAction(kind="execute_tool", tool_id="open_ptz_widget", parameters={"cameraRef": str(cam_name)}),
                    ],
                )

            # --- Audio controls intent ---
            if ("audio" in text) or ("listen" in text) or ("talk" in text) or ("two way" in text) or ("two-way" in text) or ("twoway" in text) or ("record audio" in text):
                camera = self._infer_camera_from_message(message, context)
                cam_name = None
                if camera:
                    cam_name = camera.get("name") or camera.get("id") or camera.get("camera_id")
                if not cam_name:
                    cam_name = self.conversation_context.get("last_camera")
                if not cam_name:
                    return AIResponse(message="Which camera should I open audio for? (e.g., 'Shed')", actions=[])

                mode = "listen"
                if "two" in text and "way" in text:
                    mode = "twoWay"
                elif "talk" in text:
                    mode = "talk"
                elif "record" in text:
                    mode = "record"

                return AIResponse(
                    message=f"Opening audio controls for {cam_name} ({mode}).",
                    actions=[
                        AIAction(kind="create_camera_widget", camera_id=str(cam_name)),
                        AIAction(kind="execute_tool", tool_id="open_audio_widget", parameters={"cameraRef": str(cam_name), "mode": mode}),
                    ],
                )

            # --- Depth map / point cloud intent ---
            if ("depth" in text) or ("point cloud" in text) or ("pointcloud" in text):
                camera = self._infer_camera_from_message(message, context)
                cam_name = None
                if camera:
                    cam_name = camera.get("name") or camera.get("id") or camera.get("camera_id")
                if not cam_name:
                    cam_name = self.conversation_context.get("last_camera")
                if not cam_name:
                    return AIResponse(message="Which camera should I show depth for? (e.g., 'Shed')", actions=[])

                enable = True
                if any(k in text for k in ["turn off", "disable", "stop", "hide"]):
                    enable = False

                mode = "overlay"
                if "point" in text and "cloud" in text:
                    mode = "pointcloud"
                if "slam" in text:
                    mode = "slam"

                color = None
                for c in ["turbo", "jet", "viridis", "plasma", "inferno", "magma", "bone", "ocean"]:
                    if c in text:
                        color = c
                        break

                params = {"cameraRef": str(cam_name), "enabled": enable, "mode": mode}
                if color:
                    params["colorScheme"] = color

                return AIResponse(
                    message=f"{'Enabling' if enable else 'Disabling'} depth view for {cam_name}.",
                    actions=[
                        AIAction(kind="create_camera_widget", camera_id=str(cam_name)),
                        AIAction(kind="execute_tool", tool_id="open_depth_map_widget", parameters=params),
                    ],
                )

            # --- Occlusion / "behind X" follow-ups (best handled by snapshot QA, not counting) ---
            # Example: "is there a car behind the tree" right after "how many cars at Shed"
            if (("behind the tree" in text) or ("behind tree" in text) or ("behind that tree" in text)) and any(
                k in text for k in ["car", "cars", "vehicle", "vehicles", "truck", "trucks"]
            ):
                camera = self._infer_camera_from_message(message, context)
                cam_name = None
                if camera:
                    cam_name = camera.get("name") or camera.get("id") or camera.get("camera_id")
                if not cam_name:
                    cam_name = self.conversation_context.get("last_camera")
                if not cam_name:
                    return AIResponse(message="Which camera should I check? (e.g., 'Shed')", actions=[])

                analysis_prompt = (
                    "Look specifically for a car behind the tree (even if only partially visible / occluded). "
                    "Answer yes/no, and if yes, describe where it is relative to the tree. "
                    "If it's unclear, say 'not sure'."
                )
                return AIResponse(
                    message=f"Checking for a car behind the tree on {cam_name}...",
                    actions=[
                        AIAction(
                            kind="execute_tool",
                            tool_id="take_camera_snapshot",
                            parameters={"cameraRef": str(cam_name), "analysisPrompt": analysis_prompt},
                        )
                    ],
                )

            # --- Multi-intent "open/show camera + count objects" ---
            # Example: "show me the shed cam and tell me how many cars you see"
            wants_count = any(k in text for k in ["how many", "count "]) or ("how many" in text)
            if wants_count and any(obj in text for obj in ["car", "cars", "vehicle", "vehicles", "person", "people"]):
                camera = self._infer_camera_from_message(message, context)
                cam_name = None
                if camera:
                    cam_name = camera.get("name") or camera.get("id") or camera.get("camera_id")
                if not cam_name:
                    cam_name = self.conversation_context.get("last_camera")
                if not cam_name:
                    return None  # let LLM ask clarifying question

                # Infer object classes
                classes: List[str] = []
                if "car" in text or "cars" in text or "vehicle" in text or "vehicles" in text:
                    classes.append("car")
                if "person" in text or "people" in text:
                    classes.append("person")
                if not classes:
                    classes = ["car"]

                return AIResponse(
                    message=f"Opening {cam_name} and counting {', '.join(classes)}.",
                    actions=[
                        AIAction(kind="create_camera_widget", camera_id=str(cam_name)),
                        AIAction(kind="execute_tool", tool_id="snapshot_detect", parameters={"cameraRef": str(cam_name), "objectClasses": classes, "model": "auto", "confidence": 0.25}),
                    ],
                )

            # Vehicle count intent - DISABLED to allow robust multi-detector tool to handle it
            # The new snapshot_detect tool uses multiple detectors and has Vision API fallback
            if False:  # Disabled - let LLM use snapshot_detect tool instead
                camera = self._infer_camera_from_message(message, context)
                if not camera:
                    return AIResponse(message="Which camera? Say a name like 'Workshop' or 'North Entrance'.")

                camera_id = camera.get('id') or camera.get('camera_id') or camera.get('name')
                frame_b64 = self._get_latest_frame_base64(camera_id)
                if not frame_b64:
                    return AIResponse(message=f"No image from {camera.get('name','camera')}.")

                raw_b64 = frame_b64.split(",", 1)[1] if frame_b64.startswith("data:image") else frame_b64
                # OLD CODE - replaced by snapshot_detect tool
                detections = self.detect_objects_in_image(raw_b64, "mobilenet")
                vehicle_labels = {"car", "truck", "bus", "motorcycle", "vehicle"}
                vehicles = [d for d in detections if str(d.get('class_name','')).lower() in vehicle_labels]

                colors = self._estimate_vehicle_colors(raw_b64, vehicles)
                self._last_vehicle_context = {
                    'camera_id': camera_id,
                    'camera_name': camera.get('name', camera_id),
                    'count': len(vehicles),
                    'colors': colors,
                    'timestamp': time.time()
                }

                cam_name = camera.get('name', camera_id)
                count = len(vehicles)
                suffix = "" if count else " (none visible)"
                return AIResponse(message=f"{cam_name}: {count} vehicle(s){suffix}.")

            # Vehicle color follow-up intent
            if ("what color" in text) and ("car" in text or "vehicle" in text or "they" in text):
                ctx = getattr(self, '_last_vehicle_context', None)
                if not ctx or (time.time() - ctx.get('timestamp', 0) > 120):
                    return AIResponse(message="No recent vehicle view to reference.")
                colors = ctx.get('colors') or []
                if not colors:
                    return AIResponse(message="Colors unclear.")
                # Limit to top few
                unique_colors = list(dict.fromkeys(colors))
                return AIResponse(message=", ".join(unique_colors[:3]))

            return None
        except Exception as e:
            logger.warning(f"Natural query handler failed: {e}")
            return None

    def _infer_camera_from_message(self, message: str, context: AIContext) -> Optional[Dict[str, Any]]:
        """Find a camera from NL message using provided context devices.

        Uses robust fuzzy matching so minor typos like "mian north entrance" match "Primary Entrance".
        """
        try:
            devices = context.devices or []
            cams = [d for d in devices if isinstance(d, dict) and str(d.get('type','')).lower() == 'camera']
            if not cams:
                return None
            text = (message or "").lower()
            tokens = [t for t in text.replace("\n"," ").split() if t]

            def levenshtein(a: str, b: str) -> int:
                a, b = a.lower(), b.lower()
                if len(a) == 0: return len(b)
                if len(b) == 0: return len(a)
                prev = list(range(len(b)+1))
                for i, ca in enumerate(a, 1):
                    curr = [i]
                    for j, cb in enumerate(b, 1):
                        ins = curr[j-1] + 1
                        dele = prev[j] + 1
                        sub = prev[j-1] + (0 if ca == cb else 1)
                        curr.append(min(ins, dele, sub))
                    prev = curr
                return prev[-1]

            best = None
            for cam in cams:
                name = str(cam.get('name','')).lower()
                if not name:
                    continue
                score = 0
                if name in text:
                    score += 5
                if any(tok and tok in name for tok in tokens):
                    score += 2
                # typo tolerance
                if tokens:
                    dist = min(levenshtein(tok, name) for tok in tokens)
                    if dist <= 2:
                        score += 3
                if best is None or score > best[0]:
                    best = (score, cam)

            if best and best[0] > 0:
                return best[1]
            return None
        except Exception:
            return None

    def _load_cameras_from_disk(self) -> List[Dict[str, Any]]:
        """Load camera records from disk for name resolution when context.devices isn't provided."""
        candidates: List[Dict[str, Any]] = []
        paths = [Path("data/cameras.json"), Path("cameras.json")]
        for p in paths:
            try:
                if not p.exists():
                    continue
                data = json.loads(p.read_text())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and (item.get("name") or item.get("id")):
                            candidates.append(item)
            except Exception:
                continue
        return candidates

    def _best_match_camera_name(self, message: str, cameras: List[Dict[str, Any]]) -> Optional[str]:
        """Return best matching camera display name from a message."""
        try:
            text = (message or "").strip().lower()
            if not text or not cameras:
                return None

            def norm(s: str) -> str:
                return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

            msg_norm = f" {norm(text)} "

            # Attempt explicit extraction for common phrases
            stripped = text
            stripped = re.sub(r"^(please\\s+)?(open|show|view|display)\\s+", "", stripped).strip()
            stripped = re.sub(r"\\bcamera\\b", " ", stripped).strip()
            stripped = re.sub(r"\\bthe\\b", " ", stripped).strip()
            stripped_norm = f" {norm(stripped)} "

            best_name: Optional[str] = None
            best_score = -1

            for cam in cameras:
                name = cam.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                n = norm(name)
                if not n:
                    continue
                n_pad = f" {n} "
                score = 0

                if n_pad in msg_norm:
                    score += 100 + len(n)
                if stripped and n_pad in stripped_norm:
                    score += 120 + len(n)

                n_tokens = [t for t in n.split() if t]
                for tok in n_tokens:
                    if f" {tok} " in msg_norm:
                        score += 5
                    if stripped and f" {tok} " in stripped_norm:
                        score += 6

                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score >= 12:
                return best_name
            return None
        except Exception:
            return None

    def _try_handle_open_camera_command(self, message: str, context: AIContext) -> Optional[AIResponse]:
        """Handle 'open/show <camera> camera' directly without LLM."""
        text = (message or "").strip()
        if not text:
            return None

        lowered = text.lower()
        if not any(k in lowered for k in ("open", "show", "view", "display", "pull up", "bring up")):
            return None
        if "camera" not in lowered and not lowered.startswith(("open ", "show ", "view ", "display ")):
            return None

        cams: List[Dict[str, Any]] = []
        try:
            for d in (context.devices or []):
                if isinstance(d, dict) and str(d.get("type", "")).lower() == "camera":
                    cams.append(d)
        except Exception:
            cams = []

        if not cams:
            cams = self._load_cameras_from_disk()

        camera_name = self._best_match_camera_name(text, cams)
        if not camera_name:
            return AIResponse(
                message="Which camera? Try: `open the Shed camera` (use an exact camera name).",
                actions=[],
                error=None,
            )

        action = AIAction(
            kind="create_camera_widget",
            tool_id="create_camera_widget",
            parameters={"cameraRef": camera_name},
        )
        return AIResponse(
            message=f"Opening {camera_name}.",
            actions=[action],
            error=None,
            provider="fastpath",
        )

    def _get_latest_frame_base64(self, camera_id: str) -> Optional[str]:
        """Get latest frame from stream_server as data URL; fallback via HTTP snapshot."""
        try:
            if hasattr(self, 'stream_server') and self.stream_server:
                b64 = self.stream_server.get_frame_base64(camera_id)
                if b64:
                    return b64
        except Exception as e:
            logger.warning(f"Stream server base64 fetch failed for {camera_id}: {e}")

        try:
            base_url = os.environ.get('PUBLIC_BASE_URL') or os.environ.get('API_BASE_URL') or 'http://localhost:5000'
            resp = requests.get(f"{base_url}/api/cameras/{camera_id}/snapshot", timeout=5)
            if resp.ok and resp.headers.get('Content-Type','').startswith('image/'):
                import base64 as _b64
                encoded = _b64.b64encode(resp.content).decode('utf-8')
                return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            logger.warning(f"HTTP snapshot fallback failed for {camera_id}: {e}")
        return None

    def _estimate_vehicle_colors(self, image_base64: str, detections: List[Dict[str, Any]]) -> List[str]:
        """Estimate simple color names for detected vehicles using average HSV of bbox."""
        try:
            if not detections:
                return []
            # Decode image
            image_data = base64.b64decode(image_base64)
            arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return []

            def label_color(bgr_roi: np.ndarray) -> str:
                if bgr_roi.size == 0:
                    return "unknown"
                hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
                h = float(np.mean(hsv[..., 0]))
                s = float(np.mean(hsv[..., 1])) / 255.0
                v = float(np.mean(hsv[..., 2])) / 255.0
                if v < 0.2:
                    return "black"
                if s < 0.2:
                    return "white" if v > 0.8 else "gray"
                if (h < 10) or (h > 170):
                    return "red"
                if 10 <= h < 25:
                    return "orange"
                if 25 <= h < 35:
                    return "yellow"
                if 35 <= h < 85:
                    return "green"
                if 85 <= h < 130:
                    return "blue"
                if 130 <= h < 160:
                    return "purple"
                return "gray"

            h_img, w_img = img.shape[:2]
            colors: List[str] = []
            for d in detections:
                bbox = d.get('bbox') or d.get('box') or []
                if not bbox or len(bbox) < 4:
                    continue
                x, y, w, h = bbox
                x0 = max(0, int(x))
                y0 = max(0, int(y))
                x1 = min(w_img, int(x + w))
                y1 = min(h_img, int(y + h))
                if x1 <= x0 or y1 <= y0:
                    continue
                roi = img[y0:y1, x0:x1]
                colors.append(label_color(roi))
            return colors
        except Exception as e:
            logger.warning(f"Color estimation failed: {e}")
            return []
    
    def _add_to_environment_context(self, vision_analysis: str, camera_name: str = None):
        """Add vision analysis to persistent environment context"""
        timestamp = datetime.now().isoformat()
        context_entry = {
            "timestamp": timestamp,
            "camera_name": camera_name,
            "analysis": vision_analysis
        }
        
        self.environment_context.append(context_entry)
        
        # Keep only the most recent entries
        if len(self.environment_context) > self.max_context_entries:
            self.environment_context = self.environment_context[-self.max_context_entries:]
        
        logger.info(f"Added to environment context: {camera_name} - {len(vision_analysis)} chars")
    
    def _get_environment_context_summary(self) -> str:
        """Get a summary of the environment context for the AI"""
        if not self.environment_context:
            return "No previous environment context available."
        
        summary = "Previous environment observations:\n"
        for entry in self.environment_context[-3:]:  # Last 3 entries
            camera_info = f" ({entry['camera_name']})" if entry['camera_name'] else ""
            summary += f"- {entry['timestamp']}{camera_info}: {entry['analysis'][:200]}...\n"
        
        return summary
    
    def clear_environment_context(self):
        """Clear the environment context"""
        self.environment_context = []
        logger.info("Environment context cleared")

    def _sanitize_ai_camera_references(self, text: str, allowed_names: List[str]) -> str:
        """Return original text unless it references a camera name not in allowed list.

        If the message appears to reference a non-existent camera, replace the
        entire message with a single concise clarification prompt instead of
        doing token-by-token replacements (prevents spammy repeats).
        """
        try:
            if not text:
                return text
            # Do not touch JSON-like content (e.g., planning objects)
            trimmed = text.strip()
            if (trimmed.startswith('{') and trimmed.endswith('}')) or (trimmed.startswith('[') and trimmed.endswith(']')):
                return text

            whitelist = {str(n).strip().lower() for n in (allowed_names or []) if str(n).strip()}
            # If we don't have a whitelist, leave message unchanged
            if not whitelist:
                return text

            lower_text = text.lower()
            # If the assistant is asking the user to clarify which camera, do NOT override it.
            # The previous logic was too aggressive and replaced valid clarifying questions
            # with "Unknown camera name".
            if (
                lower_text.strip().endswith("?")
                or any(
                    phrase in lower_text
                    for phrase in [
                        "which camera",
                        "what camera",
                        "which cam",
                        "what cam",
                        "specify a camera",
                        "specify the camera",
                        "please specify",
                        "choose a camera",
                        "pick a camera",
                    ]
                )
            ):
                return text
            # If the message already includes any allowed camera name, keep it as-is
            if any(n in lower_text for n in whitelist):
                return text

            # If it mentions cameras in the context of switching/opening/showing a feed but
            # none of the allowed names appear, compress to a single notice.
            mentions_camera = ('camera' in lower_text) or ('cam' in lower_text)
            looks_like_feed_reference = any(
                k in lower_text
                for k in [
                    "opening",
                    "displaying",
                    "showing",
                    "switching",
                    "viewing",
                    "camera feed",
                    "camera stream",
                    "live feed",
                ]
            )
            if mentions_camera and looks_like_feed_reference:
                allowed_preview = ", ".join(sorted(whitelist))[:200]
                return f"Unknown camera name. Allowed: {allowed_preview}."

            return text
        except Exception:
            return text
    
    def get_intent_learner_stats(self) -> Dict[str, Any]:
        """Get intent learner statistics"""
        if not self.intent_learner:
            return {"total_patterns": 0, "total_executions": 0, "error": "Intent learner not available"}
        try:
            return self.intent_learner.get_stats()
        except Exception as e:
            logger.error(f"Failed to get intent learner stats: {e}")
            return {"error": str(e)}
    
    def provide_feedback(self, user_query: str, tool: str, rating: float):
        """
        Provide feedback on a tool execution
        
        Args:
            user_query: The user's original query
            tool: Tool that was executed
            rating: 0.0-1.0 (0=bad, 1=excellent)
        """
        if not self.intent_learner:
            logger.warning("Intent learner not available, feedback ignored")
            return
        
        try:
            # Find the pattern and update its score
            pattern = self.intent_learner.find_matching_pattern(user_query, threshold=0.6)
            if pattern and pattern.tool == tool:
                # Update existing pattern
                self.intent_learner.learn_from_execution(
                    user_query=user_query,
                    tool=tool,
                    parameters=pattern.parameters,
                    success=rating > 0.5,
                    feedback_score=rating
                )
                logger.info(f"Updated pattern feedback: {pattern.id} → {rating}")
            else:
                logger.warning(f"Could not find pattern to provide feedback for: {user_query}")
        except Exception as e:
            logger.error(f"Failed to provide feedback: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get AI agent status including vision service health"""
        local_vision_status = {
            "endpoint": getattr(self.provider, "vision_endpoint", None),
            "composition": getattr(self.provider, "vision_composition", None),
            "source": getattr(self.provider, "vision_source", None),
            "healthy": False,
            "models_loaded": {}
        }
        
        # Check local vision service health
        vision_endpoint = getattr(self.provider, "vision_endpoint", None) or os.environ.get("LOCAL_VISION_ENDPOINT", "http://127.0.0.1:8101")
        try:
            # Short timeout to prevent blocking UI (critical for terminal responsiveness)
            health_response = requests.get(f"{vision_endpoint}/health", timeout=0.6)
            if health_response.status_code == 200:
                health_data = health_response.json()
                local_vision_status["healthy"] = health_data.get("status") == "ok"
                local_vision_status["models_loaded"] = health_data.get("models_loaded", {})
        except Exception:
            local_vision_status["healthy"] = False
        
        return {
            "provider_available": self.provider is not None,
            "provider_type": type(self.provider).__name__ if self.provider else None,
            "chat_model": self.chat_model,
            "vision_model": self.vision_model,
            "provider_chain": [
                {"id": entry.get("id"), "source": entry.get("source"), "type": type(entry.get("provider")).__name__}
                for entry in self.provider_chain
            ],
            "local_vision": local_vision_status,
            "rate_limit": {
                "requests": self.request_count,
                "limit": self.rate_limit_requests,
                "window": self.rate_limit_window
            },
            "environment_context_entries": len(self.environment_context),
            "natural_language_rules_count": len(self.natural_language_rules),
            "zone_monitoring_count": len(self.zone_monitoring),
            "context_memory_entries": len(self.context_memory)
        }
    
    def add_natural_language_rule(self, zone_id: str, rule_description: str, camera_id: str = None) -> str:
        """Add a natural language rule for a zone that the AI will interpret and execute."""
        rule_id = f"nl_rule_{len(self.natural_language_rules) + 1}"
        
        rule = {
            "id": rule_id,
            "zone_id": zone_id,
            "camera_id": camera_id,
            "description": rule_description,
            "created_at": datetime.now().isoformat(),
            "enabled": True,
            "trigger_count": 0,
            "last_triggered": None
        }
        
        self.natural_language_rules[rule_id] = rule
        logger.info(f"Added natural language rule: {rule_id} - {rule_description}")
        
        return rule_id
    
    def interpret_zone_rules(self, zone_id: str, detected_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Interpret natural language rules for a zone based on detected objects."""
        triggered_actions = []
        
        for rule_id, rule in self.natural_language_rules.items():
            if rule["zone_id"] == zone_id and rule["enabled"]:
                # Check if any detected objects match the rule description
                if self._evaluate_rule_condition(rule["description"], detected_objects):
                    action = self._generate_rule_action(rule, detected_objects)
                    if action:
                        triggered_actions.append(action)
                        rule["trigger_count"] += 1
                        rule["last_triggered"] = datetime.now().isoformat()
        
        return triggered_actions
    
    def _evaluate_rule_condition(self, rule_description: str, detected_objects: List[Dict[str, Any]]) -> bool:
        """Evaluate if detected objects match the natural language rule condition."""
        description_lower = rule_description.lower()
        
        # Simple keyword-based evaluation (can be enhanced with NLP)
        if "person" in description_lower or "people" in description_lower:
            return any(obj["class_name"] == "person" for obj in detected_objects)
        
        if "car" in description_lower or "vehicle" in description_lower:
            return any(obj["class_name"] in ["car", "truck", "bus", "motorcycle"] for obj in detected_objects)
        
        if "motion" in description_lower or "movement" in description_lower:
            return len(detected_objects) > 0
        
        if "intrusion" in description_lower or "unauthorized" in description_lower:
            return any(obj["class_name"] == "person" for obj in detected_objects)
        
        # Default: trigger if any objects detected
        return len(detected_objects) > 0
    
    def _generate_rule_action(self, rule: Dict[str, Any], detected_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate an action based on the natural language rule and detected objects."""
        description_lower = rule["description"].lower()
        
        action = {
            "rule_id": rule["id"],
            "zone_id": rule["zone_id"],
            "camera_id": rule["camera_id"],
            "triggered_at": datetime.now().isoformat(),
            "detected_objects": detected_objects
        }
        
        # Parse natural language actions
        if "record" in description_lower or "capture" in description_lower:
            action["type"] = "record"
            action["parameters"] = {"duration": 30, "camera_ids": [rule["camera_id"]] if rule["camera_id"] else []}
        
        elif "notify" in description_lower or "alert" in description_lower:
            action["type"] = "notify"
            action["parameters"] = {
                "message": f"Rule triggered: {rule['description']}",
                "push": True,
                "email": False
            }
        
        elif "open" in description_lower and "camera" in description_lower:
            action["type"] = "open_camera_widget"
            action["parameters"] = {"camera_id": rule["camera_id"]}
        
        elif "log" in description_lower:
            action["type"] = "log_event"
            action["parameters"] = {
                "event_type": "zone_rule_triggered",
                "message": f"Natural language rule triggered: {rule['description']}",
                "objects": [obj["class_name"] for obj in detected_objects]
            }
        
        else:
            # Default action: log the event
            action["type"] = "log_event"
            action["parameters"] = {
                "event_type": "zone_rule_triggered",
                "message": f"Rule triggered: {rule['description']}",
                "objects": [obj["class_name"] for obj in detected_objects]
            }
        
        return action
    
    def detect_objects_in_image(self, image_base64: str, model_type: str = "mobilenet") -> List[Dict[str, Any]]:
        """Detect objects in a base64 encoded image using the specified model."""
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image")
                return []
            
            # Get the appropriate detection tool
            if model_type not in self.object_detection_tools:
                logger.warning(f"Model type {model_type} not available, using mobilenet")
                model_type = "mobilenet"
            
            detection_tool = self.object_detection_tools[model_type]
            detections = detection_tool.detect_objects(image)
            
            logger.info(f"Detected {len(detections)} objects using {model_type}")
            return detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []

    def detect_objects_in_image_with_config(self, image_base64: str, model_type: str = "yolov8",
                                            target_objects: List[str] = None, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Configurable wrapper for object detection with target classes and confidence filter."""
        try:
            detections = self.detect_objects_in_image(image_base64, model_type)
            if not detections:
                return []
            filtered = []
            for obj in detections:
                ok_conf = obj.get('confidence', 0) >= confidence_threshold
                ok_class = True
                if target_objects and len(target_objects) > 0:
                    name = obj.get('class_name') or obj.get('label') or obj.get('name')
                    ok_class = (name in target_objects)
                if ok_conf and ok_class:
                    filtered.append(obj)
            return filtered
        except Exception as e:
            logger.error(f"Error in configurable object detection: {e}")
            return []

    def detect_motion(self, current_frame_base64: str, previous_frame_base64: str = None,
                      threshold: float = 0.02, min_area: int = 300) -> Dict[str, Any]:
        """Detect motion between two frames using simple frame differencing."""
        try:
            # Decode current frame
            current_data = base64.b64decode(current_frame_base64)
            current_array = np.frombuffer(current_data, dtype=np.uint8)
            current_frame = cv2.imdecode(current_array, cv2.IMREAD_GRAYSCALE)
            
            if current_frame is None:
                return {"motion_detected": False, "confidence": 0.0, "motion_type": "none"}
            
            # If no previous frame, store current frame and return no motion
            if previous_frame_base64 is None:
                self.previous_frames = current_frame
                return {"motion_detected": False, "confidence": 0.0, "motion_type": "none"}
            
            # Decode previous frame
            prev_data = base64.b64decode(previous_frame_base64)
            prev_array = np.frombuffer(prev_data, dtype=np.uint8)
            previous_frame = cv2.imdecode(prev_array, cv2.IMREAD_GRAYSCALE)
            
            if previous_frame is None:
                return {"motion_detected": False, "confidence": 0.0, "motion_type": "none"}
            
            # Ensure frames are the same size
            if current_frame.shape != previous_frame.shape:
                previous_frame = cv2.resize(previous_frame, (current_frame.shape[1], current_frame.shape[0]))
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(current_frame, previous_frame)
            
            # Apply Gaussian blur to reduce noise (smaller kernel for faster processing)
            blurred = cv2.GaussianBlur(frame_diff, (15, 15), 0)
            
            # Apply threshold (lower threshold for more sensitivity)
            _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            significant_motion = False
            total_motion_area = 0
            motion_contours = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    significant_motion = True
                    total_motion_area += area
                    motion_contours.append(contour)
            
            if not significant_motion:
                return {"motion_detected": False, "confidence": 0.0, "motion_type": "none"}
            
            # Calculate motion confidence based on area ratio
            total_pixels = current_frame.shape[0] * current_frame.shape[1]
            motion_ratio = total_motion_area / total_pixels
            confidence = min(motion_ratio / threshold, 1.0)
            
            # Analyze motion type based on contour characteristics
            motion_type = self._analyze_motion_type(motion_contours, current_frame.shape)
            
            return {
                "motion_detected": True,
                "confidence": confidence,
                "motion_type": motion_type,
                "motion_area": total_motion_area,
                "motion_ratio": motion_ratio,
                "contour_count": len(motion_contours)
            }
            
        except Exception as e:
            logger.error(f"Error in motion detection: {e}")
            return {"motion_detected": False, "confidence": 0.0, "motion_type": "none"}

    def _analyze_motion_type(self, contours: List, frame_shape: tuple) -> str:
        """Analyze the type of motion based on contour characteristics."""
        if not contours:
            return "unknown"
        
        # Calculate bounding boxes for all contours
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
        
        # Analyze motion patterns
        total_width = sum(w for _, _, w, _ in bounding_boxes)
        total_height = sum(h for _, _, _, h in bounding_boxes)
        avg_width = total_width / len(bounding_boxes)
        avg_height = total_height / len(bounding_boxes)
        
        # Check if motion looks like a vehicle (horizontal movement, appropriate aspect ratio)
        if avg_width > avg_height * 1.5:  # Wide objects suggest vehicles
            return "vehicle_like"
        elif avg_height > avg_width * 1.5:  # Tall objects suggest people
            return "person_like"
        else:
            return "general_motion"

    def _analyze_motion_type_advanced(self, contours: List, frame_shape: tuple, 
                                    camera_id: str = None, current_time: float = None) -> Dict[str, Any]:
        """Advanced motion type analysis considering speed, trajectory, and consistency. Returns motion type and direction."""
        if not contours:
            return {"motion_type": "unknown", "direction": None, "confidence": 0.0}
        
        # Initialize motion history if not exists
        if not hasattr(self, 'motion_history'):
            self.motion_history = {}
        
        if camera_id not in self.motion_history:
            self.motion_history[camera_id] = []
        
        # Calculate current motion characteristics
        current_motions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                current_motions.append({
                    'x': cx, 'y': cy, 'w': w, 'h': h, 'area': area,
                    'aspect_ratio': w / h if h > 0 else 1,
                    'timestamp': current_time or time.time()
                })
        
        # Add current motions to history
        if current_motions:
            self.motion_history[camera_id].extend(current_motions)
        
        # Keep only recent motion data (last 2 seconds)
        cutoff_time = (current_time or time.time()) - 2.0
        self.motion_history[camera_id] = [
            m for m in self.motion_history[camera_id] 
            if m['timestamp'] > cutoff_time
        ]
        
        # Need at least 3 data points for trajectory analysis
        if len(self.motion_history[camera_id]) < 3:
            return {"motion_type": "general_motion", "direction": None, "confidence": 0.3}
        
        # Analyze motion patterns
        recent_motions = self.motion_history[camera_id][-10:]  # Last 10 data points
        
        # Calculate average size and consistency
        avg_width = sum(m['w'] for m in recent_motions) / len(recent_motions)
        avg_height = sum(m['h'] for m in recent_motions) / len(recent_motions)
        avg_area = sum(m['area'] for m in recent_motions) / len(recent_motions)
        
        # Calculate size consistency (coefficient of variation)
        area_variance = sum((m['area'] - avg_area) ** 2 for m in recent_motions) / len(recent_motions)
        area_std = area_variance ** 0.5
        size_consistency = 1.0 - (area_std / avg_area) if avg_area > 0 else 0.0
        
        # Calculate speed and trajectory
        direction = None
        confidence = 0.5  # Base confidence
        
        if len(recent_motions) >= 2:
            # Calculate displacement over time
            first_motion = recent_motions[0]
            last_motion = recent_motions[-1]
            time_diff = last_motion['timestamp'] - first_motion['timestamp']
            
            if time_diff > 0:
                # Calculate total distance moved
                dx = last_motion['x'] - first_motion['x']
                dy = last_motion['y'] - first_motion['y']
                total_distance = (dx**2 + dy**2)**0.5
                
                # Calculate direction vector (normalized)
                if total_distance > 0:
                    direction = {
                        'dx': dx / total_distance,
                        'dy': dy / total_distance,
                        'angle': math.atan2(dy, dx) * 180 / math.pi  # Angle in degrees
                    }
                
                # Calculate speed (pixels per second)
                speed = total_distance / time_diff
                
                # Calculate trajectory straightness
                # For a straight line, the total distance should be close to the sum of individual steps
                step_distances = []
                for i in range(1, len(recent_motions)):
                    prev = recent_motions[i-1]
                    curr = recent_motions[i]
                    step_dx = curr['x'] - prev['x']
                    step_dy = curr['y'] - prev['y']
                    step_distance = (step_dx**2 + step_dy**2)**0.5
                    step_distances.append(step_distance)
                
                if step_distances:
                    total_step_distance = sum(step_distances)
                    trajectory_straightness = total_distance / total_step_distance if total_step_distance > 0 else 0.0
                else:
                    trajectory_straightness = 0.0
                
                # Classification logic based on user requirements
                
                # Person detection: constant person-sized object moving slowly
                person_size_range = (50, 200)  # Typical person bounding box size range
                person_speed_threshold = 50  # pixels per second
                
                if (person_size_range[0] <= avg_width <= person_size_range[1] and 
                    person_size_range[0] <= avg_height <= person_size_range[1] and
                    speed < person_speed_threshold and
                    size_consistency > 0.7):  # High size consistency
                    confidence = min(0.8 + size_consistency * 0.2, 1.0)
                    return {"motion_type": "person_like", "direction": direction, "confidence": confidence}
                
                # Vehicle detection: object moving fast in straight line for multiple frames
                vehicle_speed_threshold = 100  # pixels per second
                vehicle_straightness_threshold = 0.8  # How straight the trajectory should be
                
                if (speed > vehicle_speed_threshold and
                    trajectory_straightness > vehicle_straightness_threshold and
                    len(recent_motions) >= 5):  # Multiple frames of consistent motion
                    confidence = min(0.7 + trajectory_straightness * 0.3, 1.0)
                    return {"motion_type": "vehicle_like", "direction": direction, "confidence": confidence}
                
                # General motion with confidence based on consistency
                confidence = min(0.4 + size_consistency * 0.3 + trajectory_straightness * 0.3, 1.0)
        
        # Default to general motion if no specific patterns match
        return {"motion_type": "general_motion", "direction": direction, "confidence": confidence}

    def detect_motion_with_tracking(self, current_frame_base64: str, previous_frame_base64: str = None,
                                  camera_id: str = None, threshold: float = 0.02, min_area: int = 300) -> Dict[str, Any]:
        """Advanced motion detection with object tracking and persistent tracking points."""
        try:
            # Get current time for tracking (moved to top to fix the error)
            current_time = time.time()
            
            # Initialize tracking storage if not exists
            if not hasattr(self, 'tracking_points'):
                self.tracking_points = {}
            if not hasattr(self, 'tracking_history'):
                self.tracking_history = {}
            
            # Decode current frame
            current_data = base64.b64decode(current_frame_base64)
            current_array = np.frombuffer(current_data, dtype=np.uint8)
            current_frame = cv2.imdecode(current_array, cv2.IMREAD_COLOR)
            
            if current_frame is None:
                return {"motion_detected": False, "tracking_points": [], "motion_type": "none"}
            
            # Convert to grayscale for motion detection
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # If no previous frame, store current frame and return no motion
            if previous_frame_base64 is None:
                self.previous_frames = current_gray
                return {"motion_detected": False, "tracking_points": [], "motion_type": "none"}
            
            # Decode previous frame
            prev_data = base64.b64decode(previous_frame_base64)
            prev_array = np.frombuffer(prev_data, dtype=np.uint8)
            previous_frame = cv2.imdecode(prev_array, cv2.IMREAD_GRAYSCALE)
            
            if previous_frame is None:
                return {"motion_detected": False, "tracking_points": [], "motion_type": "none"}
            
            # Ensure frames are the same size
            if current_gray.shape != previous_frame.shape:
                previous_frame = cv2.resize(previous_frame, (current_gray.shape[1], current_gray.shape[0]))
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(current_gray, previous_frame)
            
            # Apply Gaussian blur to reduce noise (smaller kernel for faster processing)
            blurred = cv2.GaussianBlur(frame_diff, (15, 15), 0)
            
            # Apply threshold (lower threshold for more sensitivity)
            _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and get tracking points
            significant_motion = False
            total_motion_area = 0
            motion_contours = []
            tracking_points = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    significant_motion = True
                    total_motion_area += area
                    motion_contours.append(contour)
                    
                    # Calculate centroid for tracking point
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Analyze motion type for this contour using advanced analysis
                        x, y, w, h = cv2.boundingRect(contour)
                        # Use the advanced motion analysis for overall motion type
                        motion_analysis = self._analyze_motion_type_advanced(motion_contours, current_frame.shape, camera_id, current_time)
                        motion_type = motion_analysis["motion_type"]
                        direction = motion_analysis["direction"]
                        motion_confidence = motion_analysis["confidence"]
                        
                        # Calculate area-based confidence
                        area_confidence = min(area / (current_frame.shape[0] * current_frame.shape[1]) / threshold, 1.0)
                        
                        # Combine motion confidence with area confidence
                        combined_confidence = (motion_confidence * 0.7 + area_confidence * 0.3)
                        
                        # Create tracking point
                        tracking_point = {
                            "id": f"track_{len(tracking_points)}",
                            "x": cx / current_frame.shape[1],  # Normalized coordinates
                            "y": cy / current_frame.shape[0],
                            "motion_type": motion_type,
                            "confidence": combined_confidence,
                            "area": area,
                            "bbox": [x / current_frame.shape[1], y / current_frame.shape[0], 
                                   w / current_frame.shape[1], h / current_frame.shape[0]],
                            "direction": direction
                        }
                        tracking_points.append(tracking_point)
            
            if not significant_motion:
                return {"motion_detected": False, "tracking_points": [], "motion_type": "none"}
            
            # Update tracking history with improved management
            if camera_id:
                if camera_id not in self.tracking_history:
                    self.tracking_history[camera_id] = []
                
                # Add current tracking points to history with unique IDs
                for i, point in enumerate(tracking_points):
                    point["timestamp"] = current_time
                    point["id"] = f"track_{camera_id}_{current_time}_{i}"
                    self.tracking_history[camera_id].append(point)
                
                # Keep only recent tracking points (last 15 seconds) and limit to 20 points max
                cutoff_time = current_time - 15
                self.tracking_history[camera_id] = [
                    p for p in self.tracking_history[camera_id] 
                    if p["timestamp"] > cutoff_time
                ][-20:]  # Keep only last 20 points
            
            # Calculate overall motion confidence
            total_pixels = current_gray.shape[0] * current_gray.shape[1]
            motion_ratio = total_motion_area / total_pixels
            confidence = min(motion_ratio / threshold, 1.0)
            
            # Determine overall motion type using advanced analysis
            overall_motion_analysis = self._analyze_motion_type_advanced(motion_contours, current_frame.shape, camera_id, current_time)
            overall_motion_type = overall_motion_analysis["motion_type"]
            
            return {
                "motion_detected": True,
                "confidence": confidence,
                "motion_type": overall_motion_type,
                "motion_area": total_motion_area,
                "motion_ratio": motion_ratio,
                "contour_count": len(motion_contours),
                "tracking_points": tracking_points,
                "tracking_history": self.tracking_history.get(camera_id, []) if camera_id else []
            }
            
        except Exception as e:
            logger.error(f"Error in motion detection with tracking: {e}")
            return {"motion_detected": False, "tracking_points": [], "motion_type": "none"}

    def detect_cars_efficiently(self, image_base64: str, camera_id: str, 
                              previous_frame_base64: str = None) -> Dict[str, Any]:
        """Efficient car detection using motion detection first, then object detection."""
        try:
            # Step 1: Advanced motion detection with tracking
            motion_result = self.detect_motion_with_tracking(image_base64, previous_frame_base64, camera_id)
            
            if not motion_result["motion_detected"]:
                return {
                    "cars_detected": False,
                    "motion_detected": False,
                    "detection_method": "none",
                    "message": "No motion detected",
                    "tracking_points": [],
                    "tracking_history": []
                }
            
            # Step 2: Check if motion looks like a vehicle
            if motion_result["motion_type"] != "vehicle_like":
                return {
                    "cars_detected": False,
                    "motion_detected": True,
                    "motion_type": motion_result["motion_type"],
                    "detection_method": "motion_only",
                    "message": f"Motion detected but not vehicle-like: {motion_result['motion_type']}",
                    "tracking_points": motion_result.get("tracking_points", []),
                    "tracking_history": motion_result.get("tracking_history", [])
                }
            
            # Step 3: Use object detection to confirm cars
            detections = self.detect_objects_in_image(image_base64, "yolov8")
            
            # Filter for car-related objects
            car_objects = []
            for detection in detections:
                class_name = detection.get("class_name", "").lower()
                if any(car_type in class_name for car_type in ["car", "truck", "bus", "motorcycle", "vehicle"]):
                    car_objects.append(detection)
            
            if not car_objects:
                return {
                    "cars_detected": False,
                    "motion_detected": True,
                    "motion_type": motion_result["motion_type"],
                    "detection_method": "motion_and_detection",
                    "message": "Vehicle-like motion detected but no cars confirmed by object detection",
                    "tracking_points": motion_result.get("tracking_points", []),
                    "tracking_history": motion_result.get("tracking_history", [])
                }
            
            # Step 4: Log the detection and create thumbnail
            detection_log = self._log_car_detection(camera_id, car_objects, image_base64)
            
            # Step 5: Prepare chat message
            chat_message = self._prepare_car_detection_message(camera_id, car_objects, detection_log)
            
            return {
                "cars_detected": True,
                "motion_detected": True,
                "motion_type": motion_result["motion_type"],
                "detection_method": "motion_and_detection",
                "car_count": len(car_objects),
                "car_objects": car_objects,
                "detection_log": detection_log,
                "chat_message": chat_message,
                "message": f"Detected {len(car_objects)} car(s) on camera {camera_id}",
                "tracking_points": motion_result.get("tracking_points", []),
                "tracking_history": motion_result.get("tracking_history", [])
            }
            
        except Exception as e:
            logger.error(f"Error in efficient car detection: {e}")
            return {
                "cars_detected": False,
                "error": str(e),
                "message": f"Error in car detection: {e}"
            }

    def _log_car_detection(self, camera_id: str, car_objects: List[Dict], image_base64: str) -> Dict[str, Any]:
        """Log car detection with thumbnail and metadata."""
        try:
            # Create detection log entry
            detection_id = f"car_detection_{int(time.time())}_{hash(image_base64[:100]) % 10000}"
            
            # Create thumbnail (resize image to smaller size)
            image_data = base64.b64decode(image_base64)
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            # Resize to thumbnail size
            thumbnail = cv2.resize(image, (320, 240))
            _, thumbnail_encoded = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 70])
            thumbnail_base64 = base64.b64encode(thumbnail_encoded).decode('utf-8')
            
            # Create log entry
            log_entry = {
                "detection_id": detection_id,
                "timestamp": datetime.now().isoformat(),
                "camera_id": camera_id,
                "car_count": len(car_objects),
                "car_details": car_objects,
                "thumbnail": thumbnail_base64,
                "confidence_scores": [obj.get("confidence", 0) for obj in car_objects]
            }
            
            # Store in detection history
            if not hasattr(self, 'car_detection_history'):
                self.car_detection_history = []
            
            self.car_detection_history.append(log_entry)
            
            # Keep only recent detections (last 100)
            if len(self.car_detection_history) > 100:
                self.car_detection_history = self.car_detection_history[-100:]
            
            logger.info(f"Logged car detection: {detection_id} on camera {camera_id}")
            return log_entry
            
        except Exception as e:
            logger.error(f"Error logging car detection: {e}")
            return {"error": str(e)}

    def _prepare_car_detection_message(self, camera_id: str, car_objects: List[Dict], detection_log: Dict) -> str:
        """Prepare a chat message for car detection."""
        try:
            car_count = len(car_objects)
            car_types = []
            
            for obj in car_objects:
                class_name = obj.get("class_name", "vehicle")
                confidence = obj.get("confidence", 0)
                car_types.append(f"{class_name} ({confidence:.1%})")
            
            car_types_str = ", ".join(car_types)
            
            message = f"🚗 **Car Detection Alert**\n"
            message += f"**Camera:** {camera_id}\n"
            message += f"**Time:** {datetime.now().strftime('%H:%M:%S')}\n"
            message += f"**Detected:** {car_count} vehicle(s)\n"
            message += f"**Types:** {car_types_str}\n"
            message += f"**Detection ID:** {detection_log.get('detection_id', 'N/A')}"
            
            return message
            
        except Exception as e:
            logger.error(f"Error preparing car detection message: {e}")
            return f"Car detected on camera {camera_id} at {datetime.now().strftime('%H:%M:%S')}"

    def get_car_detection_history(self, camera_id: str = None, limit: int = 20) -> List[Dict]:
        """Get car detection history, optionally filtered by camera."""
        if not hasattr(self, 'car_detection_history'):
            return []
        
        history = self.car_detection_history
        
        if camera_id:
            history = [entry for entry in history if entry.get("camera_id") == camera_id]
        
        return history[-limit:] if limit else history
    
    def setup_zone_monitoring(self, zone_id: str, camera_id: str, enabled: bool = True) -> bool:
        """Setup monitoring for a specific zone."""
        self.zone_monitoring[zone_id] = {
            "camera_id": camera_id,
            "enabled": enabled,
            "last_detection": None,
            "detection_count": 0
        }
        logger.info(f"Zone monitoring setup: {zone_id} on camera {camera_id}")
        return True
    
    def recall_context(self, query: str) -> List[Dict[str, Any]]:
        """Recall relevant context from memory based on a query."""
        relevant_context = []
        query_lower = query.lower()
        
        # Search through environment context
        for entry in self.environment_context:
            if any(keyword in entry["analysis"].lower() for keyword in query_lower.split()):
                relevant_context.append({
                    "type": "environment_context",
                    "timestamp": entry["timestamp"],
                    "camera_name": entry["camera_name"],
                    "content": entry["analysis"]
                })
        
        # Search through context memory
        for memory_entry in self.context_memory:
            if any(keyword in memory_entry["content"].lower() for keyword in query_lower.split()):
                relevant_context.append({
                    "type": "context_memory",
                    "timestamp": memory_entry["timestamp"],
                    "content": memory_entry["content"]
                })
        
        # Search through natural language rules
        for rule_id, rule in self.natural_language_rules.items():
            if any(keyword in rule["description"].lower() for keyword in query_lower.split()):
                relevant_context.append({
                    "type": "natural_language_rule",
                    "rule_id": rule_id,
                    "zone_id": rule["zone_id"],
                    "content": rule["description"]
                })
        
        return relevant_context
    
    def add_to_context_memory(self, content: str, context_type: str = "general"):
        """Add information to the context memory for future recall."""
        memory_entry = {
            "content": content,
            "type": context_type,
            "timestamp": datetime.now().isoformat()
        }
        
        self.context_memory.append(memory_entry)
        
        # Keep only the most recent entries
        if len(self.context_memory) > 100:
            self.context_memory = self.context_memory[-100:]
        
        logger.info(f"Added to context memory: {context_type} - {len(content)} chars")
    
    def get_natural_language_rules(self) -> Dict[str, Any]:
        """Get all natural language rules."""
        return self.natural_language_rules
    
    def update_natural_language_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update a natural language rule."""
        if rule_id not in self.natural_language_rules:
            return False
        
        self.natural_language_rules[rule_id].update(updates)
        logger.info(f"Updated natural language rule: {rule_id}")
        return True
    
    def delete_natural_language_rule(self, rule_id: str) -> bool:
        """Delete a natural language rule."""
        if rule_id not in self.natural_language_rules:
            return False
        
        del self.natural_language_rules[rule_id]
        logger.info(f"Deleted natural language rule: {rule_id}")
        return True

    async def analyze_scene_for_focus_areas(self, image_base64: str, camera_id: str) -> Dict[str, Any]:
        """Analyze a scene using LLM to intelligently create focus areas for detection."""
        try:
            if not self.provider:
                return {"success": False, "message": "AI provider not available"}
            
            # Create a context for the AI
            context = AIContext(
                devices=[{
                    "id": camera_id,
                    "name": f"Camera {camera_id}",
                    "type": "camera",
                    "screenshot": image_base64,
                    "is_visible_on_dashboard": True
                }],
                connections=[],
                layout=[]
            )
            
            # Create a prompt for scene analysis
            prompt = f"""Analyze this camera scene and create intelligent detection zones and laser lines for security monitoring.

SCENE ANALYSIS TASK:
1. Look at the camera feed and identify key areas that need monitoring
2. Create detection zones for areas where people or vehicles might appear
3. Create laser lines for paths that people or vehicles might cross
4. Focus on areas where distant objects (like cars on roads) might be detected

DETECTION ZONES TO CREATE:
- Road areas where vehicles drive by (especially distant roads)
- Entry/exit points where people or vehicles enter/leave
- Parking areas where vehicles might be parked
- Walkways where people might walk
- Any other areas of security interest

LASER LINES TO CREATE:
- Crosswalks or paths that people cross
- Driveways or roads that vehicles cross
- Entry/exit lines where movement is detected
- Any linear paths that need monitoring

RESPONSE FORMAT:
Return a JSON response with this exact structure:
{{
  "zones": [
    {{
      "label": "Zone name",
      "description": "What this zone monitors",
      "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "color": "#24D1FF",
      "type": "vehicle|person|general"
    }}
  ],
  "lines": [
    {{
      "label": "Line name", 
      "description": "What this line monitors",
      "start": [x1, y1],
      "end": [x2, y2],
      "type": "vehicle|person|general"
    }}
  ],
  "analysis": "Brief description of what you see and why you created these zones/lines"
}}

IMPORTANT:
- Use normalized coordinates (0.0 to 1.0) for all positions
- Create zones as rectangles with 4 points
- Create lines with start and end points
- Focus on areas where distant objects might be detected
- Be specific about what each zone/line monitors

Camera ID: {camera_id}"""

            # Call the AI
            response = await self.chat(prompt, context, image_base64)
            
            if response.error:
                return {"success": False, "message": f"AI analysis failed: {response.error}"}
            
            # Parse the AI response to extract zones and lines
            try:
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*?"zones".*?"lines".*?\}', response.message, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    
                    return {
                        "success": True,
                        "zones": data.get("zones", []),
                        "lines": data.get("lines", []),
                        "analysis": data.get("analysis", "Scene analyzed successfully"),
                        "message": f"Created {len(data.get('zones', []))} zones and {len(data.get('lines', []))} lines"
                    }
                else:
                    # Fallback: create default zones based on image analysis
                    return self._create_default_focus_areas(image_base64, camera_id)
                    
            except Exception as e:
                logger.error(f"Error parsing AI response: {e}")
                # Fallback to default zones
                return self._create_default_focus_areas(image_base64, camera_id)
                
        except Exception as e:
            logger.error(f"Error in AI scene analysis: {e}")
            return {"success": False, "message": f"Scene analysis error: {str(e)}"}

    def _create_default_focus_areas(self, image_base64: str, camera_id: str) -> Dict[str, Any]:
        """Create default focus areas when AI analysis fails."""
        try:
            # Decode image
            image_data = base64.b64decode(image_base64)
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"success": False, "message": "Failed to decode image"}
            
            height, width = image.shape[:2]
            
            # Create default zones
            zones = [
                {
                    "label": "Road Detection Zone",
                    "description": "Monitors road area for vehicle detection",
                    "points": [[0.1, 0.6], [0.9, 0.6], [0.9, 0.9], [0.1, 0.9]],
                    "color": "#24D1FF",
                    "type": "vehicle"
                },
                {
                    "label": "General Motion Zone",
                    "description": "Monitors general area for motion detection",
                    "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                    "color": "#7C4DFF",
                    "type": "general"
                }
            ]
            
            # Create default lines
            lines = [
                {
                    "label": "Entry/Exit Line",
                    "description": "Monitors entry and exit movement",
                    "start": [0.3, 0.5],
                    "end": [0.7, 0.5],
                    "type": "general"
                }
            ]
            
            return {
                "success": True,
                "zones": zones,
                "lines": lines,
                "analysis": "Created default detection zones and lines",
                "message": f"Created {len(zones)} zones and {len(lines)} lines"
            }
            
        except Exception as e:
            logger.error(f"Error creating default focus areas: {e}")
            return {"success": False, "message": f"Default focus areas error: {str(e)}"}

    def check_user_detection_zones(self, camera_id: str, detected_objects: List[Dict[str, Any]], shapes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if detected objects interact with user-defined zones, lines, or tags."""
        try:
            events = []
            
            for obj in detected_objects:
                obj_bbox = obj.get('bbox', [])
                if len(obj_bbox) != 4:
                    continue
                    
                obj_x, obj_y, obj_w, obj_h = obj_bbox
                obj_center_x = obj_x + obj_w / 2
                obj_center_y = obj_y + obj_h / 2
                
                for shape in shapes:
                    if shape.get('cameraId') != camera_id:
                        continue
                        
                    shape_id = shape.get('id')
                    shape_type = shape.get('kind')
                    shape_label = shape.get('label', 'Unknown')
                    
                    interaction = False
                    interaction_type = ""
                    
                    if shape_type == 'zone':
                        # Check if object center is inside zone
                        if self._point_in_polygon(obj_center_x, obj_center_y, shape.get('pts', [])):
                            interaction = True
                            interaction_type = "entered_zone"
                            
                    elif shape_type == 'line':
                        # Check if object is near laser line
                        p1 = shape.get('p1', {})
                        p2 = shape.get('p2', {})
                        if self._point_near_line(obj_center_x, obj_center_y, p1, p2, threshold=0.05):
                            interaction = True
                            interaction_type = "crossed_line"
                            
                    elif shape_type == 'tag':
                        # Check if object is near tag
                        anchor = shape.get('anchor', {})
                        distance = math.sqrt((obj_center_x - anchor.get('x', 0))**2 + (obj_center_y - anchor.get('y', 0))**2)
                        if distance < 0.1:  # Within 10% of frame
                            interaction = True
                            interaction_type = "near_tag"
                    
                    if interaction:
                        event = {
                            'shape_id': shape_id,
                            'shape_type': shape_type,
                            'shape_label': shape_label,
                            'object_class': obj.get('class_name', 'unknown'),
                            'interaction_type': interaction_type,
                            'camera_id': camera_id,
                            'chat_message': f"🚨 **{shape_label} Alert**: {obj.get('class_name', 'object')} {interaction_type.replace('_', ' ')} at {datetime.now().strftime('%H:%M:%S')}",
                            'timestamp': datetime.now().isoformat()
                        }
                        events.append(event)
            
            return {
                'success': True,
                'data': {
                    'events': events,
                    'total_interactions': len(events)
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking user detection zones: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def handle_car_counting_request(self, camera_id: str, user_message: str = "", image_base64: str = None) -> Dict[str, Any]:
        """Handle car counting requests as an AI agent tool."""
        try:
            logger.info(f"Handling car counting request for camera {camera_id}")
            
            # Create a general counting rule for the camera
            rule_id = f"auto_count_{camera_id}_{int(time.time())}"
            rule_data = {
                'id': rule_id,
                'name': f"Auto Car Count - {camera_id}",
                'camera_id': camera_id,
                'zone_id': None,  # Will be set if zones are created
                'enabled': True,
                'direction': 'both',
                'object_types': ['car', 'truck', 'bus'],
                'confidence_threshold': 0.5,
                'created_at': datetime.now().isoformat(),
                'count_in': 0,
                'count_out': 0,
                'last_count_time': None,
                'auto_created': True,
                'user_message': user_message
            }
            
            # If we have an image, try to analyze the scene for better zone creation
            zones_created = 0
            if image_base64:
                try:
                    analysis_result = await self.analyze_scene_for_focus_areas(image_base64, camera_id)
                    if analysis_result.get('success'):
                        zones_created = len(analysis_result.get('zones', []))
                        logger.info(f"Created {zones_created} zones from scene analysis")
                except Exception as e:
                    logger.warning(f"Scene analysis failed, using default rule: {e}")
            else:
                # No image provided, just create the rule without scene analysis
                logger.info("No image provided for scene analysis, creating default car counting rule")
            
            # Generate AI response
            ai_response = f"I've set up car counting for camera {camera_id}! 🚗\n\n" \
                         f"**Auto Car Count Rule Created**\n" \
                         f"• Camera: {camera_id}\n" \
                         f"• Objects: cars, trucks, buses\n" \
                         f"• Direction: Both directions\n" \
                         f"• Status: Active ✅\n" \
                         f"• Zones Created: {zones_created}\n\n" \
                         f"The system will now automatically count vehicles in the camera view and report back to you in the chat."
            
            if user_message:
                ai_response += f"\n\n*Based on your request: \"{user_message}\"*"
            
            return {
                'success': True,
                'data': {
                    'rules_created': [rule_data],
                    'zones_created': zones_created,
                    'ai_response': ai_response,
                    'camera_id': camera_id
                }
            }
            
        except Exception as e:
            logger.error(f"Error handling car counting request: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _should_trigger_car_counting(self, message: str) -> bool:
        """Check if a message should trigger car counting setup."""
        message_lower = message.lower()
        car_counting_keywords = [
            'count cars', 'count vehicles', 'start counting', 'car counting',
            'vehicle counting', 'count traffic', 'monitor cars'
        ]
        return any(keyword in message_lower for keyword in car_counting_keywords)

    def _point_in_polygon(self, x: float, y: float, points: List[Dict[str, float]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        if len(points) < 3:
            return False
        
        inside = False
        j = len(points) - 1
        
        for i in range(len(points)):
            xi = points[i].get('x', 0)
            yi = points[i].get('y', 0)
            xj = points[j].get('x', 0)
            yj = points[j].get('y', 0)
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside

    def _point_near_line(self, x: float, y: float, p1: Dict[str, float], p2: Dict[str, float], threshold: float = 0.05) -> bool:
        """Check if a point is near a line segment."""
        x1, y1 = p1.get('x', 0), p1.get('y', 0)
        x2, y2 = p2.get('x', 0), p2.get('y', 0)
        
        # Calculate distance from point to line segment
        A = x - x1
        B = y - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            # Line is a point
            distance = math.sqrt(A * A + B * B)
        else:
            param = dot / len_sq
            if param < 0:
                xx, yy = x1, y1
            elif param > 1:
                xx, yy = x2, y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D
            
            distance = math.sqrt((x - xx) * (x - xx) + (y - yy) * (y - yy))
        
        return distance < threshold

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available object detection models."""
        return [
            {
                'id': 'yolov8',
                'name': 'YOLOv8',
                'type': 'object_detection',
                'description': 'Fast and accurate object detection model',
                'supported_analysis': ['object_detection', 'vehicle_detection', 'person_detection'],
                'status': 'available'
            },
            {
                'id': 'mobilenet',
                'name': 'MobileNet SSD',
                'type': 'object_detection',
                'description': 'Lightweight object detection model for mobile devices',
                'supported_analysis': ['object_detection', 'vehicle_detection', 'person_detection'],
                'status': 'available'
            },
            {
                'id': 'efficientdet',
                'name': 'EfficientDet',
                'type': 'object_detection',
                'description': 'Efficient object detection model with good accuracy',
                'supported_analysis': ['object_detection', 'vehicle_detection', 'person_detection'],
                'status': 'available'
            }
        ]

    def analyze_images(self, roi_images: List[Dict[str, Any]], prompt: str = "") -> List[Dict[str, Any]]:
        """
        Lightweight helper used by the stream server to attach textual labels to detected ROIs.
        For now, it passes through the provided class/confidence and echoes a concise description.

        roi_images: list of { bbox, class, confidence, image(base64) }
        returns: list of { bbox, analysis, confidence }
        """
        try:
            results: List[Dict[str, Any]] = []
            for item in roi_images:
                cls = str(item.get('class') or 'object')
                conf = float(item.get('confidence') or 0.0)
                bbox = item.get('bbox') or {}
                # Compose a short analysis string
                analysis_text = f"{cls}"
                results.append({
                    'bbox': bbox,
                    'analysis': analysis_text,
                    'confidence': conf
                })
            return results
        except Exception as e:
            logger.warning(f"analyze_images failed: {e}")
            return []

    def get_detection_data(self, query_type: str = "status", camera_id: str = None, 
                          object_type: str = None, limit: int = 50) -> Dict[str, Any]:
        """
        Get detection data for the AI agent to provide insights.
        
        Args:
            query_type: "status", "active_objects", "logs", "composite"
            camera_id: Optional camera filter
            object_type: Optional object type filter
            limit: Number of results to return
            
        Returns:
            Dictionary with detection data and analysis
        """
        try:
            if not hasattr(self, 'stream_server') or not self.stream_server:
                return {
                    'success': False,
                    'error': 'Stream server not available'
                }
            
            if query_type == "status":
                data = self.stream_server.get_detection_status()
                if data:
                    return {
                        'success': True,
                        'data': data,
                        'analysis': self._analyze_detection_status(data)
                    }
                    
            elif query_type == "active_objects":
                data = self.stream_server.get_active_objects()
                if data:
                    return {
                        'success': True,
                        'data': data,
                        'analysis': self._analyze_active_objects(data)
                    }
                    
            elif query_type == "logs":
                data = self.stream_server.get_detection_logs(
                    camera_id=camera_id, limit=limit, object_type=object_type
                )
                if data:
                    return {
                        'success': True,
                        'data': data,
                        'analysis': self._analyze_detection_logs(data)
                    }
                    
            elif query_type == "composite":
                if not camera_id:
                    return {
                        'success': False,
                        'error': 'camera_id required for composite query'
                    }
                # This would need track_id parameter
                return {
                    'success': False,
                    'error': 'composite query requires track_id parameter'
                }
            
            return {
                'success': False,
                'error': f'Unknown query type: {query_type}'
            }
            
        except Exception as e:
            logger.error(f"Error getting detection data: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _analyze_detection_status(self, status: Dict[str, Any]) -> str:
        """Analyze detection status and provide insights"""
        try:
            total_cameras = status.get('total_cameras', 0)
            active_tracks = status.get('total_active_tracks', 0)
            cameras_with_motion = status.get('cameras_with_motion', 0)
            ai_status = status.get('ai_analyzer_status', 'unknown')
            system_health = status.get('system_health', 'unknown')
            
            analysis = f"Detection System Status:\n"
            analysis += f"- {total_cameras} cameras total, {cameras_with_motion} with motion\n"
            analysis += f"- {active_tracks} objects currently being tracked\n"
            analysis += f"- AI Analyzer: {ai_status}\n"
            analysis += f"- System Health: {system_health}\n"
            
            if active_tracks > 0:
                analysis += f"\nThe system is actively tracking {active_tracks} objects across {cameras_with_motion} cameras."
            elif cameras_with_motion > 0:
                analysis += f"\nMotion detected on {cameras_with_motion} cameras but no objects are currently being tracked."
            else:
                analysis += f"\nNo motion detected across any cameras."
                
            return analysis
            
        except Exception as e:
            return f"Error analyzing status: {e}"

    def _analyze_active_objects(self, objects_data: Dict[str, Any]) -> str:
        """Analyze active objects and provide insights"""
        try:
            objects = objects_data.get('active_objects', [])
            total_count = objects_data.get('total_count', 0)
            cameras_with_objects = objects_data.get('cameras_with_objects', 0)
            
            if not objects:
                return "No objects are currently being tracked."
            
            analysis = f"Active Objects Analysis:\n"
            analysis += f"- {total_count} objects being tracked across {cameras_with_objects} cameras\n\n"
            
            # Group by object type
            object_types = {}
            camera_activity = {}
            
            for obj in objects:
                obj_type = obj.get('label', 'unknown')
                camera_id = obj.get('camera_id', 'unknown')
                confidence = obj.get('confidence', 0.0)
                age = obj.get('age', 0)
                
                if obj_type not in object_types:
                    object_types[obj_type] = []
                object_types[obj_type].append({
                    'camera': camera_id,
                    'confidence': confidence,
                    'age': age
                })
                
                if camera_id not in camera_activity:
                    camera_activity[camera_id] = 0
                camera_activity[camera_id] += 1
            
            # Object type summary
            analysis += "Object Types:\n"
            for obj_type, items in object_types.items():
                avg_confidence = sum(item['confidence'] for item in items) / len(items)
                avg_age = sum(item['age'] for item in items) / len(items)
                analysis += f"- {obj_type}: {len(items)} instances (avg confidence: {avg_confidence:.2f}, avg age: {avg_age:.1f} frames)\n"
            
            # Camera activity
            analysis += f"\nCamera Activity:\n"
            for camera_id, count in sorted(camera_activity.items(), key=lambda x: x[1], reverse=True):
                analysis += f"- {camera_id}: {count} objects\n"
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing active objects: {e}"

    def _analyze_detection_logs(self, logs_data: Dict[str, Any]) -> str:
        """Analyze detection logs and provide insights"""
        try:
            logs = logs_data.get('logs', [])
            total_count = logs_data.get('total_count', 0)
            
            if not logs:
                return "No recent detection logs available."
            
            analysis = f"Recent Detection Analysis:\n"
            analysis += f"- {total_count} detection events logged\n\n"
            
            # Group by object type and time
            object_types = {}
            recent_activity = {}
            
            for log in logs:
                obj_type = log.get('object_type', 'unknown')
                camera_id = log.get('camera_id', 'unknown')
                confidence = log.get('confidence', 0.0)
                timestamp = log.get('timestamp', '')
                
                if obj_type not in object_types:
                    object_types[obj_type] = []
                object_types[obj_type].append({
                    'camera': camera_id,
                    'confidence': confidence,
                    'timestamp': timestamp
                })
                
                # Group by hour for recent activity
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour_key = dt.strftime('%Y-%m-%d %H:00')
                    if hour_key not in recent_activity:
                        recent_activity[hour_key] = 0
                    recent_activity[hour_key] += 1
                except:
                    pass
            
            # Object type summary
            analysis += "Detection Summary by Type:\n"
            for obj_type, items in object_types.items():
                avg_confidence = sum(item['confidence'] for item in items) / len(items)
                cameras = set(item['camera'] for item in items)
                analysis += f"- {obj_type}: {len(items)} detections across {len(cameras)} cameras (avg confidence: {avg_confidence:.2f})\n"
            
            # Recent activity
            if recent_activity:
                analysis += f"\nRecent Activity (by hour):\n"
                for hour, count in sorted(recent_activity.items())[-5:]:  # Last 5 hours
                    analysis += f"- {hour}: {count} detections\n"
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing detection logs: {e}"

    def query_detections(self, user_query: str) -> Dict[str, Any]:
        """
        Handle natural language queries about detection data.
        
        Args:
            user_query: Natural language query about detections
            
        Returns:
            Dictionary with query results and analysis
        """
        try:
            query_lower = user_query.lower()
            
            # Determine query type based on keywords
            if any(word in query_lower for word in ['status', 'health', 'system']):
                return self.get_detection_data("status")
                
            elif any(word in query_lower for word in ['active', 'tracking', 'current', 'now']):
                return self.get_detection_data("active_objects")
                
            elif any(word in query_lower for word in ['log', 'history', 'recent', 'detection']):
                # Extract potential filters
                camera_id = None
                object_type = None
                
                # Look for camera mentions
                if 'camera' in query_lower:
                    # Simple extraction - could be enhanced with NLP
                    words = query_lower.split()
                    for i, word in enumerate(words):
                        if word == 'camera' and i + 1 < len(words):
                            camera_id = words[i + 1]
                            break
                
                # Look for object type mentions
                object_keywords = ['person', 'car', 'vehicle', 'truck', 'bus', 'bicycle']
                for keyword in object_keywords:
                    if keyword in query_lower:
                        object_type = keyword
                        break
                
                return self.get_detection_data("logs", camera_id=camera_id, object_type=object_type)
            
            else:
                # Default to status
                return self.get_detection_data("status")
                
        except Exception as e:
            logger.error(f"Error processing detection query: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_motion_detection(self, analysis_data: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """
        Analyze motion detection performance and provide parameter tuning recommendations.
        This method is called by the adaptive motion detection system.
        """
        try:
            # Extract key information from analysis data
            camera_id = analysis_data.get('camera_id', 'unknown')
            performance = analysis_data.get('performance_metrics', {})
            current_params = analysis_data.get('current_parameters', {})
            motion_result = analysis_data.get('motion_result', {})
            
            logger.info(f"🔍 Analyzing motion detection for camera {camera_id}")
            
            # Prepare the analysis request
            if self.provider_type == AIProviderType.OPENAI:
                response = self._analyze_motion_with_openai(analysis_data, prompt)
            else:
                # Fallback to rule-based analysis
                response = self._analyze_motion_rule_based(analysis_data)
            
            logger.info(f"✅ Motion analysis completed for camera {camera_id}")
            return response
            
        except Exception as e:
            logger.error(f"Motion detection analysis failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "recommendations": {
                    "confidence": 0.0,
                    "reasoning": "Analysis failed due to technical error"
                }
            }
    
    def _analyze_motion_with_openai(self, analysis_data: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Analyze motion detection using OpenAI vision capabilities"""
        try:
            # Check if we have vision capability
            if not analysis_data.get('motion_overlay'):
                return self._analyze_motion_text_only(analysis_data, prompt)
            
            # Prepare OpenAI vision request
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert computer vision engineer specializing in motion detection optimization. Analyze the provided images and data to recommend optimal parameters."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{analysis_data['original_frame']}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{analysis_data['motion_overlay']}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Make API call
            response = self.provider.chat_completion(
                messages=messages,
                model="gpt-4-vision-preview",
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse response
            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']
                return self._parse_motion_analysis_response(content)
            
            return self._get_fallback_analysis(analysis_data)
            
        except Exception as e:
            logger.error(f"OpenAI motion analysis failed: {e}")
            return self._analyze_motion_rule_based(analysis_data)
    
    def _analyze_motion_text_only(self, analysis_data: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Analyze motion detection using text-only LLM"""
        try:
            # Create text-based analysis
            text_prompt = f"""
{prompt}

DETAILED MOTION DATA:
{json.dumps(analysis_data, indent=2, default=str)}

Based on this data, provide motion detection parameter recommendations in JSON format.
"""
            
            response = self.provider.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a motion detection optimization expert."},
                    {"role": "user", "content": text_prompt}
                ],
                model=self.config.get('model', 'gpt-4'),
                max_tokens=1000,
                temperature=0.3
            )
            
            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']
                return self._parse_motion_analysis_response(content)
            
            return self._get_fallback_analysis(analysis_data)
            
        except Exception as e:
            logger.error(f"Text-only motion analysis failed: {e}")
            return self._analyze_motion_rule_based(analysis_data)
    
    def _analyze_motion_rule_based(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based motion analysis when LLM is unavailable"""
        try:
            performance = analysis_data.get('performance_metrics', {})
            current_params = analysis_data.get('current_parameters', {})
            motion_result = analysis_data.get('motion_result', {})
            
            # Rule-based analysis
            false_positive_rate = performance.get('false_positive_rate', 0)
            motion_frequency = performance.get('motion_frequency_per_hour', 0)
            average_score = performance.get('average_score', 0)
            
            recommendations = {}
            reasoning_parts = []
            confidence = 0.7  # Rule-based has moderate confidence
            
            # High false positive rate - increase thresholds
            if false_positive_rate > 0.3:
                recommendations['min_area'] = min(current_params.get('min_area', 1000) * 1.5, 5000)
                recommendations['mog2_var_threshold'] = min(current_params.get('mog2_var_threshold', 16) + 4, 30)
                reasoning_parts.append(f"High false positive rate ({false_positive_rate:.1%}) - increased sensitivity thresholds")
                confidence = 0.8
            
            # Low motion frequency but high average score - might be missing detections
            elif motion_frequency < 1 and average_score > 0.1:
                recommendations['min_area'] = max(current_params.get('min_area', 1000) * 0.8, 500)
                recommendations['mog2_var_threshold'] = max(current_params.get('mog2_var_threshold', 16) - 2, 8)
                reasoning_parts.append("Low detection frequency with high scores - decreased thresholds for better sensitivity")
                confidence = 0.75
            
            # Very high motion frequency - might be too sensitive
            elif motion_frequency > 20:
                recommendations['min_area'] = min(current_params.get('min_area', 1000) * 1.2, 3000)
                reasoning_parts.append("Very high motion frequency - slightly reduced sensitivity")
                confidence = 0.7
            
            # Stable performance - minor optimizations
            else:
                # Fine-tune based on environment
                environment = analysis_data.get('environment', {})
                if environment.get('lighting') == 'night':
                    recommendations['mog2_var_threshold'] = max(current_params.get('mog2_var_threshold', 16) - 1, 10)
                    reasoning_parts.append("Night lighting - slightly increased sensitivity")
                
                confidence = 0.6
            
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Performance appears stable, minimal adjustments recommended"
            
            return {
                "status": "success",
                "analysis": {
                    "performance_assessment": self._assess_performance(performance),
                    "main_issues": self._identify_main_issues(performance),
                    "false_positives_detected": false_positive_rate > 0.2
                },
                "recommendations": {
                    **recommendations,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
            }
            
        except Exception as e:
            logger.error(f"Rule-based motion analysis failed: {e}")
            return self._get_fallback_analysis(analysis_data)
    
    def _parse_motion_analysis_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response for motion analysis"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                # Validate required fields
                if 'recommendations' in parsed and 'status' in parsed:
                    return parsed
            
            # Fallback parsing for non-JSON responses
            return self._parse_text_response(content)
            
        except Exception as e:
            logger.error(f"Failed to parse motion analysis response: {e}")
            return {
                "status": "error",
                "message": "Failed to parse LLM response",
                "recommendations": {"confidence": 0.0}
            }
    
    def _parse_text_response(self, content: str) -> Dict[str, Any]:
        """Parse text-based LLM response"""
        recommendations = {}
        confidence = 0.5
        reasoning = "Parsed from text response"
        
        # Simple keyword-based parsing
        content_lower = content.lower()
        
        if "increase" in content_lower and "threshold" in content_lower:
            recommendations['mog2_var_threshold'] = 20
            confidence = 0.6
        elif "decrease" in content_lower and "threshold" in content_lower:
            recommendations['mog2_var_threshold'] = 12
            confidence = 0.6
        
        if "increase" in content_lower and "area" in content_lower:
            recommendations['min_area'] = 1500
            confidence = 0.6
        elif "decrease" in content_lower and "area" in content_lower:
            recommendations['min_area'] = 800
            confidence = 0.6
        
        return {
            "status": "success",
            "analysis": {"performance_assessment": "unknown"},
            "recommendations": {
                **recommendations,
                "confidence": confidence,
                "reasoning": reasoning
            }
        }
    
    def _assess_performance(self, performance: Dict[str, Any]) -> str:
        """Assess overall performance quality"""
        false_positive_rate = performance.get('false_positive_rate', 0)
        consistency = performance.get('detection_consistency', 0.5)
        
        if false_positive_rate < 0.1 and consistency > 0.8:
            return "excellent"
        elif false_positive_rate < 0.2 and consistency > 0.6:
            return "good"
        elif false_positive_rate < 0.4 and consistency > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _identify_main_issues(self, performance: Dict[str, Any]) -> List[str]:
        """Identify main performance issues"""
        issues = []
        
        false_positive_rate = performance.get('false_positive_rate', 0)
        motion_frequency = performance.get('motion_frequency_per_hour', 0)
        consistency = performance.get('detection_consistency', 0.5)
        
        if false_positive_rate > 0.3:
            issues.append("high_false_positive_rate")
        
        if motion_frequency > 30:
            issues.append("excessive_motion_detections")
        elif motion_frequency < 0.5:
            issues.append("low_motion_sensitivity")
        
        if consistency < 0.4:
            issues.append("inconsistent_detection_performance")
        
        return issues
    
    def _get_fallback_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get basic fallback analysis when all else fails"""
        return {
            "status": "success",
            "analysis": {
                "performance_assessment": "unknown",
                "main_issues": [],
                "false_positives_detected": False
            },
            "recommendations": {
                "confidence": 0.3,
                "reasoning": "Fallback analysis - no specific recommendations"
            }
        }
    
    def analyze_image_with_context(self, image_base64: str, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze image with additional context - used as fallback for motion analysis
        """
        try:
            # This is a fallback method for motion analysis
            if context and 'motion_result' in context:
                return self._analyze_motion_rule_based(context)
            
            # Scene analysis fallback
            if context and 'scene_analysis' in str(prompt).lower():
                return self._analyze_scene_rule_based(context)
            
            # Regular image analysis (existing functionality)
            return {
                "status": "success",
                "analysis": "Image analysis not implemented for this context",
                "recommendations": {"confidence": 0.0}
            }
            
        except Exception as e:
            logger.error(f"Image analysis with context failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def analyze_scene_summary(self, analysis_data: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """
        Analyze scene and provide detailed summary with change detection.
        This method is called by the scene analysis system.
        """
        try:
            # Extract key information from analysis data
            camera_id = analysis_data.get('camera_id', 'unknown')
            timestamp = analysis_data.get('timestamp', datetime.now().isoformat())
            
            logger.info(f"🎬 Analyzing scene for camera {camera_id}")
            
            # Prepare the analysis request
            if self.provider_type == AIProviderType.OPENAI:
                response = self._analyze_scene_with_openai(analysis_data, prompt)
            else:
                # Fallback to rule-based analysis
                response = self._analyze_scene_rule_based(analysis_data)
            
            logger.info(f"✅ Scene analysis completed for camera {camera_id}")
            return response
            
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "scene_summary": {},
                "changes_detected": {"has_changes": False},
                "security_assessment": {"threat_level": "none"}
            }
    
    def _analyze_scene_with_openai(self, analysis_data: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Analyze scene using OpenAI vision capabilities"""
        try:
            # Check if we have vision capability
            if not analysis_data.get('frame'):
                return self._analyze_scene_text_only(analysis_data, prompt)
            
            # Prepare OpenAI vision request
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert surveillance analyst specializing in scene monitoring and change detection. Analyze the provided scene image and provide detailed observations."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{analysis_data['frame']}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Make API call
            response = self.provider.chat_completion(
                messages=messages,
                model="gpt-4-vision-preview",
                max_tokens=2000,
                temperature=0.2
            )
            
            # Parse response
            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']
                return self._parse_scene_analysis_response(content)
            
            return self._get_fallback_scene_analysis(analysis_data)
            
        except Exception as e:
            logger.error(f"OpenAI scene analysis failed: {e}")
            return self._analyze_scene_rule_based(analysis_data)
    
    def _analyze_scene_text_only(self, analysis_data: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Analyze scene using text-only LLM"""
        try:
            # Create text-based analysis
            text_prompt = f"""
{prompt}

SCENE DATA (NO IMAGE AVAILABLE):
{json.dumps(analysis_data, indent=2, default=str)}

Based on this data and context, provide scene analysis in JSON format.
"""
            
            response = self.provider.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a scene analysis expert."},
                    {"role": "user", "content": text_prompt}
                ],
                model=self.config.get('model', 'gpt-4'),
                max_tokens=1500,
                temperature=0.2
            )
            
            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']
                return self._parse_scene_analysis_response(content)
            
            return self._get_fallback_scene_analysis(analysis_data)
            
        except Exception as e:
            logger.error(f"Text-only scene analysis failed: {e}")
            return self._analyze_scene_rule_based(analysis_data)
    
    def _analyze_scene_rule_based(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based scene analysis when LLM is unavailable"""
        try:
            camera_id = analysis_data.get('camera_id', 'unknown')
            motion_detected = analysis_data.get('motion_detected', False)
            lighting = analysis_data.get('lighting', 'unknown')
            previous_summaries = analysis_data.get('previous_summaries', [])
            
            # Basic scene assessment
            scene_type = "outdoor" if "north entrance" in camera_id.lower() else "general"
            
            # Detect changes based on motion and previous data
            has_changes = motion_detected or len(previous_summaries) == 0
            change_significance = "minor" if motion_detected else "none"
            
            # Basic security assessment
            threat_level = "low" if motion_detected else "none"
            
            return {
                "status": "success",
                "scene_summary": {
                    "scene_type": scene_type,
                    "overall_description": f"Camera {camera_id} monitoring area during {lighting} conditions",
                    "visible_objects": ["various objects detected by motion system"],
                    "vehicles": [],
                    "people": [],
                    "structures": ["monitoring area infrastructure"],
                    "lighting_assessment": lighting,
                    "weather_conditions": "clear"
                },
                "changes_detected": {
                    "has_changes": has_changes,
                    "change_summary": "Motion detected" if motion_detected else "No significant changes",
                    "new_objects": [],
                    "moved_objects": [],
                    "missing_objects": [],
                    "significance": change_significance
                },
                "security_assessment": {
                    "threat_level": threat_level,
                    "anomalies": [],
                    "recommendations": ["Continue monitoring"] if motion_detected else ["Normal surveillance"]
                },
                "temporal_notes": {
                    "time_of_day_factors": f"Analysis during {lighting} hours",
                    "expected_activity": "Normal for this time period",
                    "unusual_for_time": []
                }
            }
            
        except Exception as e:
            logger.error(f"Rule-based scene analysis failed: {e}")
            return self._get_fallback_scene_analysis(analysis_data)
    
    def _parse_scene_analysis_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response for scene analysis"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['scene_summary', 'changes_detected', 'security_assessment']
                if all(field in parsed for field in required_fields):
                    return parsed
            
            # Fallback parsing for non-JSON responses
            return self._parse_scene_text_response(content)
            
        except Exception as e:
            logger.error(f"Failed to parse scene analysis response: {e}")
            return {
                "status": "error",
                "message": "Failed to parse LLM response",
                "scene_summary": {},
                "changes_detected": {"has_changes": False},
                "security_assessment": {"threat_level": "none"}
            }
    
    def _parse_scene_text_response(self, content: str) -> Dict[str, Any]:
        """Parse text-based LLM response for scene analysis"""
        has_changes = any(word in content.lower() for word in ['change', 'new', 'different', 'moved'])
        threat_detected = any(word in content.lower() for word in ['threat', 'suspicious', 'unusual', 'concern'])
        
        return {
            "status": "success",
            "scene_summary": {
                "scene_type": "general",
                "overall_description": content[:200] + "..." if len(content) > 200 else content,
                "visible_objects": [],
                "vehicles": [],
                "people": [],
                "structures": [],
                "lighting_assessment": "unknown",
                "weather_conditions": "unknown"
            },
            "changes_detected": {
                "has_changes": has_changes,
                "change_summary": "Changes detected in analysis" if has_changes else "No changes noted",
                "new_objects": [],
                "moved_objects": [],
                "missing_objects": [],
                "significance": "minor" if has_changes else "none"
            },
            "security_assessment": {
                "threat_level": "low" if threat_detected else "none",
                "anomalies": [],
                "recommendations": ["Review analysis"] if threat_detected else ["Continue monitoring"]
            },
            "temporal_notes": {
                "time_of_day_factors": "Analysis based on text parsing",
                "expected_activity": "Unknown",
                "unusual_for_time": []
            }
        }
    
    def _get_fallback_scene_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get basic fallback scene analysis when all else fails"""
        return {
            "status": "success",
            "scene_summary": {
                "scene_type": "unknown",
                "overall_description": f"Camera {analysis_data.get('camera_id', 'unknown')} monitoring",
                "visible_objects": [],
                "vehicles": [],
                "people": [],
                "structures": [],
                "lighting_assessment": analysis_data.get('lighting', 'unknown'),
                "weather_conditions": "unknown"
            },
            "changes_detected": {
                "has_changes": False,
                "change_summary": "Fallback analysis - no specific changes detected",
                "new_objects": [],
                "moved_objects": [],
                "missing_objects": [],
                "significance": "none"
            },
            "security_assessment": {
                "threat_level": "none",
                "anomalies": [],
                "recommendations": ["System fallback - manual review recommended"]
            },
            "temporal_notes": {
                "time_of_day_factors": "Unknown",
                "expected_activity": "Unknown",
                "unusual_for_time": []
            }
        }