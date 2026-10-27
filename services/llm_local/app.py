"""
FastAPI Local LLM Service with OpenAI-compatible API
"""
import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice,
    ChatCompletionUsage, ChatMessage, ModelsResponse, ModelInfo,
    ServiceStatus, ModelLoadRequest, ModelDownloadRequest, ModelDownloadStatus,
    ErrorResponse
)
from .model_manager import ModelManager
from .inference_engine import InferenceEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
settings = get_settings()
model_manager = ModelManager(cache_dir=settings.model_cache_dir)
inference_engine: Optional[InferenceEngine] = None
service_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management"""
    global inference_engine
    
    logger.info("Starting Local LLM Service")
    logger.info(f"Cache directory: {settings.model_cache_dir}")
    
    # Try to load default model if specified
    if settings.default_model:
        try:
            logger.info(f"Loading default model: {settings.default_model}")
            model_path = model_manager.cache.get_model_path(settings.default_model)
            
            if not model_path:
                logger.info("Default model not cached, downloading...")
                model_path = model_manager.download_model(settings.default_model)
            
            inference_engine = InferenceEngine(
                model_path=str(model_path),
                max_model_len=settings.max_model_len,
                gpu_memory_utilization=settings.gpu_memory_utilization,
                tensor_parallel_size=settings.tensor_parallel_size,
                quantization=settings.quantization,
                dtype=settings.dtype,
                lazy_load=True  # Enable lazy loading to reduce idle resource usage
            )
            logger.info(f"Default model loaded: {settings.default_model}")
            
        except Exception as e:
            logger.error(f"Failed to load default model: {e}")
            logger.info("Service will start without a loaded model")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Local LLM Service")
    if inference_engine:
        inference_engine.shutdown()


app = FastAPI(
    title="Local LLM Service",
    description="Production-ready local LLM inference with OpenAI-compatible API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
        "uptime": time.time() - service_start_time
    }


# OpenAI-compatible endpoints
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI-compatible)
    """
    if not inference_engine:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Load a model first using POST /models/load"
        )
    
    try:
        # Generate completion
        start_time = time.time()
        
        generated_text = inference_engine.generate_chat(
            messages=[msg.dict() for msg in request.messages],
            temperature=request.temperature or settings.default_temperature,
            top_p=request.top_p or settings.default_top_p,
            max_tokens=request.max_tokens or settings.default_max_tokens,
            stop=request.stop
        )
        
        generation_time = time.time() - start_time
        
        # Count tokens (rough estimate)
        prompt_text = " ".join([msg.content for msg in request.messages])
        prompt_tokens = len(prompt_text.split())
        completion_tokens = len(generated_text.split())
        
        # Build response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        logger.info(f"Generated {completion_tokens} tokens in {generation_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)"""
    try:
        cached_models = model_manager.list_local_models()
        
        models = []
        for model_info in cached_models:
            models.append(ModelInfo(
                id=model_info["id"],
                created=int(time.time()),
                owned_by="local",
                loaded=(inference_engine is not None and 
                       inference_engine.model_name == model_info["id"]),
                size_gb=model_info.get("size_gb"),
                parameters=None
            ))
        
        return ModelsResponse(data=models)
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Service management endpoints
@app.get("/service/status", response_model=ServiceStatus)
async def get_service_status():
    """Get service status"""
    status = ServiceStatus(
        status="running" if inference_engine else "stopped",
        model_loaded=inference_engine.model_name if inference_engine else None,
        uptime_seconds=time.time() - service_start_time,
        requests_served=inference_engine.requests_served if inference_engine else 0
    )
    
    if inference_engine:
        stats = inference_engine.get_stats()
        status.gpu_memory_used = stats.get("gpu_memory_used")
        status.gpu_memory_total = stats.get("gpu_memory_total")
    
    return status


@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a model into memory"""
    global inference_engine
    
    try:
        logger.info(f"Loading model: {request.model_id}")
        
        # Check if model is cached
        model_path = model_manager.cache.get_model_path(request.model_id)
        if not model_path:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_id} not found in cache. Download it first."
            )
        
        # Shutdown existing engine
        if inference_engine:
            logger.info("Shutting down existing model")
            inference_engine.shutdown()
        
        # Load new model
        inference_engine = InferenceEngine(
            model_path=str(model_path),
            max_model_len=request.max_model_len or settings.max_model_len,
            gpu_memory_utilization=request.gpu_memory_utilization or settings.gpu_memory_utilization,
            tensor_parallel_size=settings.tensor_parallel_size,
            quantization=request.quantization or settings.quantization,
            dtype=settings.dtype
        )
        
        logger.info(f"Model loaded successfully: {request.model_id}")
        return {"status": "success", "model": request.model_id}
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/download", response_model=ModelDownloadStatus)
async def download_model(request: ModelDownloadRequest, background_tasks: BackgroundTasks):
    """Download a model from HuggingFace"""
    try:
        logger.info(f"Starting download: {request.model_id}")
        
        # Download model (blocking for now, can be backgrounded later)
        model_path = model_manager.download_model(
            model_id=request.model_id,
            revision=request.revision
        )
        
        return ModelDownloadStatus(
            model_id=request.model_id,
            status="completed",
            progress=1.0
        )
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return ModelDownloadStatus(
            model_id=request.model_id,
            status="failed",
            error=str(e)
        )


@app.get("/models/cached")
async def list_cached_models():
    """List models in cache"""
    try:
        models = model_manager.list_local_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to list cached models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/recommended")
async def list_recommended_models():
    """List recommended models"""
    return {"models": model_manager.get_recommended_models()}


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a cached model"""
    try:
        # Prevent deletion of loaded model
        if inference_engine and inference_engine.model_name == model_id:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete currently loaded model. Unload it first."
            )
        
        success = model_manager.delete_model(model_id)
        if success:
            return {"status": "success", "message": f"Model {model_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False
    )

