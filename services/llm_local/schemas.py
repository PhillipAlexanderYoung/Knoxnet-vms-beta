"""
Request/Response schemas for OpenAI-compatible API
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# OpenAI Chat Completion API Schemas
class ChatMessage(BaseModel):
    """Chat message"""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request (OpenAI-compatible)"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0


class ChatCompletionChoice(BaseModel):
    """Chat completion choice"""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response (OpenAI-compatible)"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# Model management schemas
class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: str = "model"
    created: int
    owned_by: str
    loaded: bool = False
    size_gb: Optional[float] = None
    parameters: Optional[str] = None


class ModelsResponse(BaseModel):
    """List of available models"""
    object: str = "list"
    data: List[ModelInfo]


# Service control schemas
class ServiceStatus(BaseModel):
    """Service status"""
    status: Literal["running", "stopped", "loading", "error"]
    model_loaded: Optional[str] = None
    uptime_seconds: Optional[float] = None
    requests_served: int = 0
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None


class ModelLoadRequest(BaseModel):
    """Request to load a model"""
    model_id: str
    quantization: Optional[str] = None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None


class ModelDownloadRequest(BaseModel):
    """Request to download a model from HuggingFace"""
    model_id: str
    revision: Optional[str] = "main"


class ModelDownloadStatus(BaseModel):
    """Model download status"""
    model_id: str
    status: Literal["downloading", "completed", "failed"]
    progress: float = 0.0  # 0.0 to 1.0
    downloaded_bytes: int = 0
    total_bytes: Optional[int] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: Dict[str, Any]

