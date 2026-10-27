"""
Configuration for Local LLM Service
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class ServiceSettings(BaseSettings):
    """Local LLM Service Configuration"""
    
    # Service settings
    host: str = Field(default="127.0.0.1", description="Service host")
    port: int = Field(default=8102, description="Service port")
    
    # Model settings
    default_model: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="Default model to load on startup"
    )
    model_cache_dir: str = Field(
        default="./models/llm_cache",
        description="Directory to cache downloaded models"
    )
    
    # vLLM settings
    max_model_len: int = Field(default=2048, description="Maximum sequence length")
    gpu_memory_utilization: float = Field(default=0.9, description="GPU memory utilization (0.0-1.0)")
    tensor_parallel_size: int = Field(default=1, description="Number of GPUs for tensor parallelism")
    quantization: Optional[str] = Field(default=None, description="Quantization method (awq, gptq, or None)")
    dtype: str = Field(default="auto", description="Model dtype (auto, float16, bfloat16)")
    
    # Generation defaults
    default_temperature: float = Field(default=0.7, description="Default temperature")
    default_max_tokens: int = Field(default=512, description="Default max tokens")
    default_top_p: float = Field(default=0.9, description="Default top_p")
    
    # Performance
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")
    max_concurrent_requests: int = Field(default=10, description="Max concurrent requests")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        env_prefix = "LLM_"
        case_sensitive = False


def get_settings() -> ServiceSettings:
    """Get service settings from environment"""
    return ServiceSettings()

