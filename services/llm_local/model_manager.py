"""
HuggingFace model download and management
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Callable
from huggingface_hub import snapshot_download, model_info, HfApi
from .cache import ModelCache

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage HuggingFace model downloads and storage"""
    
    def __init__(self, cache_dir: str = "./models/llm_cache"):
        self.cache = ModelCache(cache_dir)
        self.hf_api = HfApi()
        
        # Popular models for quick access
        self.recommended_models = {
            "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "stablelm": "stabilityai/stablelm-3b-4e1t",
            "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
            "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
            "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        }
    
    def download_model(
        self, 
        model_id: str, 
        revision: str = "main",
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Path:
        """
        Download model from HuggingFace Hub
        
        Args:
            model_id: HuggingFace model ID (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            revision: Model revision/branch
            progress_callback: Optional callback for download progress
        
        Returns:
            Path to downloaded model
        """
        try:
            logger.info(f"Downloading model: {model_id}")
            
            # Check if already cached
            cached_path = self.cache.get_model_path(model_id)
            if cached_path:
                logger.info(f"Model already cached at: {cached_path}")
                return cached_path
            
            # Download to cache directory
            safe_name = model_id.replace("/", "--")
            local_dir = self.cache.cache_dir / safe_name
            
            # Use snapshot_download for full model download
            downloaded_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            
            # Calculate size
            size_bytes = sum(
                f.stat().st_size 
                for f in Path(downloaded_path).rglob('*') 
                if f.is_file()
            )
            
            # Register in cache
            self.cache.add_model(model_id, Path(downloaded_path), size_bytes)
            
            logger.info(f"Model downloaded successfully: {downloaded_path}")
            logger.info(f"Size: {size_bytes / (1024**3):.2f} GB")
            
            return Path(downloaded_path)
            
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise
    
    def get_model_info(self, model_id: str) -> Dict:
        """Get model information from HuggingFace"""
        try:
            info = model_info(model_id)
            return {
                "id": model_id,
                "author": info.author,
                "downloads": getattr(info, 'downloads', 0),
                "likes": getattr(info, 'likes', 0),
                "library_name": getattr(info, 'library_name', 'unknown'),
                "tags": info.tags if hasattr(info, 'tags') else [],
            }
        except Exception as e:
            logger.error(f"Failed to get model info for {model_id}: {e}")
            return {"id": model_id, "error": str(e)}
    
    def list_local_models(self) -> list:
        """List locally cached models"""
        return self.cache.list_cached_models()
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a cached model"""
        return self.cache.delete_model(model_id)
    
    def get_recommended_models(self) -> Dict[str, str]:
        """Get list of recommended models"""
        return self.recommended_models
    
    def search_models(self, query: str, limit: int = 10) -> list:
        """Search HuggingFace for models"""
        try:
            models = self.hf_api.list_models(
                search=query,
                filter="text-generation-inference",
                limit=limit,
                sort="downloads"
            )
            return [
                {
                    "id": model.modelId,
                    "downloads": getattr(model, 'downloads', 0),
                    "likes": getattr(model, 'likes', 0),
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []

