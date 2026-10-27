"""
Model caching and storage management
"""
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ModelCache:
    """Manage cached models on disk"""
    
    def __init__(self, cache_dir: str = "./models/llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get path to cached model"""
        # Sanitize model_id for filesystem
        safe_name = model_id.replace("/", "--")
        model_path = self.cache_dir / safe_name
        
        if model_path.exists():
            return model_path
        return None
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if model is cached"""
        return self.get_model_path(model_id) is not None
    
    def add_model(self, model_id: str, model_path: Path, size_bytes: int = 0):
        """Register a cached model"""
        safe_name = model_id.replace("/", "--")
        self.metadata[model_id] = {
            "id": model_id,
            "safe_name": safe_name,
            "path": str(model_path),
            "size_bytes": size_bytes,
            "cached_at": str(Path(model_path).stat().st_mtime)
        }
        self._save_metadata()
    
    def list_cached_models(self) -> List[Dict]:
        """List all cached models"""
        cached_models = []
        for model_id, info in self.metadata.items():
            model_path = Path(info["path"])
            if model_path.exists():
                # Calculate size if not stored
                if "size_bytes" not in info or info["size_bytes"] == 0:
                    size_bytes = sum(
                        f.stat().st_size 
                        for f in model_path.rglob('*') 
                        if f.is_file()
                    )
                    info["size_bytes"] = size_bytes
                    self._save_metadata()
                
                cached_models.append({
                    "id": model_id,
                    "size_gb": info["size_bytes"] / (1024**3),
                    "cached_at": info.get("cached_at", "unknown")
                })
        return cached_models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a cached model"""
        model_path = self.get_model_path(model_id)
        if model_path and model_path.exists():
            try:
                shutil.rmtree(model_path)
                if model_id in self.metadata:
                    del self.metadata[model_id]
                    self._save_metadata()
                logger.info(f"Deleted cached model: {model_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete model {model_id}: {e}")
                return False
        return False
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes"""
        total_size = 0
        for model_id, info in self.metadata.items():
            total_size += info.get("size_bytes", 0)
        return total_size
    
    def clear_cache(self):
        """Clear entire cache (dangerous!)"""
        try:
            for model_id in list(self.metadata.keys()):
                self.delete_model(model_id)
            logger.warning("Cache cleared!")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

