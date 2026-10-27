"""
Local LLM Service Manager

Manages the lifecycle of the local LLM inference service as a subprocess.
"""
import os
import sys
import time
import logging
import subprocess
import requests
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LocalLLMService:
    """Manage local LLM service process or connect to an external one"""
    
    def __init__(self, port: int = 8102, host: str = "127.0.0.1", manage_process: bool = True):
        self.port = port
        self.host = host
        self.manage_process = manage_process
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.base_url = f"http://{host}:{port}"
        self.start_time: Optional[float] = None
        
    def start(self, model_name: Optional[str] = None, background: bool = True) -> bool:
        """
        Start the LLM service
        
        Args:
            model_name: Optional model to load on startup
            background: Run in background (default True)
        
        Returns:
            bool: True if service started successfully
        """
        if self.is_running and self.manage_process:
            logger.warning("LLM service is already running")
            return True
        
        if not self.manage_process:
            logger.info("LLM service is externally managed; checking availability instead of spawning a process")
            # Increased timeout to 120s to allow Docker containers time to start
            ready = self._wait_for_ready(timeout=120)
            self.is_running = ready
            if ready:
                logger.info("✓ External LLM service is reachable")
                return True
            logger.error("External LLM service is not reachable")
            return False
        
        try:
            logger.info(f"Starting Local LLM service on {self.host}:{self.port}")
            
            # Get Python executable
            python_exe = sys.executable
            
            # Set environment variables
            env = os.environ.copy()
            env["LLM_SERVICE_PORT"] = str(self.port)
            env["LLM_SERVICE_HOST"] = self.host
            
            if model_name:
                env["LLM_DEFAULT_MODEL"] = model_name
                logger.info(f"Will load model: {model_name}")
            
            # Get path to services/llm_local
            service_dir = Path(__file__).parent.parent / "services" / "llm_local"
            
            if not service_dir.exists():
                logger.error(f"LLM service directory not found: {service_dir}")
                return False
            
            # Start the service process
            if background:
                # Run in background with output to logs
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                
                stdout_log = log_dir / "llm_service_output.log"
                stderr_log = log_dir / "llm_service_error.log"
                
                with open(stdout_log, 'a') as out, open(stderr_log, 'a') as err:
                    self.process = subprocess.Popen(
                        [python_exe, "-m", "services.llm_local"],
                        env=env,
                        stdout=out,
                        stderr=err,
                        cwd=Path.cwd(),
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
                    )
            else:
                # Run in foreground (for debugging)
                self.process = subprocess.Popen(
                    [python_exe, "-m", "services.llm_local"],
                    env=env,
                    cwd=Path.cwd()
                )
            
            # Wait for service to be ready
            logger.info("Waiting for LLM service to be ready...")
            ready = self._wait_for_ready(timeout=60)
            
            if ready:
                self.is_running = True
                self.start_time = time.time()
                logger.info(f"✓ Local LLM service started successfully (PID: {self.process.pid})")
                return True
            else:
                logger.error("LLM service failed to start within timeout")
                self.stop()
                return False
                
        except Exception as e:
            logger.error(f"Failed to start LLM service: {e}")
            self.is_running = False
            return False
    
    def stop(self) -> bool:
        """Stop the LLM service"""
        if not self.manage_process:
            logger.info("LLM service is externally managed; stop() is a no-op")
            self.is_running = False
            return True
        
        if not self.process:
            logger.info("LLM service is not running")
            return True
        
        try:
            logger.info("Stopping Local LLM service...")
            
            # Try graceful shutdown first
            self.process.terminate()
            
            # Wait for up to 10 seconds
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if not stopped
                logger.warning("Service did not stop gracefully, force killing...")
                self.process.kill()
                self.process.wait(timeout=5)
            
            self.is_running = False
            self.process = None
            self.start_time = None
            logger.info("✓ Local LLM service stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping LLM service: {e}")
            return False
    
    def restart(self, model_name: Optional[str] = None) -> bool:
        """Restart the service"""
        logger.info("Restarting Local LLM service...")
        if self.manage_process:
            self.stop()
            time.sleep(2)  # Brief pause
        return self.start(model_name=model_name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        status = {
            "running": self.is_running or self.health_check(),
            "pid": self.process.pid if self.process else None,
            "uptime_seconds": time.time() - self.start_time if self.start_time else None,
            "url": self.base_url,
            "managed_mode": "internal" if self.manage_process else "external",
        }
        
        # Try to get detailed status from service
        if status["running"]:
            try:
                response = requests.get(
                    f"{self.base_url}/service/status",
                    timeout=2
                )
                if response.status_code == 200:
                    service_status = response.json()
                    status.update(service_status)
            except Exception as e:
                logger.debug(f"Could not fetch service status: {e}")
                status["error"] = str(e)
        
        return status
    
    def list_models(self) -> list:
        """List available models"""
        if not self._ensure_running():
            return []
        
        try:
            response = requests.get(
                f"{self.base_url}/models/cached",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        
        return []
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model"""
        if not self._ensure_running():
            return {"success": False, "error": "Service not running"}
        
        try:
            response = requests.post(
                f"{self.base_url}/models/load",
                json={"model_id": model_id},
                timeout=120  # Model loading can take time
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {"success": False, "error": str(e)}
    
    def download_model(self, model_id: str, revision: str = "main") -> Dict[str, Any]:
        """Download a model from HuggingFace"""
        if not self._ensure_running():
            return {"success": False, "error": "Service not running"}
        
        try:
            response = requests.post(
                f"{self.base_url}/models/download",
                json={"model_id": model_id, "revision": revision},
                timeout=600  # Downloads can take a long time
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return {"success": False, "error": str(e)}
    
    def health_check(self) -> bool:
        """Check if service is healthy"""
        try:
            url = f"{self.base_url}/health"
            response = requests.get(
                url,
                timeout=2
            )
            if response.status_code == 200:
                return True
            logger.warning(f"LLM health check failed: {response.status_code} - {response.text}")
            return False
        except Exception as e:
            logger.warning(f"LLM health check connection failed to {self.base_url}: {e}")
            return False
    
    def _wait_for_ready(self, timeout: int = 60) -> bool:
        """Wait for service to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check if process is still alive
                if self.manage_process and self.process and self.process.poll() is not None:
                    logger.error("Service process terminated unexpectedly")
                    return False
                
                # Check health endpoint
                if self.health_check():
                    return True
                
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Waiting for LLM service... {e}")
                time.sleep(1)
        
        return False
    
    def _ensure_running(self) -> bool:
        """Ensure the service is reachable."""
        if self.is_running:
            return True
        if self.health_check():
            self.is_running = True
            return True
        return False
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.process:
            try:
                self.stop()
            except:
                pass

