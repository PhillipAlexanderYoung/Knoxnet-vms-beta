"""
vLLM inference engine wrapper
"""
import logging
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("vLLM not available, falling back to transformers")
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    AsyncLLMEngine = None


class InferenceEngine:
    """Production inference engine using vLLM or transformers fallback"""
    
    def __init__(
        self,
        model_path: str,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        dtype: str = "auto",
        lazy_load: bool = True  # Enable lazy loading by default
    ):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.quantization = quantization
        self.dtype = dtype
        self.lazy_load = lazy_load
        
        self.engine = None
        self.tokenizer = None
        self.model_name = Path(model_path).name
        self.start_time = time.time()
        self.requests_served = 0
        self._model_loaded = False
        
        # Only initialize if lazy loading is disabled
        if not self.lazy_load:
            self._initialize_engine()
        else:
            logger.info(f"Lazy loading enabled for model: {self.model_name}. Model will load on first request.")
    
    def _import_torch(self):
        try:
            import torch
            return torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is not installed. Install services/llm_local/requirements.gpu.txt "
                "or run scripts/install-local-ai-extras.ps1."
            ) from exc

    def _import_transformers_stack(self):
        torch = self._import_torch()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers is not installed. Install services/llm_local/requirements.gpu.txt "
                "or run scripts/install-local-ai-extras.ps1."
            ) from exc
        return torch, AutoModelForCausalLM, AutoTokenizer

    def _initialize_engine(self):
        """Initialize the inference engine"""
        if self._model_loaded:
            return  # Already loaded
            
        try:
            logger.info(f"Loading model: {self.model_path}")
            if VLLM_AVAILABLE:
                logger.info(f"Initializing vLLM engine with model: {self.model_path}")
                self.engine = LLM(
                    model=self.model_path,
                    max_model_len=self.max_model_len,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    tensor_parallel_size=self.tensor_parallel_size,
                    quantization=self.quantization,
                    dtype=self.dtype,
                    trust_remote_code=True,
                )
                self.tokenizer = self.engine.get_tokenizer()
                logger.info("vLLM engine initialized successfully")
            else:
                logger.info("Falling back to transformers pipeline")
                torch, AutoModelForCausalLM, AutoTokenizer = self._import_transformers_stack()
                
                # Check GPU availability
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                if torch.cuda.is_available():
                    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                # Load model with GPU optimization
                self.engine = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",  # Automatically distributes to GPU
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                )
                
                logger.info(f"Transformers model loaded on {device}")
                logger.info("Transformers pipeline initialized successfully")
            
            self._model_loaded = True
            logger.info("Model loading complete")
                
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate completion for a prompt"""
        # Lazy load model on first request
        if not self._model_loaded:
            logger.info("First request received, loading model...")
            self._initialize_engine()
        
        try:
            self.requests_served += 1
            
            if VLLM_AVAILABLE and self.engine:
                # Use vLLM
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop
                )
                
                outputs = self.engine.generate([prompt], sampling_params)
                generated_text = outputs[0].outputs[0].text
                
            else:
                # Use transformers model directly
                torch = self._import_torch()
                
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                # Move to same device as model
                device = next(self.engine.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.engine.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode only the new tokens
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate chat completion"""
        try:
            # Format messages into a prompt
            prompt = self._format_chat_prompt(messages)
            return self.generate(prompt, temperature, top_p, max_tokens, stop)
            
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            raise
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt"""
        # Try to use the tokenizer's chat template if available
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Failed to use chat template: {e}")
        
        # Fallback to simple format
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        uptime = time.time() - self.start_time
        
        stats = {
            "model": self.model_name,
            "uptime_seconds": uptime,
            "requests_served": self.requests_served,
            "vllm_enabled": VLLM_AVAILABLE,
        }
        
        # Try to get GPU memory info
        try:
            torch = self._import_torch()
            if torch.cuda.is_available():
                stats["gpu_memory_used"] = torch.cuda.memory_allocated() / (1024**3)
                stats["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            pass
        
        return stats
    
    def shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down inference engine")
        if self.engine:
            try:
                if VLLM_AVAILABLE and hasattr(self.engine, 'shutdown'):
                    self.engine.shutdown()
            except Exception as e:
                logger.error(f"Error during engine shutdown: {e}")
        self.engine = None

