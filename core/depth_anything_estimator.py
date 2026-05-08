"""
Depth Anything V2 Estimator
Wrapper for state-of-the-art monocular depth estimation using Depth Anything V2
"""
import cv2
import torch
import numpy as np
from pathlib import Path
import subprocess
import sys
import logging
import contextlib
from typing import Optional

logger = logging.getLogger(__name__)

from core.paths import get_models_dir

_REPO_DIR = Path(__file__).resolve().parent.parent / 'depth_anything_v2_repo'
_REPO_URL = "https://github.com/DepthAnything/Depth-Anything-V2"


def _ensure_depth_anything_repo() -> Path:
    """Clone the Depth-Anything-V2 repo if it isn't already present."""
    marker = _REPO_DIR / "depth_anything_v2" / "dpt.py"
    if marker.exists():
        return _REPO_DIR

    logger.info(
        "[DepthAnything] depth_anything_v2_repo not found – cloning %s …",
        _REPO_URL,
    )
    try:
        subprocess.check_call(
            ["git", "clone", "--depth", "1", _REPO_URL, str(_REPO_DIR)],
            timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "git is not installed. Install git and restart, or manually clone "
            f"{_REPO_URL} into {_REPO_DIR}"
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to clone Depth-Anything-V2 repo (exit {exc.returncode}). "
            f"Check your network connection or manually clone {_REPO_URL} into {_REPO_DIR}"
        ) from exc

    if not marker.exists():
        raise RuntimeError(
            f"Cloned repo at {_REPO_DIR} does not contain depth_anything_v2/dpt.py. "
            "The upstream repo structure may have changed."
        )
    logger.info("[DepthAnything] Repo cloned successfully.")
    return _REPO_DIR


def _import_depth_anything_v2():
    """Lazily import DepthAnythingV2 from the cloned repo."""
    repo_path = _ensure_depth_anything_repo()
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from depth_anything_v2.dpt import DepthAnythingV2
    return DepthAnythingV2


class DepthAnythingEstimator:
    """
    Depth Anything V2 wrapper for high-quality monocular depth estimation.
    Significantly more accurate than traditional methods.
    
    Optimized for production with GPU acceleration and configurable precision.
    """
    
    def __init__(self, model_size='vits', device=None, use_fp16=None, optimize=True, memory_fraction: Optional[float] = None):
        """
        Initialize Depth Anything V2
        
        Args:
            model_size: 'vits' (Small - 24.8M, fast), 'vitb' (Base - 97.5M), 'vitl' (Large - 335.3M)
            device: 'cuda', 'cpu', or None (auto-detect)
            use_fp16: Use half precision (FP16) for faster inference on GPU. None = auto (True for CUDA)
            optimize: Enable optimizations (CUDNN benchmark, JIT compilation)
        """
        DepthAnythingV2 = _import_depth_anything_v2()

        # Device selection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if isinstance(self.device, str) and self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.error("[DepthAnything] CUDA requested but not available in this PyTorch build. Forcing CPU mode.")
            logger.error(
                "[DepthAnything] Fix by installing a CUDA-enabled PyTorch wheel, e.g.: "
                "pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio"
            )
            self.device = "cpu"

        if self.device == "cpu":
            logger.warning(
                "[DepthAnything] Running on CPU. Depth inference will be VERY slow. "
                "A CUDA-capable NVIDIA GPU is strongly recommended for real-time depth estimation."
            )

        # Normalize device handling and CUDA checks
        self.is_cuda = str(self.device).startswith('cuda') and torch.cuda.is_available()
        self._device_index = 0
        if self.is_cuda:
            try:
                # Parse explicit device index if provided (e.g., 'cuda:1')
                if isinstance(self.device, str) and ':' in self.device:
                    self._device_index = int(str(self.device).split(':', 1)[1])
                else:
                    self._device_index = torch.cuda.current_device()
                torch.cuda.set_device(self._device_index)
            except Exception as e:
                logger.warning(f"[DepthAnything] Could not set CUDA device: {e}")
                self._device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0

        # FP16 optimization (only on CUDA)
        if use_fp16 is None:
            self.use_fp16 = self.is_cuda
        else:
            self.use_fp16 = bool(use_fp16) and self.is_cuda
        
        self.optimize = optimize
        
        # Model configurations
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        if model_size not in model_configs:
            raise ValueError(f"Invalid model_size: {model_size}. Must be one of {list(model_configs.keys())}")
        
        self.model_size = model_size
        self.model = DepthAnythingV2(**model_configs[model_size])
        
        # Load weights
        checkpoint_path = get_models_dir() / "depth_anything" / f"depth_anything_v2_{model_size}.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {checkpoint_path}. "
                f"Please download from: "
                f"https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_{model_size}.pth"
            )
        
        logger.info(f"[DepthAnything] Loading {model_size} model from {checkpoint_path}")
        
        # VERIFY CUDA before loading
        if self.is_cuda:
            if not torch.cuda.is_available():
                logger.error("[DepthAnything] ✗ CUDA NOT AVAILABLE! Forcing CPU mode")
                logger.error("[DepthAnything] Install CUDA-enabled PyTorch:")
                logger.error("[DepthAnything]   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                self.device = 'cpu'
                self.use_fp16 = False
            else:
                logger.info(f"[DepthAnything] ✓ CUDA is available")
                logger.info(f"[DepthAnything]   PyTorch CUDA version: {torch.version.cuda}")
                logger.info(f"[DepthAnything]   cuDNN version: {torch.backends.cudnn.version()}")

                # Optionally limit process memory usage to share GPU with other workloads
                try:
                    if memory_fraction is not None and hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                        frac = max(0.1, min(float(memory_fraction), 0.95))
                        torch.cuda.set_per_process_memory_fraction(frac, device=self._device_index)
                        logger.info(f"[DepthAnything] Set per-process GPU memory fraction to {frac:.2f} on cuda:{self._device_index}")
                except Exception as e:
                    logger.warning(f"[DepthAnything] Failed to set GPU memory fraction: {e}")
        
        # Load weights on CPU first to avoid CUDA deserialization issues, then move model
        logger.info(f"[DepthAnything] Loading model weights (CPU deserialization, then move to {self.device})")
        try:
            weights = torch.load(checkpoint_path, map_location='cpu')
            # Some checkpoints store under 'state_dict'
            if isinstance(weights, dict) and 'state_dict' in weights:
                weights = weights['state_dict']
            self.model.load_state_dict(weights)
        except Exception as e:
            logger.error(f"[DepthAnything] Failed to load weights: {e}")
            raise
        self.model = self.model.to(self.device).eval()

        # Use a dedicated CUDA stream to reduce contention with other GPU tasks (e.g., object detection)
        self.cuda_stream = None
        if self.is_cuda and torch.cuda.is_available():
            try:
                self.cuda_stream = torch.cuda.Stream()
                logger.info("[DepthAnything] Using dedicated CUDA stream for depth inference")
            except Exception as e:
                logger.warning(f"[DepthAnything] Could not create CUDA stream: {e}")
        
        # VERIFY model is actually on GPU
        if self.is_cuda:
            try:
                first_param_device = next(self.model.parameters()).device
                logger.info(f"[DepthAnything] Model first parameter device: {first_param_device}")
                if first_param_device.type != 'cuda':
                    logger.error(f"[DepthAnything] ✗ MODEL IS ON {first_param_device.type}, NOT CUDA!")
                    logger.error("[DepthAnything] This will be VERY SLOW. Check PyTorch installation.")
            except Exception as e:
                logger.warning(f"[DepthAnything] Could not verify model device: {e}")
        
        # Apply FP16 optimization if enabled
        if self.use_fp16:
            logger.info("[DepthAnything] Converting model to FP16 for faster inference")
            self.model = self.model.half()
        
        # Enable CUDA optimizations
        if self.optimize and self.is_cuda:
            logger.info("[DepthAnything] Enabling CUDA optimizations")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        
        # Verify GPU is actually available and working
        if self.is_cuda:
            if not torch.cuda.is_available():
                logger.error("[DepthAnything] CUDA requested but not available! Falling back to CPU")
                self.device = 'cpu'
                self.model = self.model.cpu()
                self.use_fp16 = False
            else:
                cuda_device = self._device_index
                cuda_name = torch.cuda.get_device_name(cuda_device)
                cuda_memory = torch.cuda.get_device_properties(cuda_device).total_memory / (1024**3)
                logger.info(f"[DepthAnything] ✓ GPU Detected: {cuda_name} ({cuda_memory:.1f}GB)")
                
                # Warmup to compile CUDA kernels
                logger.info("[DepthAnything] Warming up GPU...")
                dummy_input = torch.randn(1, 3, 518, 518).to(self.device)
                if self.use_fp16:
                    dummy_input = dummy_input.half()
                with torch.no_grad():
                    _ = self.model.forward(dummy_input)
                logger.info("[DepthAnything] ✓ GPU warmup complete")
        
        logger.info(f"[DepthAnything] Loaded {model_size} model on {self.device} "
                   f"(FP16: {self.use_fp16}, Optimized: {self.optimize})")
        
        # Stats
        self.frame_count = 0
        self.total_time = 0.0
    
    def estimate_depth(self, frame):
        """
        GPU-optimized depth estimation from BGR frame.
        Performs all preprocessing with torch ops on the selected device.
        Returns uint8 depth map of original frame size.
        """
        import time
        import torch.nn.functional as F
        start_time = time.time()

        with torch.no_grad():
            stream_ctx = (torch.cuda.stream(self.cuda_stream) if self.cuda_stream is not None else contextlib.nullcontext())
            with stream_ctx:
                # Step 1: numpy -> torch tensor on target device (single transfer)
                h, w = frame.shape[:2]
                frame_tensor = torch.from_numpy(frame)
                expected_device_type = 'cuda' if self.is_cuda else 'cpu'
                if frame_tensor.device.type != expected_device_type:
                    # Non-blocking transfer to overlap with other streams when possible
                    non_blocking = self.is_cuda
                    frame_tensor = frame_tensor.to(self.device, non_blocking=non_blocking)

                # Step 2: BGR->RGB, HWC->CHW, float [0,1]
                image = frame_tensor.flip(-1).permute(2, 0, 1).float() / 255.0

                # Step 3: Resize to model input
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(518, 518),
                    mode='bilinear',
                    align_corners=False
                )

                # Step 4: Normalization (ImageNet)
                if self.use_fp16:
                    image = image.half()
                mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=image.dtype).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=image.dtype).view(1, 3, 1, 1)
                image = (image - mean) / std

                # Step 5: Model inference (with optional autocast on CUDA)
                if self.use_fp16 and self.is_cuda:
                    try:
                        from torch.cuda.amp import autocast
                        with autocast():
                            depth = self.model.forward(image)
                    except Exception:
                        depth = self.model.forward(image)
                else:
                    depth = self.model.forward(image)

                # Step 6: Resize back to original size
                depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

                # Step 7: Normalize on device, final transfer once
                depth_min = depth.min()
                depth_max = depth.max()
                if (depth_max - depth_min).item() > 1e-8:
                    depth = (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth = torch.zeros_like(depth)

                # Asynchronously move to CPU when using CUDA stream; synchronize before numpy conversion
                if self.is_cuda and self.cuda_stream is not None:
                    depth_cpu = (depth * 255).byte().to('cpu', non_blocking=True)
                    # Ensure all ops in our stream are complete before converting to numpy
                    self.cuda_stream.synchronize()
                    depth_uint8 = depth_cpu.numpy()
                else:
                    depth_uint8 = (depth * 255).byte().cpu().numpy()

        # Update stats
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed

        if self.frame_count % 30 == 0:
            avg_fps = self.frame_count / self.total_time if self.total_time > 0 else 0
            logger.info(
                f"[DepthAnything] Frames: {self.frame_count}, Avg FPS: {avg_fps:.1f}, Last: {elapsed*1000:.1f}ms, Device: {self.device}"
            )

        return depth_uint8
    
    def estimate_depth_raw(self, frame):
        """
        Estimate depth and return raw depth values (not normalized)
        Useful for metric depth applications
        
        Args:
            frame: numpy array (H, W, 3) BGR from cv2
            
        Returns:
            depth: numpy array (H, W) raw depth values
        """
        with torch.no_grad():
            depth = self.model.infer_image(frame)
        return depth
    
    def get_stats(self):
        """Get performance statistics"""
        avg_fps = self.frame_count / self.total_time if self.total_time > 0 else 0
        return {
            'frame_count': self.frame_count,
            'avg_fps': avg_fps,
            'total_time': self.total_time,
            'device': self.device,
            'model_size': self.model_size
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.frame_count = 0
        self.total_time = 0.0


# Quick test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python depth_anything_estimator.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("Initializing Depth Anything V2 (Small)...")
    estimator = DepthAnythingEstimator(model_size='vits')
    
    print(f"Reading image from {image_path}...")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        sys.exit(1)
    
    print("Estimating depth...")
    depth = estimator.estimate_depth(frame)
    
    # Save result
    output_path = "depth_anything_test_output.jpg"
    cv2.imwrite(output_path, depth)
    
    # Also save a colored version
    depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    cv2.imwrite("depth_anything_test_output_colored.jpg", depth_colored)
    
    print(f"\nSuccess!")
    print(f"Grayscale depth saved to: {output_path}")
    print(f"Colored depth saved to: depth_anything_test_output_colored.jpg")
    print(f"\nStats: {estimator.get_stats()}")

