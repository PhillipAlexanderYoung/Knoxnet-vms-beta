"""
Depth Map Processor
Monocular and stereo depth estimation with ORB-based SLAM features
"""

import cv2
import numpy as np
import logging
import threading
import time
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum
import base64

logger = logging.getLogger(__name__)


class DepthMode(Enum):
    """Depth estimation modes"""
    MONOCULAR = "monocular"
    STEREO = "stereo"
    ORB_SLAM = "orb_slam"
    MULTI_CAMERA = "multi_camera"
    DEPTH_ANYTHING = "depth_anything"


class ColorMap(Enum):
    """Color map options for depth visualization"""
    JET = cv2.COLORMAP_JET
    VIRIDIS = cv2.COLORMAP_VIRIDIS
    PLASMA = cv2.COLORMAP_PLASMA
    TURBO = cv2.COLORMAP_TURBO
    INFERNO = cv2.COLORMAP_INFERNO
    MAGMA = cv2.COLORMAP_MAGMA
    BONE = cv2.COLORMAP_BONE
    OCEAN = cv2.COLORMAP_OCEAN


@dataclass
class DepthConfig:
    """Configuration for depth processing"""
    mode: DepthMode = DepthMode.MONOCULAR
    color_map: ColorMap = ColorMap.TURBO
    min_depth: float = 0.0
    max_depth: float = 10.0
    num_disparities: int = 64
    block_size: int = 15
    enable_orb: bool = False  # Disabled by default for max speed (enable for SLAM)
    orb_features: int = 500  # Reduced from 800 for speed
    fps_limit: int = 30  # Increased to 30 for GPU mode - GPU can handle it!
    enable_fusion: bool = False
    camera_pair: Optional[Tuple[str, str]] = None
    # DepthAnythingV2 specific settings
    model_size: str = 'vits'  # 'vits' (fast), 'vitb' (balanced), 'vitl' (accurate)
    device: str = 'auto'  # auto-select GPU if available; fall back to CPU
    use_fp16: bool = True  # Use half precision for 2x speed on GPU
    optimize: bool = True  # Enable CUDA optimizations (TorchScript JIT)
    memory_fraction: Optional[float] = None  # Optional per-process GPU memory fraction (0.1-0.95)


@dataclass
class DepthFrame:
    """Depth frame data"""
    depth_map: np.ndarray
    color_mapped: np.ndarray
    orb_features: Optional[List[cv2.KeyPoint]] = None
    timestamp: float = 0.0
    camera_id: str = ""
    mode: str = ""
    stats: Dict[str, Any] = None


class DepthProcessor:
    """
    Processes camera frames to generate depth maps using various techniques
    Self-sufficient with its own camera connections (WebRTC → RTSP → fallback)
    """
    
    def __init__(self, stream_server=None):
        """Initialize depth processor"""
        self.active_processors: Dict[str, Dict[str, Any]] = {}
        self.depth_configs: Dict[str, DepthConfig] = {}
        self._running = False
        self._lock = threading.Lock()
        
        # Stereo matchers
        self.stereo_matchers: Dict[str, cv2.StereoSGBM] = {}
        
        # ORB detector
        self.orb = cv2.ORB_create()
        
        # Feature tracking for SLAM-lite
        self.prev_frames: Dict[str, np.ndarray] = {}
        self.prev_keypoints: Dict[str, List[cv2.KeyPoint]] = {}
        self.prev_descriptors: Dict[str, np.ndarray] = {}
        
        # Depth Anything V2 (lazy-loaded)
        self.depth_anything = None
        
        # Stream server reference (optional optimization)
        self.stream_server = stream_server
        
        # Processing threads for each camera
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.thread_stop_events: Dict[str, threading.Event] = {}
        
        # Latest depth frames for websocket streaming
        self.latest_depth_frames: Dict[str, DepthFrame] = {}
        
        # Independent video captures for each camera
        self.video_captures: Dict[str, cv2.VideoCapture] = {}
        
        logger.info("✓ Depth processor initialized (self-sufficient mode)")
    
    def start_processing(self, camera_id: str, config: Optional[DepthConfig] = None) -> bool:
        """
        Start depth processing for a camera
        
        Args:
            camera_id: Camera identifier
            config: Depth processing configuration
            
        Returns:
            True if started successfully
        """
        try:
            with self._lock:
                # Check if already running - ensure single process per camera
                if camera_id in self.active_processors:
                    logger.warning(f"⚠ Depth processing already active for camera {camera_id} - stopping and restarting")
                    # Release lock before calling stop_processing (it needs the lock)
                    pass
            
            # Stop existing process if running (outside lock to avoid deadlock)
            if camera_id in self.active_processors:
                logger.info(f"Stopping existing depth process for {camera_id} before restart")
                self.stop_processing(camera_id)
                time.sleep(0.2)  # Brief pause for cleanup
            
            # Now safe to start fresh (or start for first time)
            with self._lock:
                if config is None:
                    config = DepthConfig()
            
            # Validate GPU availability and optimize for maximum FPS
            try:
                import torch
                if config.device == 'cuda' and not torch.cuda.is_available():
                    logger.warning("⚠ CUDA requested but not available, falling back to CPU")
                    config.device = 'cpu'
                    # Reduce FPS limit for CPU (slower)
                    if config.fps_limit > 15:
                        config.fps_limit = 15
                        logger.info("→ FPS limit reduced to 15 for CPU mode")
                elif config.device == 'cuda':
                    # CUDA is available - optimize for maximum FPS
                    logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
                    logger.info(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                    
                    if config.mode == DepthMode.DEPTH_ANYTHING:
                        # Ensure GPU optimizations are enabled for max FPS
                        config.use_fp16 = True
                        config.optimize = True
                        
                        # Increase FPS limit for GPU (it can handle it!)
                        if config.fps_limit < 20:
                            config.fps_limit = 30  # GPU can easily handle 30 FPS
                            logger.info("✓ FPS limit increased to 30 for GPU mode")
                        
                        logger.info("✓ GPU optimizations enabled (FP16 + TorchScript JIT)")
                        logger.info(f"✓ Target FPS: {config.fps_limit}")
            except Exception as e:
                logger.warning(f"Error checking GPU: {e}")
            
            with self._lock:
                self.depth_configs[camera_id] = config
                
                # Initialize stereo matcher if needed
                if config.mode in [DepthMode.STEREO, DepthMode.MULTI_CAMERA]:
                    self._init_stereo_matcher(camera_id, config)
                
                # Initialize Depth Anything if needed (GPU priority)
                if config.mode == DepthMode.DEPTH_ANYTHING:
                    self._init_depth_anything(config)
                
                # Initialize ORB if enabled
                if config.enable_orb:
                    self.orb.setMaxFeatures(config.orb_features)
                
                self.active_processors[camera_id] = {
                    'config': config,
                    'last_frame_time': 0,
                    'frame_count': 0,
                    'active': True,
                    'fps': 0,
                    'last_fps_update': time.time(),
                    'start_time': time.time()
                }

            # Depth and object detection can run concurrently on GPU
            # Both use the GPU efficiently without conflicts
            
            # Start independent camera connection and processing thread
            self._start_processing_thread(camera_id, config)
            logger.info(f"✓ Started depth processing for camera {camera_id} (mode: {config.mode.value}, device: {config.device})")
            
            return True
                
        except Exception as e:
            logger.error(f"Failed to start depth processing for {camera_id}: {e}")
            return False
    
    def stop_processing(self, camera_id: str) -> bool:
        """
        Stop depth processing for a camera
        Gracefully shuts down thread, releases VideoCapture, and cleans up GPU memory
        """
        try:
            logger.info(f"[DepthProc-{camera_id}] Stopping depth processing...")
            
            # First, set stop event to signal thread to exit
            if camera_id in self.thread_stop_events:
                self.thread_stop_events[camera_id].set()
                logger.debug(f"[DepthProc-{camera_id}] Stop event set")
            
            # Release VideoCapture to unblock any pending reads
            # This MUST happen after setting stop event but before waiting for thread
            if camera_id in self.video_captures:
                try:
                    cap = self.video_captures[camera_id]
                    cap.release()
                    logger.debug(f"[DepthProc-{camera_id}] Released VideoCapture")
                except Exception as e:
                    logger.debug(f"[DepthProc-{camera_id}] Error releasing VideoCapture: {e}")
                try:
                    del self.video_captures[camera_id]
                except Exception:
                    pass
                
            # Wait for thread to exit gracefully
            if camera_id in self.processing_threads:
                thread = self.processing_threads[camera_id]
                thread.join(timeout=2.0)  # 2 second timeout
                if thread.is_alive():
                    logger.warning(f"[DepthProc-{camera_id}] ⚠ Processing thread did not exit within timeout")
                else:
                    logger.debug(f"[DepthProc-{camera_id}] ✓ Processing thread exited cleanly")
                del self.processing_threads[camera_id]
                
            if camera_id in self.thread_stop_events:
                del self.thread_stop_events[camera_id]
            
            # Clean up processor state
            with self._lock:
                was_active = camera_id in self.active_processors
                
                if camera_id in self.active_processors:
                    del self.active_processors[camera_id]
                
                if camera_id in self.depth_configs:
                    del self.depth_configs[camera_id]
                
                if camera_id in self.stereo_matchers:
                    del self.stereo_matchers[camera_id]
                
                # Clean up tracking data
                self.prev_frames.pop(camera_id, None)
                self.prev_keypoints.pop(camera_id, None)
                self.prev_descriptors.pop(camera_id, None)
                self.latest_depth_frames.pop(camera_id, None)
                
                if was_active:
                    logger.info(f"✓ Stopped depth processing for camera {camera_id}")
                else:
                    logger.info(f"✓ Depth processing stop called for {camera_id} (was not active)")

            # No detection state restoration needed (depth and detection can coexist)

            # Keep GPU model loaded for performance (lazy cleanup)
            # Model will be reused if depth processing starts again
            # Only release GPU cache if no active processors remain
            try:
                with self._lock:
                    no_active = len(self.active_processors) == 0
                
                if no_active:
                    logger.info("[DepthProcessor] No active processors")
                    
                    # Only clear GPU cache, keep model loaded for fast restart
                    # Model will stay in VRAM (~1.5GB) but ready for instant reuse
                    try:
                        import torch
                        if hasattr(torch, 'cuda') and torch.cuda.is_available():
                            # Synchronize to ensure thread cleanup is complete
                            torch.cuda.synchronize()
                            # Light cleanup - just clear unused cache
                            torch.cuda.empty_cache()
                            logger.info(f"✓ GPU cache cleared (model kept loaded for fast restart)")
                    except Exception as e:
                        logger.warning(f"Error clearing GPU cache: {e}")
                    
                    # Light garbage collection
                    try:
                        import gc
                        gc.collect()
                    except Exception:
                        pass
                    
                    logger.info("✓ Depth processor stopped (GPU model cached for performance)")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
                import traceback
                traceback.print_exc()
            
            return True
                
        except Exception as e:
            logger.error(f"Failed to stop depth processing for {camera_id}: {e}")
            return False
    
    def _start_processing_thread(self, camera_id: str, config: DepthConfig):
        """Start dedicated processing thread that pulls frames from stream server"""
        stop_event = threading.Event()
        self.thread_stop_events[camera_id] = stop_event
        
        thread = threading.Thread(
            target=self._processing_loop,
            args=(camera_id, config, stop_event),
            daemon=True,
            name=f"DepthProc-{camera_id}"
        )
        thread.start()
        self.processing_threads[camera_id] = thread
        
        logger.info(f"✓ Started processing thread for {camera_id}")
    
    def _get_camera_config(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Load camera configuration from cameras.json"""
        try:
            import json
            from pathlib import Path
            
            cameras_file = Path(__file__).parent.parent / 'cameras.json'
            if cameras_file.exists():
                with open(cameras_file, 'r') as f:
                    cameras = json.load(f)
                    for cam in cameras:
                        if cam.get('id') == camera_id:
                            return cam
            return None
        except Exception as e:
            logger.error(f"Failed to load camera config: {e}")
            return None
    
    def _connect_to_camera(self, camera_id: str) -> Optional[cv2.VideoCapture]:
        """
        Connect directly to camera using best available method
        Priority: Stream Server → RTSP → MediaMTX path
        """
        logger.info(f"[DepthProc-{camera_id}] Connecting to camera...")
        
        # Priority 1: Try stream server if available
        if self.stream_server and hasattr(self.stream_server, 'active_streams'):
            if camera_id in self.stream_server.active_streams:
                logger.info(f"[DepthProc-{camera_id}] ✓ Using stream server (fastest)")
                return None  # Will use stream server's last_frame
        
        # Priority 2: Try RTSP URL from camera config
        camera_config = self._get_camera_config(camera_id)
        if camera_config:
            rtsp_url = camera_config.get('rtsp_url') or camera_config.get('url')
            
            if rtsp_url:
                logger.info(f"[DepthProc-{camera_id}] Trying RTSP: {rtsp_url}")
                try:
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    
                    # CRITICAL: Minimize buffering for real-time processing
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer
                    cap.set(cv2.CAP_PROP_FPS, 30)  # Match camera FPS
                    
                    # Reduce timeouts for faster detection of issues
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
                    
                    if cap.isOpened():
                        logger.info(f"[DepthProc-{camera_id}] ✓ Connected via RTSP (low-latency mode)")
                        return cap
                    else:
                        logger.warning(f"[DepthProc-{camera_id}] RTSP connection failed")
                        cap.release()
                except Exception as e:
                    logger.warning(f"[DepthProc-{camera_id}] RTSP error: {e}")
            
            # Priority 3: Try MediaMTX path
            mediamtx_path = camera_config.get('mediamtx_path')
            if mediamtx_path:
                # Construct RTSP URL from MediaMTX path
                mediamtx_rtsp = f"rtsp://localhost:8554/{mediamtx_path}"
                logger.info(f"[DepthProc-{camera_id}] Trying MediaMTX RTSP: {mediamtx_rtsp}")
                try:
                    cap = cv2.VideoCapture(mediamtx_rtsp, cv2.CAP_FFMPEG)
                    
                    # Same low-latency settings
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
                    
                    if cap.isOpened():
                        logger.info(f"[DepthProc-{camera_id}] ✓ Connected via MediaMTX (low-latency mode)")
                        return cap
                    else:
                        logger.warning(f"[DepthProc-{camera_id}] MediaMTX connection failed")
                        cap.release()
                except Exception as e:
                    logger.warning(f"[DepthProc-{camera_id}] MediaMTX error: {e}")
        
        logger.error(f"[DepthProc-{camera_id}] ✗ Failed to connect to camera via any method")
        return None
    
    def _processing_loop(self, camera_id: str, config: DepthConfig, stop_event: threading.Event):
        """
        Main processing loop - connects to camera and processes frames
        Tries: Stream Server → RTSP → MediaMTX (automatic fallback)
        """
        logger.info(f"[DepthProc-{camera_id}] Processing loop started")
        logger.info(f"[DepthProc-{camera_id}] Config: mode={config.mode.value}, fps_limit={config.fps_limit}")
        
        # Connect to camera
        cap = self._connect_to_camera(camera_id)
        if cap:
            self.video_captures[camera_id] = cap
            logger.info(f"[DepthProc-{camera_id}] Using independent VideoCapture")
        
        frame_interval = 1.0 / config.fps_limit
        last_process_time = 0
        frames_received = 0
        connection_method = "stream_server" if cap is None else "direct_rtsp"
        
        # For aggressive buffer flushing
        last_flush_time = 0
        flush_interval = 0.1  # Flush buffer every 100ms
        
        while not stop_event.is_set():
            try:
                current_time = time.time()
                
                # AGGRESSIVE BUFFER FLUSHING - Read continuously to stay real-time
                # This prevents frame backlog that causes 10-15s delay
                if cap and (current_time - last_flush_time) > flush_interval:
                    # Check stop event before flushing
                    if stop_event.is_set():
                        break
                    
                    # Flush old frames more aggressively
                    flushed = 0
                    for _ in range(10):  # Flush up to 10 old frames
                        if stop_event.is_set():
                            break
                        try:
                            ret = cap.grab()  # Grab without decoding (faster)
                            if not ret:
                                break
                            flushed += 1
                        except cv2.error:
                            # VideoCapture was released, exit gracefully
                            logger.debug(f"[DepthProc-{camera_id}] VideoCapture released during flush")
                            break
                    
                    if flushed > 5 and frames_received > 0:
                        logger.debug(f"[DepthProc-{camera_id}] Flushed {flushed} buffered frames")
                    
                    last_flush_time = current_time
                
                # Check stop event again
                if stop_event.is_set():
                    break
                
                # Check if enough time has passed for next processing
                if current_time - last_process_time < frame_interval:
                    time.sleep(0.005)  # Shorter sleep for more responsive flushing
                    continue
                
                # Get frame from best available source
                frame = None
                
                # Method 1: Direct VideoCapture (RTSP/MediaMTX)
                if cap:
                    try:
                        # Read the current (flushed) frame and decode it
                        ret, frame = cap.retrieve()  # Retrieve last grabbed frame
                        if not ret or frame is None:
                            # Fallback: try normal read
                            ret, frame = cap.read()
                    except cv2.error:
                        # VideoCapture was released externally (during stop), exit gracefully
                        logger.debug(f"[DepthProc-{camera_id}] VideoCapture released during read")
                        break
                    
                    if not ret or frame is None:
                        logger.warning(f"[DepthProc-{camera_id}] Failed to read from VideoCapture")
                        # If stopping, don't attempt reconnection
                        if stop_event.is_set():
                            logger.debug(f"[DepthProc-{camera_id}] Stop requested during read failure, exiting")
                            break
                        
                        # Otherwise try to reconnect
                        try:
                            cap.release()
                        except:
                            pass
                        
                        logger.info(f"[DepthProc-{camera_id}] Attempting reconnection...")
                        cap = self._connect_to_camera(camera_id)
                        if cap:
                            self.video_captures[camera_id] = cap
                        time.sleep(0.5)
                        continue
                    
                    if frames_received == 0:
                        logger.info(f"[DepthProc-{camera_id}] ✓ First frame from VideoCapture! Shape: {frame.shape}")
                        logger.info(f"[DepthProc-{camera_id}] ✓ Real-time buffer flushing enabled")
                    frames_received += 1
                
                # Method 2: Stream server (if VideoCapture failed or not available)
                elif self.stream_server and hasattr(self.stream_server, 'active_streams'):
                    stream_info = self.stream_server.active_streams.get(camera_id)
                    if stream_info and 'last_frame' in stream_info:
                        frame = stream_info.get('last_frame')
                        if frame is not None:
                            if frames_received == 0:
                                logger.info(f"[DepthProc-{camera_id}] ✓ First frame from stream server! Shape: {frame.shape}")
                            frames_received += 1
                
                if frame is None:
                    time.sleep(0.1)  # Wait for stream
                    continue
                
                # Check stop event before expensive processing
                if stop_event.is_set():
                    logger.debug(f"[DepthProc-{camera_id}] Stop requested before processing, exiting")
                    break
                
                # Process frame on GPU
                last_process_time = current_time
                depth_frame = self._process_frame_internal(camera_id, frame, config)
                
                # Check stop event after processing (in case it took a while)
                if stop_event.is_set():
                    logger.debug(f"[DepthProc-{camera_id}] Stop requested after processing, exiting")
                    break
                
                if depth_frame:
                    # Store for API access
                    self.latest_depth_frames[camera_id] = depth_frame
                    
                    # Update FPS stats
                    processor_data = self.active_processors.get(camera_id)
                    if processor_data:
                        processor_data['frame_count'] += 1
                        processor_data['last_frame_time'] = current_time
                        
                        # Update FPS every second
                        if current_time - processor_data['last_fps_update'] >= 1.0:
                            fps = processor_data['frame_count'] / (current_time - processor_data['last_fps_update'])
                            processor_data['fps'] = fps
                            processor_data['frame_count'] = 0
                            processor_data['last_fps_update'] = current_time
                            
                            logger.info(f"[DepthProc-{camera_id}] Processing at {fps:.1f} FPS (method: {connection_method})")
                
            except Exception as e:
                logger.error(f"[DepthProc-{camera_id}] Processing error: {e}")
                logger.exception(e)
                # Check if we should continue or exit
                if stop_event.is_set():
                    break
                time.sleep(0.1)
        
        # Cleanup
        logger.info(f"[DepthProc-{camera_id}] Processing loop exiting, cleaning up...")
        if cap:
            try:
                cap.release()
                logger.debug(f"[DepthProc-{camera_id}] Released VideoCapture in cleanup")
            except Exception as e:
                logger.debug(f"[DepthProc-{camera_id}] Error releasing VideoCapture in cleanup: {e}")
            
            if camera_id in self.video_captures:
                try:
                    del self.video_captures[camera_id]
                except:
                    pass
        
        logger.info(f"[DepthProc-{camera_id}] Processing loop stopped cleanly (processed {frames_received} total frames)")
    
    def get_latest_depth_frame(self, camera_id: str) -> Optional[DepthFrame]:
        """Get the latest processed depth frame for a camera"""
        return self.latest_depth_frames.get(camera_id)
    
    def process_frame(self, camera_id: str, frame: np.ndarray) -> Optional[DepthFrame]:
        """
        Process a frame to generate depth map (for manual/HTTP mode)
        
        Args:
            camera_id: Camera identifier
            frame: Input frame (BGR format)
            
        Returns:
            DepthFrame with depth map and visualization
        """
        try:
            if camera_id not in self.active_processors:
                return None
            
            processor_data = self.active_processors[camera_id]
            config = processor_data['config']
            
            # FPS limiting
            current_time = time.time()
            if current_time - processor_data['last_frame_time'] < 1.0 / config.fps_limit:
                return None
            
            processor_data['last_frame_time'] = current_time
            processor_data['frame_count'] += 1
            
            return self._process_frame_internal(camera_id, frame, config)
            
        except Exception as e:
            logger.error(f"Failed to process frame for {camera_id}: {e}")
            return None
    
    def _process_frame_internal(self, camera_id: str, frame: np.ndarray, config: DepthConfig) -> Optional[DepthFrame]:
        """
        Internal frame processing - called by both manual and automatic modes
        
        Args:
            camera_id: Camera identifier
            frame: Input frame (BGR format)
            config: Depth configuration
            
        Returns:
            DepthFrame with depth map and visualization
        """
        try:
            current_time = time.time()
            
            # Process based on mode
            if config.mode == DepthMode.MONOCULAR:
                depth_frame = self._process_monocular(camera_id, frame, config)
            elif config.mode == DepthMode.STEREO:
                depth_frame = self._process_stereo(camera_id, frame, config)
            elif config.mode == DepthMode.ORB_SLAM:
                depth_frame = self._process_orb_slam(camera_id, frame, config)
            elif config.mode == DepthMode.MULTI_CAMERA:
                depth_frame = self._process_multi_camera(camera_id, frame, config)
            elif config.mode == DepthMode.DEPTH_ANYTHING:
                depth_frame = self._process_depth_anything(camera_id, frame, config)
            else:
                depth_frame = self._process_monocular(camera_id, frame, config)
            
            if depth_frame:
                depth_frame.timestamp = current_time
                depth_frame.camera_id = camera_id
                depth_frame.mode = config.mode.value
            
            return depth_frame
            
        except Exception as e:
            logger.error(f"Failed to process frame for {camera_id}: {e}")
            return None
    
    def _process_monocular(self, camera_id: str, frame: np.ndarray, config: DepthConfig) -> Optional[DepthFrame]:
        """
        Advanced monocular depth estimation using multiple cues and structure analysis
        Suitable for SLAM and 3D reconstruction
        """
        try:
            if frame is None or frame.size == 0:
                logger.error(f"Invalid frame received for {camera_id}")
                return None
            
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # ===== 1. EDGE AND CONTOUR DETECTION =====
            # Multi-scale edge detection for better boundary detection
            edges_fine = cv2.Canny(gray, 30, 90)
            edges_coarse = cv2.Canny(gray, 50, 150)
            edges = cv2.bitwise_or(edges_fine, edges_coarse)
            
            # Dilate edges slightly to make them more prominent
            kernel = np.ones((2, 2), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # ===== 2. GRADIENT-BASED DEPTH (Structure from Motion Cue) =====
            # Sobel gradients for texture/detail analysis
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobelx**2 + sobely**2)
            gradient_depth = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # ===== 3. LAPLACIAN SHARPNESS (Focus/Blur Cue) =====
            # Sharp = close, blurry = far (depth from defocus)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
            laplacian_abs = np.absolute(laplacian)
            focus_depth = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # ===== 4. DISTANCE TRANSFORM (Proximity to Edges) =====
            # Distance from edges gives depth boundaries
            dist_transform = cv2.distanceTransform(255 - edges_dilated, cv2.DIST_L2, 5)
            dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # Invert: near edges = closer
            edge_proximity_depth = 255 - dist_normalized
            
            # ===== 5. SUPERPIXEL SEGMENTATION (Object Boundaries) =====
            # Use superpixels to find object regions
            try:
                # SLIC superpixels for segmentation
                slic = cv2.ximgproc.createSuperpixelSLIC(gray, region_size=20, ruler=10.0)
                slic.iterate(10)
                segments = slic.getLabels()
                
                # Calculate depth variance per segment (objects have consistent depth)
                segment_depth = np.zeros_like(gray, dtype=np.float32)
                for seg_id in np.unique(segments):
                    mask = segments == seg_id
                    # Segments closer to camera have more detail
                    seg_mean_gradient = gradient_mag[mask].mean()
                    segment_depth[mask] = seg_mean_gradient
                
                segment_depth = cv2.normalize(segment_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            except:
                # Fallback if SLIC not available
                segment_depth = gradient_depth
            
            # ===== 6. PERSPECTIVE CUE (Y-Position Heuristic) =====
            # Bottom of image = closer (ground plane assumption)
            y_coords = np.arange(h).reshape(h, 1)
            perspective_depth = np.repeat((h - y_coords) * 255 / h, w, axis=1).astype(np.uint8)
            
            # Apply gaussian blur to perspective for smooth gradient
            perspective_depth = cv2.GaussianBlur(perspective_depth, (31, 31), 0)
            
            # ===== 7. INTENSITY CUE (Brightness) =====
            # Normalize brightness as potential depth indicator
            intensity_depth = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            # ===== 8. COMBINE ALL DEPTH CUES =====
            # Weighted combination of all cues
            depth_map = np.zeros_like(gray, dtype=np.float32)
            depth_map += gradient_depth.astype(np.float32) * 0.25    # Texture detail
            depth_map += focus_depth.astype(np.float32) * 0.20       # Sharpness
            depth_map += edge_proximity_depth.astype(np.float32) * 0.20  # Edge distance
            depth_map += segment_depth.astype(np.float32) * 0.15     # Segmentation
            depth_map += perspective_depth.astype(np.float32) * 0.10 # Y-position
            depth_map += intensity_depth.astype(np.float32) * 0.10   # Brightness
            
            # Normalize to 0-255 range
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # ===== 9. ENHANCE DEPTH DISCONTINUITIES =====
            # Make object boundaries more pronounced
            depth_edges = cv2.Canny(depth_map, 50, 150)
            depth_map_enhanced = depth_map.copy()
            depth_map_enhanced[depth_edges > 0] = 255  # Highlight depth discontinuities
            
            # Blend original depth with enhanced version
            depth_map = cv2.addWeighted(depth_map, 0.85, depth_map_enhanced, 0.15, 0)
            
            # ===== 10. BILATERAL FILTERING (Preserve Edges, Smooth Regions) =====
            # This is key for SLAM - preserves object boundaries while smoothing surfaces
            depth_map = cv2.bilateralFilter(depth_map, 9, 75, 75)
            
            # ===== 11. ADAPTIVE HISTOGRAM EQUALIZATION =====
            # Enhance local contrast for better depth perception
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            depth_map = clahe.apply(depth_map)
            
            # ===== 12. GENERATE VISUALIZATION =====
            # Apply color map for visualization
            color_mapped = cv2.applyColorMap(depth_map, config.color_map.value)
            
            # Overlay edge contours in white for structure visualization
            edge_overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edge_overlay[edges > 0] = [255, 255, 255]  # White edges
            color_mapped = cv2.addWeighted(color_mapped, 0.85, edge_overlay, 0.15, 0)
            
            # ===== 13. EXTRACT FEATURES FOR SLAM =====
            orb_features = None
            keypoints_data = []
            if config.enable_orb:
                keypoints, descriptors = self.orb.detectAndCompute(gray, None)
                orb_features = keypoints
                
                # Store keypoint data with depth values for 3D reconstruction
                for kp in keypoints[:500]:  # Limit to top 500 features
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    if 0 <= x < w and 0 <= y < h:
                        depth_value = depth_map[y, x]
                        keypoints_data.append({
                            'x': x,
                            'y': y,
                            'depth': int(depth_value),
                            'size': float(kp.size),
                            'response': float(kp.response)
                        })
            
            # ===== 14. CALCULATE STATISTICS =====
            unique_values = len(np.unique(depth_map))
            
            # Calculate depth distribution
            hist, _ = np.histogram(depth_map, bins=50, range=(0, 255))
            depth_entropy = -np.sum((hist / hist.sum()) * np.log2((hist / hist.sum()) + 1e-10))
            
            stats = {
                'min_depth': float(np.min(depth_map)),
                'max_depth': float(np.max(depth_map)),
                'mean_depth': float(np.mean(depth_map)),
                'std_depth': float(np.std(depth_map)),
                'unique_values': int(unique_values),
                'depth_entropy': float(depth_entropy),
                'features': len(orb_features) if orb_features else 0,
                'keypoints': keypoints_data,
                'edges_detected': int(np.count_nonzero(edges)),
                'algorithm': 'multi-cue-monocular-v2'
            }
            
            logger.debug(f"Depth map: range={stats['min_depth']:.1f}-{stats['max_depth']:.1f}, "
                        f"mean={stats['mean_depth']:.1f}, entropy={depth_entropy:.2f}, "
                        f"unique={unique_values}, features={stats['features']}")
            
            return DepthFrame(
                depth_map=depth_map,
                color_mapped=color_mapped,
                orb_features=orb_features,
                stats=stats
            )
            
        except Exception as e:
            logger.error(f"Monocular processing failed: {e}")
            logger.exception(e)
            return None
    
    def _process_stereo(self, camera_id: str, frame: np.ndarray, config: DepthConfig) -> Optional[DepthFrame]:
        """Process stereo depth using semi-global block matching"""
        try:
            # For single camera, simulate stereo by using previous frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if camera_id not in self.prev_frames:
                self.prev_frames[camera_id] = gray
                return self._process_monocular(camera_id, frame, config)
            
            left = self.prev_frames[camera_id]
            right = gray
            
            # Get stereo matcher
            stereo = self.stereo_matchers.get(camera_id)
            if stereo is None:
                return self._process_monocular(camera_id, frame, config)
            
            # Compute disparity
            disparity = stereo.compute(left, right).astype(np.float32) / 16.0
            
            # Normalize disparity to depth map
            disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # Apply color map
            color_mapped = cv2.applyColorMap(disparity_normalized, config.color_map.value)
            
            # ORB features (no drawing)
            orb_features = None
            if config.enable_orb:
                keypoints, _ = self.orb.detectAndCompute(gray, None)
                orb_features = keypoints
            
            # Update previous frame
            self.prev_frames[camera_id] = gray
            
            stats = {
                'min_depth': float(np.min(disparity_normalized)),
                'max_depth': float(np.max(disparity_normalized)),
                'mean_depth': float(np.mean(disparity_normalized)),
                'features': len(orb_features) if orb_features else 0
            }
            
            return DepthFrame(
                depth_map=disparity_normalized,
                color_mapped=color_mapped,
                orb_features=orb_features,
                stats=stats
            )
            
        except Exception as e:
            logger.error(f"Stereo processing failed: {e}")
            return None
    
    def _process_orb_slam(self, camera_id: str, frame: np.ndarray, config: DepthConfig) -> Optional[DepthFrame]:
        """Process depth with ORB-SLAM lite approach"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect ORB keypoints and descriptors
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            # If we have previous frame, match features
            if camera_id in self.prev_keypoints and camera_id in self.prev_descriptors:
                # Create BFMatcher
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                
                # Match descriptors
                if descriptors is not None and self.prev_descriptors[camera_id] is not None:
                    matches = bf.match(self.prev_descriptors[camera_id], descriptors)
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    # Create depth map based on feature matches and motion
                    depth_map = np.zeros(gray.shape, dtype=np.uint8)
                    
                    for match in matches[:100]:  # Top 100 matches
                        prev_pt = self.prev_keypoints[camera_id][match.queryIdx].pt
                        curr_pt = keypoints[match.trainIdx].pt
                        
                        # Calculate motion (simple parallax estimation)
                        motion = np.linalg.norm(np.array(prev_pt) - np.array(curr_pt))
                        
                        # Draw depth based on motion (more motion = closer)
                        depth_value = min(255, int(motion * 10))
                        cv2.circle(depth_map, (int(curr_pt[0]), int(curr_pt[1])), 
                                 10, depth_value, -1)
                    
                    # Dilate and blur to fill depth map
                    kernel = np.ones((15, 15), np.uint8)
                    depth_map = cv2.dilate(depth_map, kernel, iterations=2)
                    depth_map = cv2.GaussianBlur(depth_map, (31, 31), 0)
                    
                else:
                    depth_map = np.zeros(gray.shape, dtype=np.uint8)
            else:
                depth_map = np.zeros(gray.shape, dtype=np.uint8)
            
            # Apply color map (no keypoint drawing for clean visualization)
            color_mapped = cv2.applyColorMap(depth_map, config.color_map.value)
            
            # Store for next frame
            self.prev_keypoints[camera_id] = keypoints
            self.prev_descriptors[camera_id] = descriptors
            self.prev_frames[camera_id] = gray
            
            stats = {
                'features': len(keypoints),
                'matches': 0,  # Would track matches count
                'mean_depth': float(np.mean(depth_map)) if depth_map is not None else 0
            }
            
            return DepthFrame(
                depth_map=depth_map,
                color_mapped=color_mapped,
                orb_features=keypoints,
                stats=stats
            )
            
        except Exception as e:
            logger.error(f"ORB-SLAM processing failed: {e}")
            return None
    
    def _process_multi_camera(self, camera_id: str, frame: np.ndarray, config: DepthConfig) -> Optional[DepthFrame]:
        """Process depth using multiple camera fusion"""
        # For now, fallback to stereo processing
        # In full implementation, this would fuse data from camera_pair
        return self._process_stereo(camera_id, frame, config)
    
    def _init_stereo_matcher(self, camera_id: str, config: DepthConfig):
        """Initialize stereo matcher for camera"""
        try:
            # Create StereoSGBM matcher
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=config.num_disparities,
                blockSize=config.block_size,
                P1=8 * 3 * config.block_size ** 2,
                P2=32 * 3 * config.block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
            
            self.stereo_matchers[camera_id] = stereo
            logger.info(f"✓ Initialized stereo matcher for {camera_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize stereo matcher: {e}")
    
    def _init_depth_anything(self, config: Optional[DepthConfig] = None):
        """Initialize Depth Anything V2 estimator (lazy-loaded)"""
        try:
            if self.depth_anything is None:
                from core.depth_anything_estimator import DepthAnythingEstimator
                import torch
                
                # Use config settings if provided
                if config:
                    model_size = config.model_size
                    device = None if config.device == 'auto' else config.device
                    use_fp16 = config.use_fp16
                    optimize = config.optimize
                    memory_fraction = config.memory_fraction
                else:
                    # Defaults: fast model, auto device, FP16 enabled
                    model_size = 'vits'
                    device = None
                    use_fp16 = True
                    optimize = True
                    memory_fraction = None
                
                # Log what we're requesting
                logger.info(f"[DepthProcessor] Initializing Depth Anything V2")
                logger.info(f"[DepthProcessor]   Requested - size={model_size}, device={device or 'auto'}, fp16={use_fp16}, optimize={optimize}")
                
                # Check CUDA availability
                cuda_available = torch.cuda.is_available()
                logger.info(f"[DepthProcessor]   CUDA Available: {cuda_available}")
                if cuda_available:
                    logger.info(f"[DepthProcessor]   GPU: {torch.cuda.get_device_name(0)}")
                
                self.depth_anything = DepthAnythingEstimator(
                    model_size=model_size,
                    device=device,
                    use_fp16=use_fp16,
                    optimize=optimize,
                    memory_fraction=memory_fraction
                )
                
                # Verify what device was actually used
                actual_device = self.depth_anything.device
                logger.info(f"[DepthProcessor] ✓ Depth Anything V2 loaded successfully")
                logger.info(f"[DepthProcessor]   ACTUAL DEVICE: {actual_device}")
                if actual_device == 'cuda':
                    logger.info(f"[DepthProcessor]   ✓ GPU ACCELERATION ACTIVE")
                else:
                    logger.warning(f"[DepthProcessor]   ⚠ RUNNING ON CPU (SLOW!)")
        except Exception as e:
            logger.error(f"Failed to initialize Depth Anything V2: {e}")
            logger.exception(e)
            logger.error("Falling back to standard monocular depth estimation")
            self.depth_anything = None
    
    def _process_depth_anything(self, camera_id: str, frame: np.ndarray, config: DepthConfig) -> Optional[DepthFrame]:
        """
        Process depth using Depth Anything V2 (state-of-the-art monocular depth)
        Significantly more accurate than traditional methods
        """
        try:
            import time
            t_start = time.time()
            
            if frame is None or frame.size == 0:
                logger.error(f"Invalid frame received for {camera_id}")
                return None
            
            # Ensure Depth Anything is initialized
            if self.depth_anything is None:
                logger.info(f"[DepthProc-{camera_id}] Initializing Depth Anything V2...")
                self._init_depth_anything(config)
                if self.depth_anything is None:
                    logger.error(f"[DepthProc-{camera_id}] Failed to initialize Depth Anything, falling back")
                    # Fallback to monocular if initialization failed
                    return self._process_monocular(camera_id, frame, config)
                logger.info(f"[DepthProc-{camera_id}] ✓ Depth Anything initialized on {self.depth_anything.device}")
            
            t_init = time.time()
            
            # Estimate depth using Depth Anything V2
            depth_map = self.depth_anything.estimate_depth(frame)
            
            t_depth = time.time()
            
            # Apply color map for visualization (FAST)
            color_mapped = cv2.applyColorMap(depth_map, config.color_map.value)
            
            t_colormap = time.time()
            
            # Extract ORB features if enabled (for SLAM) - THIS IS SLOW ON CPU!
            orb_features = None
            keypoints_data = []
            if config.enable_orb:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Reduce keypoints for speed - only extract top features
                temp_orb = cv2.ORB_create()
                temp_orb.setMaxFeatures(min(config.orb_features, 500))  # Cap at 500 for speed
                temp_orb.setFastThreshold(20)  # Higher threshold = fewer features = faster
                
                keypoints, descriptors = temp_orb.detectAndCompute(gray, None)
                orb_features = keypoints
                
                # Only store keypoints if we have few (for speed)
                h, w = depth_map.shape
                if len(keypoints) < 500:
                    for kp in keypoints:
                        x, y = int(kp.pt[0]), int(kp.pt[1])
                        if 0 <= x < w and 0 <= y < h:
                            depth_value = depth_map[y, x]
                            keypoints_data.append({
                                'x': x,
                                'y': y,
                                'depth': int(depth_value),
                                'size': float(kp.size),
                                'response': float(kp.response)
                            })
            
            t_orb = time.time()
            
            # Log detailed timing on first few frames
            if hasattr(self, '_depth_frame_count'):
                self._depth_frame_count += 1
            else:
                self._depth_frame_count = 1
            
            if self._depth_frame_count <= 10 or self._depth_frame_count % 30 == 0:
                total_time = (t_orb - t_start) * 1000
                depth_time = (t_depth - t_init) * 1000
                colormap_time = (t_colormap - t_depth) * 1000
                orb_time = (t_orb - t_colormap) * 1000
                logger.info(f"[DepthProc-{camera_id}] Frame #{self._depth_frame_count}: "
                           f"Total={total_time:.1f}ms (Depth={depth_time:.1f}ms, Color={colormap_time:.1f}ms, ORB={orb_time:.1f}ms), "
                           f"Device={self.depth_anything.device}")
            
            # Calculate statistics
            stats = {
                'min_depth': float(np.min(depth_map)),
                'max_depth': float(np.max(depth_map)),
                'mean_depth': float(np.mean(depth_map)),
                'std_depth': float(np.std(depth_map)),
                'unique_values': int(len(np.unique(depth_map))),
                'features': len(orb_features) if orb_features else 0,
                'keypoints': keypoints_data,
                'algorithm': 'depth-anything-v2'
            }
            
            logger.debug(f"Depth Anything V2: range={stats['min_depth']:.1f}-{stats['max_depth']:.1f}, "
                        f"mean={stats['mean_depth']:.1f}, features={stats['features']}")
            
            return DepthFrame(
                depth_map=depth_map,
                color_mapped=color_mapped,
                orb_features=orb_features,
                stats=stats
            )
            
        except Exception as e:
            logger.error(f"Depth Anything processing failed: {e}")
            logger.exception(e)
            # Fallback to monocular
            return self._process_monocular(camera_id, frame, config)
    
    def generate_point_cloud(self, depth_frame: DepthFrame, downsample: int = 4) -> Dict[str, Any]:
        """
        Generate 3D point cloud from depth map
        
        Args:
            depth_frame: Depth frame with depth map
            downsample: Downsampling factor for performance (1 = full resolution)
            
        Returns:
            Dictionary with point cloud data
        """
        try:
            depth_map = depth_frame.depth_map
            h, w = depth_map.shape
            
            # Camera intrinsics (simplified pinhole model)
            # Assuming standard FOV of ~60 degrees
            focal_length = w / (2 * np.tan(np.radians(30)))
            cx, cy = w / 2, h / 2
            
            # Generate point cloud
            points = []
            colors = []
            
            # Get color mapped image for coloring points
            color_img = depth_frame.color_mapped
            
            for y in range(0, h, downsample):
                for x in range(0, w, downsample):
                    # Get depth value (normalized 0-255)
                    depth_val = depth_map[y, x]
                    
                    # Skip invalid depths
                    if depth_val == 0:
                        continue
                    
                    # Convert depth to real-world scale (arbitrary units)
                    # Higher depth value = closer to camera
                    z = (255 - depth_val) / 255.0 * 10.0  # 0-10 units
                    
                    # Back-project to 3D using pinhole camera model
                    x_3d = (x - cx) * z / focal_length
                    y_3d = (y - cy) * z / focal_length
                    z_3d = z
                    
                    points.append([float(x_3d), float(y_3d), float(z_3d)])
                    
                    # Get color from color-mapped image
                    b, g, r = color_img[y, x]
                    colors.append([int(r), int(g), int(b)])
            
            return {
                'points': points,
                'colors': colors,
                'count': len(points),
                'downsample': downsample,
                'width': w,
                'height': h
            }
            
        except Exception as e:
            logger.error(f"Failed to generate point cloud: {e}")
            return {'points': [], 'colors': [], 'count': 0}
    
    def encode_depth_frame(self, depth_frame: DepthFrame, include_point_cloud: bool = False) -> Dict[str, Any]:
        """
        Encode depth frame for transmission (OPTIMIZED)
        
        Args:
            depth_frame: DepthFrame to encode
            include_point_cloud: Whether to include 3D point cloud data
            
        Returns:
            Dictionary with encoded data
        """
        try:
            # Encode with FASTEST settings - quality doesn't matter as much for depth maps
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, 75,  # Lower quality = faster encoding
                cv2.IMWRITE_JPEG_OPTIMIZE, 0,   # Disable optimize = faster
                cv2.IMWRITE_JPEG_PROGRESSIVE, 0  # Disable progressive = faster
            ]
            
            _, buffer = cv2.imencode('.jpg', depth_frame.color_mapped, encode_params)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                'image': f"data:image/jpeg;base64,{image_base64}",
                'timestamp': depth_frame.timestamp,
                'camera_id': depth_frame.camera_id,
                'mode': depth_frame.mode,
                'stats': depth_frame.stats or {}
            }
            
            # Optionally include point cloud for 3D visualization (skip for speed)
            if include_point_cloud:
                # Generate highly downsampled point cloud for performance
                point_cloud = self.generate_point_cloud(depth_frame, downsample=8)
                result['point_cloud'] = point_cloud
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to encode depth frame: {e}")
            return {}
    
    def update_config(self, camera_id: str, config_updates: Dict[str, Any]) -> bool:
        """Update depth processing configuration"""
        try:
            if camera_id not in self.depth_configs:
                return False
            
            config = self.depth_configs[camera_id]
            
            # Update config fields
            if 'color_map' in config_updates:
                color_map_name = config_updates['color_map'].upper()
                if hasattr(ColorMap, color_map_name):
                    config.color_map = getattr(ColorMap, color_map_name)
            
            if 'mode' in config_updates:
                mode_name = config_updates['mode'].upper()
                if hasattr(DepthMode, mode_name):
                    new_mode = getattr(DepthMode, mode_name)
                    if new_mode != config.mode:
                        config.mode = new_mode
                        # Reinitialize stereo matcher if needed
                        if new_mode in [DepthMode.STEREO, DepthMode.MULTI_CAMERA]:
                            self._init_stereo_matcher(camera_id, config)
                        # Initialize Depth Anything if needed
                        elif new_mode == DepthMode.DEPTH_ANYTHING:
                            self._init_depth_anything(config)
            
            if 'fps_limit' in config_updates:
                config.fps_limit = int(config_updates['fps_limit'])
            
            if 'enable_orb' in config_updates:
                config.enable_orb = bool(config_updates['enable_orb'])
            
            if 'orb_features' in config_updates:
                config.orb_features = int(config_updates['orb_features'])
                self.orb.setMaxFeatures(config.orb_features)
            
            # DepthAnythingV2 specific config
            reinit_depth_anything = False
            if 'model_size' in config_updates:
                config.model_size = str(config_updates['model_size'])
                reinit_depth_anything = True
            
            if 'device' in config_updates:
                config.device = str(config_updates['device'])
                reinit_depth_anything = True
            
            if 'use_fp16' in config_updates:
                config.use_fp16 = bool(config_updates['use_fp16'])
                reinit_depth_anything = True
            
            if 'optimize' in config_updates:
                config.optimize = bool(config_updates['optimize'])
                reinit_depth_anything = True

            if 'memory_fraction' in config_updates:
                try:
                    val = config_updates['memory_fraction']
                    config.memory_fraction = None if val is None else float(val)
                    reinit_depth_anything = True
                except Exception:
                    logger.warning("Invalid memory_fraction in config update; ignoring")
            
            # Reinitialize DepthAnything if settings changed
            if reinit_depth_anything and config.mode == DepthMode.DEPTH_ANYTHING:
                logger.info(f"Reinitializing Depth Anything V2 with new settings...")
                self.depth_anything = None  # Force reinitialization
                self._init_depth_anything(config)
            
            logger.info(f"✓ Updated depth config for {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config for {camera_id}: {e}")
            return False
    
    def get_active_processors(self) -> List[str]:
        """Get list of active camera IDs"""
        return list(self.active_processors.keys())
    
    def cleanup(self):
        """Cleanup all resources"""
        # Stop all processing threads
        for camera_id in list(self.thread_stop_events.keys()):
            self.stop_processing(camera_id)
        
        # Release all video captures
        for camera_id, cap in list(self.video_captures.items()):
            try:
                cap.release()
            except:
                pass
        
        with self._lock:
            self.active_processors.clear()
            self.depth_configs.clear()
            self.stereo_matchers.clear()
            self.prev_frames.clear()
            self.prev_keypoints.clear()
            self.prev_descriptors.clear()
            self.video_captures.clear()
            self.latest_depth_frames.clear()
        
        logger.info("✓ Depth processor cleaned up")


# Global singleton instance
_depth_processor_instance: Optional[DepthProcessor] = None


def get_depth_processor(stream_server=None) -> DepthProcessor:
    """Get or create the global depth processor instance"""
    global _depth_processor_instance
    if _depth_processor_instance is None:
        _depth_processor_instance = DepthProcessor(stream_server=stream_server)
    elif stream_server and not _depth_processor_instance.stream_server:
        # Update stream server reference if not set
        _depth_processor_instance.stream_server = stream_server
        logger.info("✓ Stream server linked to depth processor")
    return _depth_processor_instance

