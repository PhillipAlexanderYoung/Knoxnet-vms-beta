import cv2
import numpy as np
import base64
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor


@dataclass
class MotionRegion:
    x: int
    y: int
    w: int
    h: int
    area: int


@dataclass
class MotionResult:
    has_motion: bool
    score: float
    regions: List[MotionRegion]
    overlay_frame: Optional[np.ndarray] = None
    analysis_data: Optional[Dict[str, Any]] = None


class SimpleMotionDetector:
    """
    Tried and true motion detector using MOG2 background subtraction.
    This is the most reliable method used in production systems.
    """

    def __init__(self, camera_id: str = None, ai_agent=None, enable_learning: bool = True, **kwargs):
        # Camera identification and AI integration
        self.camera_id = camera_id or "unknown"
        self.ai_agent = ai_agent
        self.enable_learning = enable_learning
        
        # Use MOG2 background subtractor - optimized for various lighting conditions
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,        # Number of frames to build background model
            varThreshold=16,    # Threshold for pixel variation (will be adaptive)
            detectShadows=False # Disable shadows for more stable motion masks
        )
        
        # Adaptive parameters that can be tuned by LLM (ensure they're never None)
        self.min_area = 5000    # Minimum area to consider motion
        self.kernel_size = 5    # Small kernel for noise reduction
        self.mog2_history = 500
        self.mog2_var_threshold = 32
        # Additional tunables for speed/quality tradeoffs and mask quality
        self.downscale_width = 320  # Max longer side for processing (pixels)
        self.dilate_iterations = 1  # Morphology iterations for stability
        self.learning_rate = 0.02   # Background model learning rate
        # Backward-compat keyword mapping (ignore unknowns)
        try:
            if 'min_area' in kwargs and kwargs['min_area'] is not None:
                self.min_area = int(kwargs['min_area'])
            if 'blur_size' in kwargs and kwargs['blur_size'] is not None:
                self.kernel_size = int(kwargs['blur_size'])
            if 'threshold' in kwargs and kwargs['threshold'] is not None:
                self.mog2_var_threshold = int(kwargs['threshold'])
            if 'history_frames' in kwargs and kwargs['history_frames'] is not None:
                # Map roughly to MOG2 history
                hf = int(kwargs['history_frames'])
                self.mog2_history = max(50, min(2000, hf * 120))
        except Exception:
            pass
        # Normalize min area relative to frame size (portion of frame area considered motion)
        self.min_area_norm = 0.005  # 0.5% of frame by default
        
        # Ensure all parameters are valid numbers
        self._validate_parameters()
        
        # LLM Learning system
        self._learning_data = {
            "analysis_count": 0,
            "last_analysis": 0,
            "analysis_interval": 7200,  # 2 hours in seconds
            "performance_history": [],
            "parameter_history": [],
            "false_positive_count": 0,
            "total_detections": 0
        }
        
        # Scene Analysis system
        self.scene_analysis_enabled = False
        self._scene_analysis_data = {
            "analysis_count": 0,
            "last_scene_analysis": 0,
            "scene_analysis_interval": 1800,  # 30 minutes for scene analysis
            "scene_history": [],
            "object_tracking": {},
            "scene_changes": [],
            "baseline_scene": None
        }
        
        # Performance tracking
        self._detection_history = []
        self._max_history = 100
        
        # Threading for async LLM analysis
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        logging.getLogger(__name__).info(f"Enhanced motion detector initialized for camera {self.camera_id}")

        # Preview storage for tuner
        self._last_fg_mask: Optional[np.ndarray] = None
        self._last_mask_time: float = 0.0
        # Last grayscale frame for glitch heuristics
        self._last_gray: Optional[np.ndarray] = None
        self._last_gray_time: float = 0.0

    def _validate_parameters(self) -> None:
        """Ensure all parameters are valid numbers to prevent NoneType errors"""
        # Set safe defaults for any None or invalid parameters
        if not isinstance(self.min_area, (int, float)) or self.min_area is None:
            self.min_area = 1000
        if not isinstance(self.kernel_size, (int, float)) or self.kernel_size is None:
            self.kernel_size = 3
        if not isinstance(self.mog2_history, (int, float)) or self.mog2_history is None:
            self.mog2_history = 500
        if not isinstance(self.mog2_var_threshold, (int, float)) or self.mog2_var_threshold is None:
            self.mog2_var_threshold = 16
        if not isinstance(self.downscale_width, (int, float)) or self.downscale_width is None:
            self.downscale_width = 320
        if not isinstance(self.dilate_iterations, (int, float)) or self.dilate_iterations is None:
            self.dilate_iterations = 1
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate is None:
            self.learning_rate = 0.02
            
        # Ensure parameters are within reasonable ranges
        self.min_area = max(100, min(10000, self.min_area))
        self.kernel_size = max(1, min(15, self.kernel_size))
        self.mog2_history = max(50, min(2000, self.mog2_history))
        self.mog2_var_threshold = max(4, min(100, self.mog2_var_threshold))
        self.downscale_width = max(64, min(640, int(self.downscale_width)))
        self.dilate_iterations = max(0, min(5, int(self.dilate_iterations)))
        self.learning_rate = max(0.0, min(0.5, float(self.learning_rate)))

    def reset(self) -> None:
        """Reset the detector state"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=int(self.mog2_history or 500),
            varThreshold=int(self.mog2_var_threshold or 16),
            detectShadows=False
        )

    def detect(self, frame_bgr: np.ndarray) -> MotionResult:
        """
        Detect motion using MOG2 background subtraction.
        This is the most reliable method for motion detection.
        """
        try:
            h, w = frame_bgr.shape[:2]
            if w <= 0 or h <= 0:
                return MotionResult(False, 0.0, [])

            # Convert to grayscale for better performance
            if len(frame_bgr.shape) == 3:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame_bgr

                # Adaptive preprocessing for different lighting conditions
                gray = self._adaptive_preprocessing(gray)

            # Downscale for speed while maintaining quality (configurable)
            target_long_side = float(self.downscale_width or 480.0)
            scale = min(1.0, target_long_side / float(max(w, h)))
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                h, w = new_h, new_w

                # Adaptive MOG2 parameters based on lighting conditions
                self._adapt_mog2_parameters(gray)

            # Apply background subtraction - this is the core of reliable motion detection
            # Use a tunable learning rate for responsiveness vs stability
            try:
                fg_mask = self.bg_subtractor.apply(gray, learningRate=float(self.learning_rate))
            except Exception:
                fg_mask = self.bg_subtractor.apply(gray)
        
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=int(self.dilate_iterations))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=int(self.dilate_iterations))

            # Store last mask in original frame size for preview
            try:
                if scale < 1.0:
                    mask_up = cv2.resize(fg_mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_up = fg_mask
                self._last_fg_mask = mask_up
                self._last_mask_time = time.time()
            except Exception:
                pass

            # Quick glitch suppression: ignore frames that show striped artifacts across most columns
            try:
                if self._is_glitch_frame(gray, fg_mask):
                    # Update last mask preview for transparency but ignore as motion
                    try:
                        self._last_fg_mask = cv2.resize(fg_mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else fg_mask
                        self._last_mask_time = time.time()
                    except Exception:
                        pass
                    return MotionResult(False, 0.0, [])
            except Exception:
                # Never fail the pipeline due to glitch checks
                pass

            # Find contours in the foreground mask
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            # Filter contours by area
            regions = []
            total_motion_area = 0
            # Compute effective area threshold in the current (possibly downscaled) space
            original_frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
            effective_min_area = self._get_effective_min_area(original_frame_area)
            # Convert threshold to scaled space used for contour areas
            area_threshold_scaled = int(effective_min_area * (scale * scale))
        
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= area_threshold_scaled:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Scale back to original frame size if we downscaled
                    if scale < 1.0:
                        x = int(x / scale)
                        y = int(y / scale)
                        w = int(w / scale)
                        h = int(h / scale)
                    
                    regions.append(MotionRegion(x=x, y=y, w=w, h=h, area=w*h))
                    total_motion_area += w * h

            # Suppress tiny corner HUD/OSD flickers (e.g., camera OSD clocks or UI artifacts)
            # This does not ignore a fixed region; it only filters very small boxes hugging corners.
            try:
                if regions:
                    fw = frame_bgr.shape[1]
                    fh = frame_bgr.shape[0]
                    margin_x = max(8, int(0.12 * fw))
                    margin_y = max(8, int(0.12 * fh))
                    max_area = int(0.02 * (fw * fh))  # up to 2% of frame

                    def is_corner_flicker(r: MotionRegion) -> bool:
                        if r.area > max_area:
                            return False
                        # small box dimensions as additional guard
                        if r.w > int(0.18 * fw) or r.h > int(0.18 * fh):
                            return False
                        near_left = r.x <= margin_x
                        near_right = (r.x + r.w) >= (fw - margin_x)
                        near_top = r.y <= margin_y
                        near_bottom = (r.y + r.h) >= (fh - margin_y)
                        # Consider only when a box touches both an edge in X and Y (i.e., a corner)
                        return (near_left or near_right) and (near_top or near_bottom)

                    filtered = [r for r in regions if not is_corner_flicker(r)]
                    if len(filtered) != len(regions):
                        regions = filtered
                        total_motion_area = sum(r.w * r.h for r in regions)
            except Exception:
                # Never fail due to suppression logic
                pass

            # Calculate motion score (percentage of frame with motion)
            frame_area = original_frame_area
            score = total_motion_area / frame_area if frame_area > 0 else 0.0

            # Create motion overlay for LLM analysis
            overlay_frame = None
            if self.enable_learning:
                overlay_frame = self._create_motion_overlay(frame_bgr, regions, score, fg_mask)
            
            # Create result
            result = MotionResult(
                has_motion=len(regions) > 0,
                score=score,
                regions=regions,
                overlay_frame=overlay_frame
            )
            
            # Track performance and potentially trigger LLM analysis
            if self.enable_learning:
                self._track_detection_performance(result)
                self._check_llm_analysis_trigger(frame_bgr, result)
            
            # Check for scene analysis
            if self.scene_analysis_enabled:
                self._check_scene_analysis_trigger(frame_bgr, result)
            
            return result
            
        except Exception as e:
            # Comprehensive error logging with parameter state
            logging.getLogger(__name__).error(
                f"Motion detection error for camera {self.camera_id}: {e}\n"
                f"Parameters: min_area={self.min_area}, kernel_size={self.kernel_size}, "
                f"mog2_history={self.mog2_history}, mog2_var_threshold={self.mog2_var_threshold}"
            )
            # Return safe default result
            return MotionResult(False, 0.0, [])

    def _is_glitch_frame(self, gray: np.ndarray, fg_mask: np.ndarray) -> bool:
        """Detect common camera glitch artifacts (vertical banding/striping, full-frame corruption).
        Returns True when the frame should be ignored as motion.
        """
        try:
            mask_bin = (fg_mask > 0).astype(np.uint8)
            h, w = mask_bin.shape[:2]
            if w == 0 or h == 0:
                return False

            # Coverage of motion mask
            cover = float(mask_bin.mean())

            # Fraction of columns that are mostly active (vertical stripes)
            col_active = (mask_bin.sum(axis=0) / float(h))
            frac_cols_high = float((col_active > 0.7).mean())

            # Fraction of rows mostly active (helps differentiate global motion vs stripes)
            row_active = (mask_bin.sum(axis=1) / float(w))
            frac_rows_high = float((row_active > 0.7).mean())

            # Heuristic 1: strong vertical striping across many columns with large overall coverage
            if cover > 0.45 and frac_cols_high > 0.5 and frac_rows_high < 0.25:
                return True

            # Heuristic 2: near full-frame activation likely from decode glitch
            if cover > 0.85:
                return True

            # Heuristic 3: extreme high-frequency alternation columns pattern
            # Count transitions along columns for a central band
            mid_band = mask_bin[max(0, h//4):min(h, 3*h//4), :]
            if mid_band.size > 0:
                transitions = np.abs(np.diff(mid_band, axis=1)).sum() / float(mid_band.shape[0])
                # Many rapid transitions across width indicates alternating stripes
                if transitions / max(1, w) > 0.6 and cover > 0.3:
                    return True

            return False
        except Exception:
            return False

    def _get_effective_min_area(self, frame_area: int) -> int:
        """Return the minimum pixel area in original scale to consider as motion."""
        try:
            base_area = int(self.min_area)
            norm_area = int(max(1, self.min_area_norm * frame_area))
            return max(base_area, norm_area)
        except Exception:
            return int(self.min_area)
    
    def _adaptive_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """
        Universal adaptive preprocessing for ALL lighting conditions and scenarios.
        Automatically detects and optimizes for any environment.
        """
        try:
            # Comprehensive image analysis
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            min_val, max_val = np.min(gray), np.max(gray)
            dynamic_range = max_val - min_val
            
            # Detect various lighting scenarios
            is_very_dark = mean_brightness < 50        # Very low light
            is_dark = mean_brightness < 80             # Low light (night/indoor)
            is_bright = mean_brightness > 180          # Very bright (sunny day)
            is_overexposed = max_val >= 250            # Overexposed areas
            
            has_high_contrast = std_brightness > 60    # High contrast (lights/shadows)
            has_low_contrast = std_brightness < 20     # Low contrast (fog/uniform)
            has_poor_range = dynamic_range < 100       # Poor dynamic range
            
            # Apply appropriate enhancement based on conditions
            processed = gray.copy()
            
            # Handle overexposure and bright lights first
            if is_overexposed or (has_high_contrast and mean_brightness > 100):
                # Reduce saturation from bright lights (headlights, sun, etc.)
                processed = np.clip(processed, 0, 240)
            
            # Apply contrast enhancement based on lighting conditions
            if is_very_dark or (is_dark and has_low_contrast):
                # Very dark or low contrast - aggressive enhancement
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                processed = clahe.apply(processed)
                
            elif is_dark and has_high_contrast:
                # Dark with high contrast (nighttime with lights) - balanced enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(processed)
                
                # Blend based on local brightness to preserve natural look
                dark_mask = processed < 80
                processed = np.where(dark_mask,
                                   0.7 * enhanced + 0.3 * processed,  # More enhancement in dark areas
                                   0.4 * enhanced + 0.6 * processed)  # Less in bright areas
                processed = processed.astype(np.uint8)
                
            elif is_bright and has_low_contrast:
                # Bright but low contrast (overcast day, fog) - gentle enhancement
                clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
                processed = clahe.apply(processed)
                
            elif has_poor_range:
                # Poor dynamic range - stretch histogram
                if dynamic_range > 10:  # Avoid division by zero
                    processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
                else:
                    # Very uniform image - gentle contrast enhancement
                    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                    processed = clahe.apply(processed)
            
            # Final noise reduction for enhanced images
            if not np.array_equal(processed, gray):
                # Apply gentle denoising only if we enhanced the image
                processed = cv2.bilateralFilter(processed, 5, 20, 20)
            
            return processed
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Adaptive preprocessing failed: {e}")
            return gray  # Return original on error
    
    def _adapt_mog2_parameters(self, gray: np.ndarray) -> None:
        """
        Dynamically adapt MOG2 parameters based on current scene conditions.
        Handles all scenarios: day/night, indoor/outdoor, high/low contrast.
        """
        try:
            # Analyze current frame characteristics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Debug logging to catch None values
            if self.mog2_var_threshold is None:
                logging.getLogger(__name__).error(f"Camera {self.camera_id}: mog2_var_threshold is None! Fixing...")
                self.mog2_var_threshold = 16
            
            # Calculate optimal variance threshold based on scene
            base_threshold = self.mog2_var_threshold or 16  # Default to 16 if None
            
            # Adapt for different lighting conditions
            if mean_brightness < 50:
                # Very dark scenes - more sensitive to small changes
                new_threshold = max(8, base_threshold - 4)
            elif mean_brightness < 80:
                # Dark scenes (night) - moderately sensitive
                new_threshold = max(10, base_threshold - 2)
            elif mean_brightness > 200:
                # Very bright scenes - less sensitive to avoid noise
                new_threshold = min(40, base_threshold + 6)
            elif mean_brightness > 150:
                # Bright scenes - slightly less sensitive
                new_threshold = min(30, base_threshold + 3)
            else:
                # Normal lighting - use base threshold
                new_threshold = base_threshold
            
            # Adapt for contrast levels
            if std_brightness > 80:
                # Very high contrast - reduce sensitivity to avoid false positives
                new_threshold = min(50, new_threshold + 4)
            elif std_brightness < 15:
                # Very low contrast - increase sensitivity
                new_threshold = max(6, new_threshold - 3)
            
            # Ensure threshold is valid
            if new_threshold is None or not isinstance(new_threshold, (int, float)):
                new_threshold = 16  # Safe default
                
            # Only recreate if threshold changed significantly
            if abs(new_threshold - base_threshold) > 2:
                self.mog2_var_threshold = new_threshold
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=int(self.mog2_history),
                    varThreshold=int(new_threshold),
                    detectShadows=False
                )
                
        except Exception as e:
            logging.getLogger(__name__).error(f"MOG2 adaptation failed: {e}")
    
    def _create_motion_overlay(self, frame_bgr: np.ndarray, regions: List[MotionRegion], 
                              score: float, fg_mask: np.ndarray) -> np.ndarray:
        """Create visual overlay showing motion detection results for LLM analysis"""
        overlay = frame_bgr.copy()
        
        # Draw motion regions
        for i, region in enumerate(regions):
            # Draw bounding box
            cv2.rectangle(overlay, (region.x, region.y), 
                         (region.x + region.w, region.y + region.h), 
                         (0, 255, 0), 2)
            
            # Add region info
            info_text = f"R{i+1}: {region.area}"
            cv2.putText(overlay, info_text, (region.x, region.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add overall motion info
        motion_text = f"Motion: {'YES' if len(regions) > 0 else 'NO'} | Score: {score:.3f}"
        cv2.putText(overlay, motion_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add parameter info
        param_text = f"MinArea: {self.min_area} | VarThresh: {self.mog2_var_threshold}"
        cv2.putText(overlay, param_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(overlay, timestamp, (10, overlay.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show foreground mask in corner (for LLM analysis)
        if fg_mask is not None:
            mask_small = cv2.resize(fg_mask, (160, 120))
            mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_HOT)
            overlay[10:130, overlay.shape[1]-170:overlay.shape[1]-10] = mask_colored
            cv2.putText(overlay, "FG Mask", (overlay.shape[1]-160, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return overlay
    
    def _track_detection_performance(self, result: MotionResult) -> None:
        """Track detection performance for LLM analysis"""
        detection_data = {
            "timestamp": datetime.now(),
            "has_motion": result.has_motion,
            "score": result.score,
            "regions_count": len(result.regions),
            "total_area": sum(r.area for r in result.regions)
        }
        
        self._detection_history.append(detection_data)
        
        # Keep history manageable
        if len(self._detection_history) > self._max_history:
            self._detection_history.pop(0)
        
        # Update learning data
        self._learning_data["total_detections"] += 1
        
        # Simple false positive detection (rapid successive detections)
        if len(self._detection_history) >= 3:
            recent = self._detection_history[-3:]
            if all(d["has_motion"] for d in recent):
                time_diffs = [
                    (recent[i]["timestamp"] - recent[i-1]["timestamp"]).total_seconds()
                    for i in range(1, len(recent))
                ]
                if all(diff < 2.0 for diff in time_diffs):  # All within 2 seconds
                    self._learning_data["false_positive_count"] += 1
    
    def _check_llm_analysis_trigger(self, frame_bgr: np.ndarray, result: MotionResult) -> None:
        """Check if LLM analysis should be triggered"""
        if not self.ai_agent or not self.enable_learning:
            return
        
        current_time = time.time()
        last_analysis = self._learning_data.get("last_analysis") or 0
        
        # Time-based trigger
        if current_time - last_analysis < self._learning_data["analysis_interval"]:
            return
        
        # Performance-based triggers
        total_detections = self._learning_data["total_detections"]
        false_positives = self._learning_data["false_positive_count"]
        
        should_analyze = False
        
        # Trigger if false positive rate is high
        if total_detections > 20 and false_positives / total_detections > 0.3:
            should_analyze = True
        
        # Trigger for regular learning (every analysis interval)
        elif current_time - last_analysis >= self._learning_data["analysis_interval"]:
            should_analyze = True
        
        if should_analyze:
            self._schedule_llm_analysis(frame_bgr, result)
    
    def _schedule_llm_analysis(self, frame_bgr: np.ndarray, result: MotionResult) -> None:
        """Schedule LLM analysis in background thread"""
        try:
            # Update last analysis time immediately to prevent multiple triggers
            self._learning_data["last_analysis"] = time.time()
            self._learning_data["analysis_count"] += 1
            
            # Schedule async analysis
            self._executor.submit(self._perform_llm_analysis, frame_bgr, result)
            
            logging.getLogger(__name__).info(
                f"Scheduled LLM analysis #{self._learning_data['analysis_count']} for camera {self.camera_id}"
            )
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to schedule LLM analysis: {e}")
    
    def _perform_llm_analysis(self, frame_bgr: np.ndarray, result: MotionResult) -> None:
        """Perform LLM analysis and apply recommendations"""
        try:
            if not self.ai_agent:
                return
            
            # Prepare analysis data
            analysis_data = self._prepare_analysis_data(frame_bgr, result)
            
            # Call AI agent for motion analysis
            recommendations = self._call_ai_agent_analysis(analysis_data)
            
            # Apply recommendations
            if recommendations and recommendations.get("status") == "success":
                self._apply_recommendations(recommendations)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"LLM analysis failed for camera {self.camera_id}: {e}")
    
    def _prepare_analysis_data(self, frame_bgr: np.ndarray, result: MotionResult) -> Dict[str, Any]:
        """Prepare data for LLM analysis"""
        # Encode frames to base64
        _, frame_buffer = cv2.imencode('.jpg', frame_bgr)
        frame_b64 = base64.b64encode(frame_buffer).decode('utf-8')
        
        overlay_b64 = None
        if result.overlay_frame is not None:
            _, overlay_buffer = cv2.imencode('.jpg', result.overlay_frame)
            overlay_b64 = base64.b64encode(overlay_buffer).decode('utf-8')
        
        # Get performance metrics
        performance_metrics = self._get_performance_metrics()
        
        # Determine environment context
        current_hour = datetime.now().hour
        lighting = "day" if 6 <= current_hour <= 18 else "night"
        
        return {
            "camera_id": self.camera_id,
            "timestamp": datetime.now().isoformat(),
            "original_frame": frame_b64,
            "motion_overlay": overlay_b64,
            "motion_result": {
                "has_motion": result.has_motion,
                "score": result.score,
                "regions": [asdict(r) for r in result.regions]
            },
            "current_parameters": {
                "min_area": self.min_area,
                "kernel_size": self.kernel_size,
                "mog2_history": self.mog2_history,
                "mog2_var_threshold": self.mog2_var_threshold
            },
            "performance_metrics": performance_metrics,
            "environment": {
                "lighting": lighting,
                "analysis_count": self._learning_data["analysis_count"]
            }
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        total_detections = self._learning_data["total_detections"]
        false_positives = self._learning_data["false_positive_count"]
        
        # Calculate recent motion frequency
        recent_hour = [
            d for d in self._detection_history 
            if (datetime.now() - d["timestamp"]).total_seconds() < 3600
        ]
        motion_frequency = sum(1 for d in recent_hour if d["has_motion"])
        
        return {
            "total_detections": total_detections,
            "false_positive_count": false_positives,
            "false_positive_rate": false_positives / max(total_detections, 1),
            "motion_frequency_per_hour": motion_frequency,
            "average_score": np.mean([d["score"] for d in self._detection_history]) if self._detection_history else 0,
            "detection_consistency": self._calculate_consistency()
        }
    
    def _calculate_consistency(self) -> float:
        """Calculate detection consistency score"""
        if len(self._detection_history) < 10:
            return 0.5
        
        scores = [d["score"] for d in self._detection_history[-20:]]
        score_variance = np.var(scores)
        
        # Lower variance = higher consistency
        consistency = max(0, 1 - score_variance * 10)
        return min(1, consistency)
    
    def _call_ai_agent_analysis(self, analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call AI agent for motion detection analysis"""
        try:
            # Create analysis prompt for the AI agent
            prompt = self._create_analysis_prompt(analysis_data)
            
            # Call the AI agent's analyze method
            if hasattr(self.ai_agent, 'analyze_motion_detection'):
                return self.ai_agent.analyze_motion_detection(analysis_data, prompt)
            else:
                # Fallback to general analysis
                return self.ai_agent.analyze_image_with_context(
                    analysis_data["original_frame"],
                    prompt,
                    context=analysis_data
                )
                
        except Exception as e:
            logging.getLogger(__name__).error(f"AI agent analysis failed: {e}")
            return None
    
    def _create_analysis_prompt(self, analysis_data: Dict[str, Any]) -> str:
        """Create analysis prompt for AI agent"""
        perf = analysis_data["performance_metrics"]
        params = analysis_data["current_parameters"]
        
        return f"""
Analyze this motion detection system performance and recommend parameter adjustments.

CAMERA: {analysis_data['camera_id']}
LIGHTING: {analysis_data['environment']['lighting']}
ANALYSIS #: {analysis_data['environment']['analysis_count']}

CURRENT PERFORMANCE:
- Total Detections: {perf['total_detections']}
- False Positive Rate: {perf['false_positive_rate']:.2%}
- Motion Frequency: {perf['motion_frequency_per_hour']}/hour
- Average Score: {perf['average_score']:.3f}
- Consistency: {perf['detection_consistency']:.2f}

CURRENT PARAMETERS:
- Min Area: {params['min_area']}
- Kernel Size: {params['kernel_size']}
- MOG2 History: {params['mog2_history']}
- MOG2 Var Threshold: {params['mog2_var_threshold']}

MOTION DETECTION RESULT:
- Motion Detected: {analysis_data['motion_result']['has_motion']}
- Score: {analysis_data['motion_result']['score']:.3f}
- Regions: {len(analysis_data['motion_result']['regions'])}

Please analyze the motion detection overlay image and provide specific parameter recommendations to:
1. Reduce false positives if rate > 20%
2. Improve detection accuracy
3. Optimize for the current environment

Respond with JSON format:
{{
    "status": "success",
    "analysis": {{
        "performance_assessment": "excellent|good|fair|poor",
        "main_issues": ["issue1", "issue2"],
        "false_positives_detected": boolean
    }},
    "recommendations": {{
        "min_area": new_value_or_null,
        "mog2_var_threshold": new_value_or_null,
        "mog2_history": new_value_or_null,
        "kernel_size": new_value_or_null,
        "confidence": 0.0_to_1.0,
        "reasoning": "explanation"
    }}
}}
"""
    
    def _apply_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """Apply LLM recommendations to motion detection parameters"""
        try:
            recs = recommendations.get("recommendations", {})
            confidence = recs.get("confidence", 0)
            
            # Only apply recommendations with sufficient confidence
            if confidence < 0.6:
                logging.getLogger(__name__).info(
                    f"Skipping low-confidence recommendations (confidence: {confidence:.2f})"
                )
                return
            
            changes_made = []
            
            # Apply parameter changes
            if "min_area" in recs and recs["min_area"] is not None:
                old_value = self.min_area
                self.min_area = max(500, min(10000, int(recs["min_area"])))  # Bounds check
                changes_made.append(f"min_area: {old_value} -> {self.min_area}")
            
            if "mog2_var_threshold" in recs and recs["mog2_var_threshold"] is not None:
                old_value = self.mog2_var_threshold
                self.mog2_var_threshold = max(8, min(50, int(recs["mog2_var_threshold"])))
                changes_made.append(f"mog2_var_threshold: {old_value} -> {self.mog2_var_threshold}")
                
                # Recreate background subtractor with new threshold
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=self.mog2_history,
                    varThreshold=self.mog2_var_threshold,
                    detectShadows=False
                )
            
            if "mog2_history" in recs and recs["mog2_history"] is not None:
                old_value = self.mog2_history
                self.mog2_history = max(100, min(1000, int(recs["mog2_history"])))
                changes_made.append(f"mog2_history: {old_value} -> {self.mog2_history}")
                
                # Recreate background subtractor with new history
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=self.mog2_history,
                    varThreshold=self.mog2_var_threshold,
                    detectShadows=False
                )
            
            if "kernel_size" in recs and recs["kernel_size"] is not None:
                old_value = self.kernel_size
                self.kernel_size = max(1, min(7, int(recs["kernel_size"])))
                changes_made.append(f"kernel_size: {old_value} -> {self.kernel_size}")
            
            # Validate parameters after any changes
            if changes_made:
                self._validate_parameters()
            
            # Log changes
            if changes_made:
                reasoning = recs.get("reasoning", "No reasoning provided")
                logging.getLogger(__name__).info(
                    f"Applied LLM recommendations for camera {self.camera_id} "
                    f"(confidence: {confidence:.2f}): {', '.join(changes_made)}"
                )
                logging.getLogger(__name__).info(f"Reasoning: {reasoning}")
                
                # Store parameter history
                self._learning_data["parameter_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "changes": changes_made,
                    "confidence": confidence,
                    "reasoning": reasoning
                })
                
                # Adjust analysis interval based on confidence
                if confidence > 0.8:
                    # High confidence, reduce frequency
                    self._learning_data["analysis_interval"] = min(14400, self._learning_data["analysis_interval"] * 1.5)
                elif confidence < 0.7:
                    # Lower confidence, increase frequency
                    self._learning_data["analysis_interval"] = max(3600, self._learning_data["analysis_interval"] * 0.8)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to apply recommendations: {e}")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status and performance"""
        return {
            "camera_id": self.camera_id,
            "learning_enabled": self.enable_learning,
            "analysis_count": self._learning_data["analysis_count"],
            "last_analysis": self._learning_data.get("last_analysis"),
            "next_analysis_in_seconds": max(0,
                self._learning_data["analysis_interval"] -
                (time.time() - (self._learning_data.get("last_analysis") or 0))
            ),
            "current_parameters": {
                "min_area": self.min_area,
                "kernel_size": self.kernel_size,
                "mog2_history": self.mog2_history,
                "mog2_var_threshold": self.mog2_var_threshold
            },
            "performance_metrics": self._get_performance_metrics(),
            "parameter_history": self._learning_data["parameter_history"][-5:],  # Last 5 changes
        }
    
    def force_analysis(self) -> None:
        """Force immediate LLM analysis on next detection"""
        self._learning_data["last_analysis"] = 0
        logging.getLogger(__name__).info(f"Forced analysis scheduled for camera {self.camera_id}")
    
    def enable_adaptive_learning(self, enabled: bool) -> None:
        """Enable or disable adaptive learning"""
        self.enable_learning = enabled
        logging.getLogger(__name__).info(
            f"Adaptive learning {'enabled' if enabled else 'disabled'} for camera {self.camera_id}"
        )
    
    def enable_scene_analysis(self, enabled: bool) -> None:
        """Enable or disable scene analysis"""
        self.scene_analysis_enabled = enabled
        if enabled:
            # Reset scene analysis data when enabling
            self._scene_analysis_data = {
                "analysis_count": 0,
                "last_scene_analysis": None,
                "scene_analysis_interval": 1800,  # 30 minutes
                "scene_history": [],
                "object_tracking": {},
                "scene_changes": [],
                "baseline_scene": None
            }
        logging.getLogger(__name__).info(
            f"Scene analysis {'enabled' if enabled else 'disabled'} for camera {self.camera_id}"
        )
    
    def _check_scene_analysis_trigger(self, frame_bgr: np.ndarray, result: MotionResult) -> None:
        """Check if scene analysis should be triggered"""
        if not self.ai_agent or not self.scene_analysis_enabled:
            return
        
        current_time = time.time()
        last_analysis = self._scene_analysis_data.get("last_scene_analysis") or 0
        
        # Time-based trigger (every 30 minutes)
        if current_time - last_analysis >= self._scene_analysis_data["scene_analysis_interval"]:
            self._schedule_scene_analysis(frame_bgr, result)
    
    def _schedule_scene_analysis(self, frame_bgr: np.ndarray, result: MotionResult) -> None:
        """Schedule scene analysis in background thread"""
        try:
            # Update last analysis time immediately to prevent multiple triggers
            self._scene_analysis_data["last_scene_analysis"] = time.time()
            self._scene_analysis_data["analysis_count"] += 1
            
            # Schedule async analysis
            self._executor.submit(self._perform_scene_analysis, frame_bgr, result)
            
            logging.getLogger(__name__).info(
                f"Scheduled scene analysis #{self._scene_analysis_data['analysis_count']} for camera {self.camera_id}"
            )
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to schedule scene analysis: {e}")
    
    def _perform_scene_analysis(self, frame_bgr: np.ndarray, result: MotionResult) -> None:
        """Perform AI scene analysis and comparison"""
        try:
            if not self.ai_agent:
                return
            
            # Prepare scene analysis data
            analysis_data = self._prepare_scene_analysis_data(frame_bgr, result)
            
            # Call AI agent for scene analysis
            scene_summary = self._call_ai_agent_scene_analysis(analysis_data)
            
            # Store scene summary and detect changes
            if scene_summary and scene_summary.get("status") == "success":
                self._process_scene_summary(scene_summary)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Scene analysis failed for camera {self.camera_id}: {e}")
    
    def _prepare_scene_analysis_data(self, frame_bgr: np.ndarray, result: MotionResult) -> Dict[str, Any]:
        """Prepare data for scene analysis"""
        # Encode frame to base64
        _, frame_buffer = cv2.imencode('.jpg', frame_bgr)
        frame_b64 = base64.b64encode(frame_buffer).decode('utf-8')
        
        # Get current time and lighting
        current_time = datetime.now()
        current_hour = current_time.hour
        lighting = "day" if 6 <= current_hour <= 18 else "night"
        
        # Get previous scene summaries for comparison
        recent_summaries = self._scene_analysis_data["scene_history"][-3:] if self._scene_analysis_data["scene_history"] else []
        
        return {
            "camera_id": self.camera_id,
            "timestamp": current_time.isoformat(),
            "frame": frame_b64,
            "motion_detected": result.has_motion,
            "motion_score": result.score,
            "motion_regions": [asdict(r) for r in result.regions],
            "lighting": lighting,
            "analysis_count": self._scene_analysis_data["analysis_count"],
            "previous_summaries": recent_summaries,
            "baseline_scene": self._scene_analysis_data.get("baseline_scene")
        }
    
    def _call_ai_agent_scene_analysis(self, analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call AI agent for scene analysis"""
        try:
            # Create scene analysis prompt
            prompt = self._create_scene_analysis_prompt(analysis_data)
            
            # Call the AI agent's scene analysis method
            if hasattr(self.ai_agent, 'analyze_scene_summary'):
                return self.ai_agent.analyze_scene_summary(analysis_data, prompt)
            else:
                # Fallback to general analysis
                return self.ai_agent.analyze_image_with_context(
                    analysis_data["frame"],
                    prompt,
                    context=analysis_data
                )
                
        except Exception as e:
            logging.getLogger(__name__).error(f"AI agent scene analysis failed: {e}")
            return None
    
    def _create_scene_analysis_prompt(self, analysis_data: Dict[str, Any]) -> str:
        """Create scene analysis prompt for AI agent"""
        previous_summaries = analysis_data.get("previous_summaries", [])
        baseline_scene = analysis_data.get("baseline_scene")
        
        prompt = f"""
You are an intelligent surveillance analyst monitoring camera {analysis_data['camera_id']}.
Analyze this scene and provide a detailed summary, noting any changes from previous observations.

CURRENT CONTEXT:
- Camera: {analysis_data['camera_id']}
- Time: {analysis_data['timestamp']}
- Lighting: {analysis_data['lighting']}
- Analysis #: {analysis_data['analysis_count']}
- Motion Detected: {analysis_data['motion_detected']}
- Motion Score: {analysis_data['motion_score']:.3f}

ANALYSIS TASKS:
1. Describe the overall scene (location type, layout, key features)
2. Identify all visible objects, vehicles, people, and structures
3. Note lighting conditions, weather, and environmental factors
4. Compare with previous observations to detect changes
5. Identify any new, moved, or missing objects
6. Assess security-relevant changes or anomalies
7. Analyze movement patterns and activity levels appropriate for current conditions
8. Detect if vehicles are parked vs. moving through the scene
9. Note any unusual activity patterns for the time of day and lighting conditions
10. Assess image quality factors (brightness, contrast, visibility) that may affect monitoring

PREVIOUS SCENE SUMMARIES:
{json.dumps(previous_summaries, indent=2) if previous_summaries else "None - this is the first analysis"}

BASELINE SCENE:
{json.dumps(baseline_scene, indent=2) if baseline_scene else "Not established yet"}

Respond with JSON format:
{{
    "status": "success",
    "scene_summary": {{
        "scene_type": "parking_lot|entrance|warehouse|outdoor|indoor|other",
        "overall_description": "detailed description of the scene",
        "visible_objects": ["object1", "object2", "..."],
        "vehicles": [
            {{"type": "car|truck|motorcycle", "location": "description", "color": "color", "status": "parked|moving"}}
        ],
        "people": [
            {{"count": number, "activity": "description", "location": "description"}}
        ],
        "structures": ["building", "fence", "gate", "..."],
        "lighting_assessment": "bright|dim|artificial|natural|mixed",
        "weather_conditions": "clear|cloudy|rain|snow|fog"
    }},
    "changes_detected": {{
        "has_changes": boolean,
        "change_summary": "description of what changed",
        "new_objects": ["list of new objects"],
        "moved_objects": ["list of moved objects"],
        "missing_objects": ["list of missing objects"],
        "significance": "minor|moderate|major|critical"
    }},
    "security_assessment": {{
        "threat_level": "none|low|medium|high|critical",
        "anomalies": ["list of unusual observations"],
        "recommendations": ["suggested actions or monitoring focus"]
    }},
    "temporal_notes": {{
        "time_of_day_factors": "how time affects the scene",
        "expected_activity": "what activity is normal for this time",
        "unusual_for_time": ["things that seem unusual for this time"]
    }}
}}

Provide detailed, accurate observations that will help build a comprehensive understanding of this scene over time.
"""
        return prompt
    
    def _process_scene_summary(self, scene_summary: Dict[str, Any]) -> None:
        """Process and store scene summary results"""
        try:
            summary_data = scene_summary.get("scene_summary", {})
            changes_data = scene_summary.get("changes_detected", {})
            security_data = scene_summary.get("security_assessment", {})
            
            # Create scene entry
            scene_entry = {
                "timestamp": datetime.now().isoformat(),
                "analysis_count": self._scene_analysis_data["analysis_count"],
                "scene_summary": summary_data,
                "changes_detected": changes_data,
                "security_assessment": security_data,
                "temporal_notes": scene_summary.get("temporal_notes", {})
            }
            
            # Store in scene history
            self._scene_analysis_data["scene_history"].append(scene_entry)
            
            # Keep only last 50 entries
            if len(self._scene_analysis_data["scene_history"]) > 50:
                self._scene_analysis_data["scene_history"] = self._scene_analysis_data["scene_history"][-50:]
            
            # Set baseline scene if not established
            if not self._scene_analysis_data.get("baseline_scene") and summary_data:
                self._scene_analysis_data["baseline_scene"] = summary_data
                logging.getLogger(__name__).info(f"Established baseline scene for camera {self.camera_id}")
            
            # Log significant changes
            if changes_data.get("has_changes") and changes_data.get("significance") in ["major", "critical"]:
                logging.getLogger(__name__).warning(
                    f"Significant scene change detected for camera {self.camera_id}: {changes_data.get('change_summary')}"
                )
            
            # Log security concerns
            threat_level = security_data.get("threat_level", "none")
            if threat_level in ["high", "critical"]:
                logging.getLogger(__name__).warning(
                    f"Security concern for camera {self.camera_id}: Threat level {threat_level}"
                )
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to process scene summary: {e}")
    
    def get_scene_analysis_status(self) -> Dict[str, Any]:
        """Get current scene analysis status"""
        return {
            "camera_id": self.camera_id,
            "scene_analysis_enabled": self.scene_analysis_enabled,
            "analysis_count": self._scene_analysis_data["analysis_count"],
            "last_analysis": self._scene_analysis_data.get("last_scene_analysis"),
            "next_analysis_in_seconds": (max(0,
                self._scene_analysis_data["scene_analysis_interval"] -
                (time.time() - (self._scene_analysis_data.get("last_scene_analysis") or 0))
            ) if self.scene_analysis_enabled else 0),
            "baseline_established": bool(self._scene_analysis_data.get("baseline_scene")),
            "total_scene_entries": len(self._scene_analysis_data["scene_history"]),
            "recent_changes": len([
                entry for entry in self._scene_analysis_data["scene_history"][-10:]
                if entry.get("changes_detected", {}).get("has_changes", False)
            ])
        }
    
    def get_scene_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent scene analysis history"""
        return self._scene_analysis_data["scene_history"][-limit:] if self._scene_analysis_data["scene_history"] else []
    
    def force_scene_analysis(self) -> None:
        """Force immediate scene analysis"""
        self._scene_analysis_data["last_scene_analysis"] = 0
        logging.getLogger(__name__).info(f"Forced scene analysis scheduled for camera {self.camera_id}")


# Keep the old MotionDetector for backward compatibility
class MotionDetector(SimpleMotionDetector):
    """
    Legacy motion detector - now just an alias for SimpleMotionDetector
    """
    def __init__(self, **kwargs):
        # Ignore all parameters - use the proven settings
        super().__init__()


