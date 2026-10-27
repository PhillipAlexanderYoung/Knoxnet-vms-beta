#!/usr/bin/env python3
"""
Detection Optimization System
Fixes inconsistent motion detection and false positive issues
"""

import logging
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)

@dataclass
class DetectionStats:
    """Track detection statistics for optimization"""
    camera_id: str
    total_detections: int = 0
    false_positives: int = 0
    missed_detections: int = 0
    last_detection: Optional[datetime] = None
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0

class DetectionOptimizer:
    """Optimizes motion detection settings based on performance"""
    
    def __init__(self, base_url: str = "http://localhost:5000/api"):
        self.base_url = base_url
        self.stats: Dict[str, DetectionStats] = {}
        self.optimization_running = False
        self.optimization_thread: Optional[threading.Thread] = None
        
        # Default optimization settings
        self.default_settings = {
            'sensitivity': 50,
            'threshold': 30,
            'min_area': 500,
            'max_area': 50000,
            'cooldown_period': 5,  # seconds
            'motion_zones': []
        }
        
    def start_optimization(self):
        """Start automatic detection optimization"""
        if self.optimization_running:
            logger.warning("Detection optimization already running")
            return
            
        self.optimization_running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        logger.info("🔧 Detection optimization started")
        
    def stop_optimization(self):
        """Stop automatic detection optimization"""
        self.optimization_running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        logger.info("🛑 Detection optimization stopped")
        
    def _optimization_loop(self):
        """Main optimization loop"""
        while self.optimization_running:
            try:
                self._analyze_all_cameras()
                self._apply_optimizations()
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(60)
                
    def _analyze_all_cameras(self):
        """Analyze detection performance for all cameras"""
        try:
            # Get all cameras
            resp = requests.get(f"{self.base_url}/cameras")
            cameras = resp.json()['data']
            
            for camera in cameras:
                camera_id = camera['id']
                camera_name = camera['name']
                
                # Initialize stats if needed
                if camera_id not in self.stats:
                    self.stats[camera_id] = DetectionStats(camera_id)
                
                # Analyze detection performance
                self._analyze_camera_detections(camera_id, camera_name)
                
        except Exception as e:
            logger.error(f"Error analyzing cameras: {e}")
            
    def _analyze_camera_detections(self, camera_id: str, camera_name: str):
        """Analyze detection performance for a specific camera"""
        try:
            # Get recent tracks
            tracks_resp = requests.get(f"{self.base_url}/cameras/{camera_id}/tracks")
            if tracks_resp.status_code != 200:
                logger.warning(f"Could not get tracks for {camera_name}")
                return
                
            tracks = tracks_resp.json().get('data', [])
            recent_tracks = self._get_recent_tracks(tracks, minutes=10)
            
            stats = self.stats[camera_id]
            stats.total_detections = len(recent_tracks)
            
            # Analyze detection patterns
            if recent_tracks:
                stats.last_detection = datetime.now()
                
                # Check for false positives (rapid successive detections)
                false_positives = self._detect_false_positives(recent_tracks)
                stats.false_positives = false_positives
                
                # Calculate rates
                stats.detection_rate = len(recent_tracks) / 10.0  # per minute
                stats.false_positive_rate = false_positives / len(recent_tracks) if recent_tracks else 0
                
                logger.debug(f"📊 {camera_name}: {len(recent_tracks)} detections, {false_positives} false positives")
                
        except Exception as e:
            logger.error(f"Error analyzing detections for {camera_name}: {e}")
            
    def _get_recent_tracks(self, tracks: List[dict], minutes: int = 10) -> List[dict]:
        """Get tracks from the last N minutes"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            recent_tracks = []
            
            for track in tracks:
                timestamp_str = track.get('timestamp')
                if timestamp_str:
                    try:
                        track_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if track_time > cutoff_time:
                            recent_tracks.append(track)
                    except:
                        continue
                        
            return recent_tracks
        except Exception as e:
            logger.error(f"Error getting recent tracks: {e}")
            return []
            
    def _detect_false_positives(self, tracks: List[dict]) -> int:
        """Detect potential false positives based on patterns"""
        if len(tracks) < 2:
            return 0
            
        false_positives = 0
        
        # Check for rapid successive detections (within 2 seconds)
        for i in range(1, len(tracks)):
            try:
                prev_time = datetime.fromisoformat(tracks[i-1]['timestamp'].replace('Z', '+00:00'))
                curr_time = datetime.fromisoformat(tracks[i]['timestamp'].replace('Z', '+00:00'))
                
                time_diff = (curr_time - prev_time).total_seconds()
                
                if time_diff < 2.0:  # Very rapid detections
                    false_positives += 1
                    
            except:
                continue
                
        return false_positives
        
    def _apply_optimizations(self):
        """Apply optimizations based on analysis"""
        for camera_id, stats in self.stats.items():
            try:
                # Get current settings
                settings_resp = requests.get(f"{self.base_url}/cameras/{camera_id}/motion/settings")
                if settings_resp.status_code != 200:
                    continue
                    
                current_settings = settings_resp.json().get('data', {})
                
                # Apply optimizations based on performance
                new_settings = self._calculate_optimal_settings(stats, current_settings)
                
                if new_settings != current_settings:
                    self._update_camera_settings(camera_id, new_settings)
                    
            except Exception as e:
                logger.error(f"Error applying optimizations for camera {camera_id}: {e}")
                
    def _calculate_optimal_settings(self, stats: DetectionStats, current_settings: dict) -> dict:
        """Calculate optimal settings based on detection statistics"""
        new_settings = current_settings.copy()
        
        # High false positive rate - increase threshold
        if stats.false_positive_rate > 0.3:
            current_threshold = current_settings.get('threshold', 30)
            new_threshold = min(current_threshold + 10, 80)
            new_settings['threshold'] = new_threshold
            logger.info(f"🔧 Increasing threshold to {new_threshold} due to high false positives")
            
        # Low detection rate - decrease threshold
        elif stats.detection_rate < 0.1 and stats.false_positive_rate < 0.1:
            current_threshold = current_settings.get('threshold', 30)
            new_threshold = max(current_threshold - 5, 10)
            new_settings['threshold'] = new_threshold
            logger.info(f"🔧 Decreasing threshold to {new_threshold} due to low detections")
            
        # High detection rate but good quality - fine-tune sensitivity
        elif stats.detection_rate > 2.0 and stats.false_positive_rate < 0.2:
            current_sensitivity = current_settings.get('sensitivity', 50)
            new_sensitivity = min(current_sensitivity + 5, 80)
            new_settings['sensitivity'] = new_sensitivity
            logger.info(f"🔧 Increasing sensitivity to {new_sensitivity} for better detection")
            
        return new_settings
        
    def _update_camera_settings(self, camera_id: str, settings: dict):
        """Update camera motion detection settings"""
        try:
            response = requests.put(
                f"{self.base_url}/cameras/{camera_id}/motion/settings",
                json=settings
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Updated settings for camera {camera_id}")
            else:
                logger.warning(f"⚠️ Failed to update settings for camera {camera_id}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error updating settings for camera {camera_id}: {e}")
            
    def get_optimization_report(self) -> dict:
        """Get optimization report for all cameras"""
        report = {
            'optimization_running': self.optimization_running,
            'cameras': {}
        }
        
        for camera_id, stats in self.stats.items():
            report['cameras'][camera_id] = {
                'total_detections': stats.total_detections,
                'false_positives': stats.false_positives,
                'detection_rate': stats.detection_rate,
                'false_positive_rate': stats.false_positive_rate,
                'last_detection': stats.last_detection.isoformat() if stats.last_detection else None
            }
            
        return report
        
    def reset_camera_settings(self, camera_id: str):
        """Reset camera to default settings"""
        try:
            response = requests.put(
                f"{self.base_url}/cameras/{camera_id}/motion/settings",
                json=self.default_settings
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Reset settings for camera {camera_id}")
            else:
                logger.warning(f"⚠️ Failed to reset settings for camera {camera_id}")
                
        except Exception as e:
            logger.error(f"Error resetting settings for camera {camera_id}: {e}")







