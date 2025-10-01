"""
Seat Occupancy Detection System
==============================

Main detection system for analyzing office seat occupancy using computer vision.
"""

import cv2
import torch
import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json

from src.config import Config
from src.utils.general import non_max_suppression, scale_coords, letterbox
from src.utils.torch_utils import select_device, time_synchronized
from src.utils.plots import plot_detections, color_list

logger = logging.getLogger(__name__)

class SeatDetector:
    """Main seat detection and occupancy tracking class"""
    
    def __init__(self, config: Config):
        """
        Initialize seat detector
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = self._setup_device()
        self.model = self._load_model()
        self.class_names = config.detection_config['classes']
        
        # Tracking data structures
        self.chair_states = {}  # {chair_id: bool}
        self.chair_times = {}   # {chair_id: total_time}
        self.chair_start_times = {}  # {chair_id: start_time}
        self.occupancy_history = {}  # {chair_id: [(timestamp, occupied), ...]}
        
        # Performance monitoring
        self.frame_count = 0
        self.inference_times = []
        self.fps_history = []
        
        logger.info("SeatDetector initialized successfully")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        device_config = self.config.model_config.get('device', 'auto')
        if device_config == 'auto':
            device = select_device()
        else:
            device = select_device(device_config)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self) -> torch.nn.Module:
        """Load YOLOv5 model"""
        model_name = self.config.model_config['name']
        
        try:
            # Load model from torch hub
            model = torch.hub.load('ultralytics/yolov5', model_name)
            model.to(self.device)
            
            # Set model parameters
            model.conf = self.config.model_config['confidence_threshold']
            model.iou = self.config.model_config['iou_threshold']
            
            logger.info(f"Loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection dictionaries
        """
        start_time = time_synchronized()
        
        # Run inference
        with torch.no_grad():
            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()
        
        inference_time = time_synchronized() - start_time
        self.inference_times.append(inference_time)
        
        # Process detections
        processed_detections = []
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det
            
            # Filter by confidence and class
            if confidence >= self.config.model_config['confidence_threshold']:
                class_name = self.class_names[int(class_id)] if int(class_id) < len(self.class_names) else f"Class_{int(class_id)}"
                
                processed_detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'class_name': class_name
                })
        
        return processed_detections
    
    def filter_chairs(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter chair detections based on size criteria
        
        Args:
            detections: List of detections
            
        Returns:
            Filtered chair detections
        """
        chair_detections = []
        frame_area = self.config.model_config['input_size'][0] * self.config.model_config['input_size'][1]
        
        for det in detections:
            if det['class_name'] == 'chair':
                x1, y1, x2, y2 = det['bbox']
                box_area = (x2 - x1) * (y2 - y1)
                
                min_area = self.config.detection_config['min_chair_area'] * frame_area
                max_area = self.config.detection_config['max_chair_area'] * frame_area
                
                if min_area < box_area < max_area:
                    chair_detections.append(det)
        
        return chair_detections
    
    def track_chairs(self, chair_detections: List[Dict[str, Any]], 
                    person_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Track chair occupancy based on person-chair overlap
        
        Args:
            chair_detections: List of chair detections
            person_detections: List of person detections
            
        Returns:
            Dictionary with tracking results
        """
        current_time = time.time()
        tracking_results = {}
        
        # Process each chair
        for i, chair_det in enumerate(chair_detections):
            chair_id = f"chair_{i}"
            
            # Initialize chair tracking if not exists
            if chair_id not in self.chair_states:
                self.chair_states[chair_id] = False
                self.chair_times[chair_id] = 0.0
                self.chair_start_times[chair_id] = 0.0
                self.occupancy_history[chair_id] = []
            
            # Check for person-chair overlap
            chair_bbox = chair_det['bbox']
            person_detected_near_chair = False
            
            for person_det in person_detections:
                if self._check_overlap(chair_bbox, person_det['bbox']):
                    person_detected_near_chair = True
                    break
            
            # Update occupancy state
            was_occupied = self.chair_states[chair_id]
            is_occupied = person_detected_near_chair
            
            if is_occupied and not was_occupied:
                # Chair becomes occupied
                self.chair_states[chair_id] = True
                self.chair_start_times[chair_id] = current_time
                logger.debug(f"Chair {chair_id} became occupied")
                
            elif not is_occupied and was_occupied:
                # Chair becomes empty
                self.chair_states[chair_id] = False
                occupancy_duration = current_time - self.chair_start_times[chair_id]
                self.chair_times[chair_id] += occupancy_duration
                self.chair_start_times[chair_id] = 0.0
                logger.debug(f"Chair {chair_id} became empty after {occupancy_duration:.2f}s")
            
            # Record occupancy history
            self.occupancy_history[chair_id].append((current_time, is_occupied))
            
            # Store tracking result
            tracking_results[chair_id] = {
                'bbox': chair_bbox,
                'occupied': is_occupied,
                'confidence': chair_det['confidence'],
                'total_occupied_time': self.chair_times[chair_id],
                'current_session_start': self.chair_start_times[chair_id] if is_occupied else None
            }
        
        return tracking_results
    
    def _check_overlap(self, bbox1: List[int], bbox2: List[int]) -> bool:
        """
        Check if two bounding boxes overlap
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            True if boxes overlap
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Check for overlap
        overlap = (x1_1 < x2_2 and x2_1 > x1_2 and y1_1 < y2_2 and y2_1 > y1_2)
        
        if overlap:
            # Calculate IoU
            intersection_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - intersection_area
            
            iou = intersection_area / union_area if union_area > 0 else 0
            return iou >= self.config.detection_config['overlap_threshold']
        
        return False
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame for seat occupancy detection
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with processing results
        """
        self.frame_count += 1
        
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Separate chairs and persons
        chair_detections = [d for d in detections if d['class_name'] == 'chair']
        person_detections = [d for d in detections if d['class_name'] == 'person']
        
        # Filter chairs by size
        filtered_chairs = self.filter_chairs(chair_detections)
        
        # Track occupancy
        tracking_results = self.track_chairs(filtered_chairs, person_detections)
        
        # Calculate performance metrics
        current_fps = 1.0 / np.mean(self.inference_times[-10:]) if len(self.inference_times) >= 10 else 0
        self.fps_history.append(current_fps)
        
        return {
            'frame_number': self.frame_count,
            'timestamp': time.time(),
            'detections': detections,
            'chair_detections': filtered_chairs,
            'person_detections': person_detections,
            'tracking_results': tracking_results,
            'performance': {
                'fps': current_fps,
                'inference_time': self.inference_times[-1] if self.inference_times else 0,
                'avg_inference_time': np.mean(self.inference_times[-10:]) if len(self.inference_times) >= 10 else 0
            }
        }
    
    def get_occupancy_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate occupancy statistics for all chairs
        
        Returns:
            Dictionary with statistics for each chair
        """
        stats = {}
        current_time = time.time()
        
        for chair_id, history in self.occupancy_history.items():
            if not history:
                continue
            
            # Calculate statistics
            total_occupied_time = self.chair_times[chair_id]
            
            # Add current session if occupied
            if self.chair_states[chair_id] and self.chair_start_times[chair_id] > 0:
                total_occupied_time += current_time - self.chair_start_times[chair_id]
            
            # Calculate occupancy percentage
            total_time = current_time - history[0][0] if history else 0
            occupancy_percentage = (total_occupied_time / total_time * 100) if total_time > 0 else 0
            
            # Count occupancy events
            num_events = 0
            prev_occupied = None
            for _, occupied in history:
                if prev_occupied is not None and occupied != prev_occupied:
                    num_events += 1
                prev_occupied = occupied
            
            # Calculate average occupancy duration
            occupancy_durations = []
            session_start = None
            
            for timestamp, occupied in history:
                if occupied and session_start is None:
                    session_start = timestamp
                elif not occupied and session_start is not None:
                    occupancy_durations.append(timestamp - session_start)
                    session_start = None
            
            # Handle ongoing session
            if session_start is not None:
                occupancy_durations.append(current_time - session_start)
            
            avg_duration = np.mean(occupancy_durations) if occupancy_durations else 0
            
            stats[chair_id] = {
                'total_occupied_time': total_occupied_time,
                'occupancy_percentage': occupancy_percentage,
                'num_occupancy_events': num_events,
                'avg_occupancy_duration': avg_duration,
                'current_status': 'occupied' if self.chair_states[chair_id] else 'empty',
                'total_sessions': len(occupancy_durations)
            }
        
        return stats
    
    def save_results(self, output_path: str):
        """
        Save detection and tracking results
        
        Args:
            output_path: Path to save results
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save occupancy statistics
        stats = self.get_occupancy_statistics()
        stats_file = output_dir / 'occupancy_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save detailed history
        history_file = output_dir / 'occupancy_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.occupancy_history, f, indent=2)
        
        # Save performance metrics
        performance_file = output_dir / 'performance_metrics.json'
        performance_data = {
            'total_frames': self.frame_count,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'total_processing_time': sum(self.inference_times) if self.inference_times else 0
        }
        with open(performance_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.chair_states.clear()
        self.chair_times.clear()
        self.chair_start_times.clear()
        self.occupancy_history.clear()
        self.frame_count = 0
        self.inference_times.clear()
        self.fps_history.clear()
        logger.info("Tracking data reset")
