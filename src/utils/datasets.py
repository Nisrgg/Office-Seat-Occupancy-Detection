"""
Dataset Utilities
=================

Utilities for loading and processing video datasets for seat occupancy detection.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    """Dataset class for video files"""
    
    def __init__(self, video_paths: List[str], transform=None, max_frames: Optional[int] = None):
        """
        Initialize video dataset
        
        Args:
            video_paths: List of video file paths
            transform: Optional transform to be applied on frames
            max_frames: Maximum number of frames to process per video
        """
        self.video_paths = video_paths
        self.transform = transform
        self.max_frames = max_frames
        self.frame_data = self._prepare_frame_data()
    
    def _prepare_frame_data(self) -> List[Dict[str, Any]]:
        """Prepare frame data for all videos"""
        frame_data = []
        
        for video_idx, video_path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                continue
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Limit frames if max_frames is specified
            if self.max_frames and frame_count > self.max_frames:
                frame_count = self.max_frames
            
            for frame_idx in range(frame_count):
                frame_data.append({
                    'video_idx': video_idx,
                    'video_path': video_path,
                    'frame_idx': frame_idx,
                    'total_frames': frame_count,
                    'fps': fps
                })
            
            cap.release()
        
        return frame_data
    
    def __len__(self):
        return len(self.frame_data)
    
    def __getitem__(self, idx):
        """Get frame at index"""
        frame_info = self.frame_data[idx]
        
        # Load frame
        cap = cv2.VideoCapture(frame_info['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_info['frame_idx'])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.warning(f"Could not read frame {frame_info['frame_idx']} from {frame_info['video_path']}")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Apply transform if provided
        if self.transform:
            frame = self.transform(frame)
        
        return {
            'frame': frame,
            'frame_info': frame_info
        }

class SeatOccupancyDataset(Dataset):
    """Dataset class for seat occupancy data with annotations"""
    
    def __init__(self, data_path: str, annotations_path: Optional[str] = None):
        """
        Initialize seat occupancy dataset
        
        Args:
            data_path: Path to data directory
            annotations_path: Path to annotations file
        """
        self.data_path = Path(data_path)
        self.annotations_path = annotations_path
        self.annotations = self._load_annotations()
        self.samples = self._prepare_samples()
    
    def _load_annotations(self) -> Dict[str, Any]:
        """Load annotations from file"""
        if not self.annotations_path or not Path(self.annotations_path).exists():
            return {}
        
        try:
            with open(self.annotations_path, 'r') as f:
                if self.annotations_path.endswith('.json'):
                    return json.load(f)
                else:
                    # Assume CSV format
                    import pandas as pd
                    df = pd.read_csv(self.annotations_path)
                    return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            return {}
    
    def _prepare_samples(self) -> List[Dict[str, Any]]:
        """Prepare samples from data directory"""
        samples = []
        
        # Look for video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        for ext in video_extensions:
            video_files = list(self.data_path.glob(f'**/*{ext}'))
            for video_file in video_files:
                samples.append({
                    'type': 'video',
                    'path': str(video_file),
                    'name': video_file.stem
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get sample at index"""
        sample = self.samples[idx]
        
        if sample['type'] == 'video':
            return self._load_video_sample(sample)
        else:
            raise ValueError(f"Unknown sample type: {sample['type']}")
    
    def _load_video_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Load video sample"""
        cap = cv2.VideoCapture(sample['path'])
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        return {
            'frames': frames,
            'sample_info': sample,
            'annotations': self.annotations.get(sample['name'], {})
        }

def load_images_and_labels(data_path: str, img_size: int = 640, augment: bool = False):
    """
    Load images and labels from directory
    
    Args:
        data_path: Path to data directory
        img_size: Target image size
        augment: Whether to apply augmentation
    
    Returns:
        Dataset object
    """
    return SeatOccupancyDataset(data_path)

def create_data_loader(dataset: Dataset, batch_size: int = 1, shuffle: bool = False, 
                      num_workers: int = 0, pin_memory: bool = True) -> DataLoader:
    """
    Create data loader for dataset
    
    Args:
        dataset: Dataset object
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
    
    Returns:
        DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn if hasattr(dataset, 'collate_fn') else None
    )

def collate_fn(batch):
    """Custom collate function for batching"""
    if isinstance(batch[0], dict):
        # Handle dictionary batches
        keys = batch[0].keys()
        return {key: [item[key] for item in batch] for key in keys}
    else:
        # Handle tensor batches
        return torch.stack(batch)

class VideoProcessor:
    """Video processing utilities"""
    
    def __init__(self, video_path: str):
        """
        Initialize video processor
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get specific frame from video
        
        Args:
            frame_number: Frame number to retrieve
        
        Returns:
            Frame as numpy array or None if failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            return frame
        else:
            return None
    
    def get_frames(self, start_frame: int = 0, end_frame: Optional[int] = None) -> List[np.ndarray]:
        """
        Get range of frames from video
        
        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number (None for end of video)
        
        Returns:
            List of frames
        """
        if end_frame is None:
            end_frame = self.frame_count
        
        frames = []
        for frame_num in range(start_frame, min(end_frame, self.frame_count)):
            frame = self.get_frame(frame_num)
            if frame is not None:
                frames.append(frame)
        
        return frames
    
    def save_frames(self, output_dir: str, frame_interval: int = 1, 
                   start_frame: int = 0, end_frame: Optional[int] = None):
        """
        Save frames to directory
        
        Args:
            output_dir: Output directory
            frame_interval: Save every Nth frame
            start_frame: Starting frame
            end_frame: Ending frame
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if end_frame is None:
            end_frame = self.frame_count
        
        saved_count = 0
        for frame_num in range(start_frame, min(end_frame, self.frame_count), frame_interval):
            frame = self.get_frame(frame_num)
            if frame is not None:
                timestamp = frame_num / self.fps
                filename = f"frame_{frame_num:06d}_t{timestamp:.2f}s.jpg"
                cv2.imwrite(str(output_path / filename), frame)
                saved_count += 1
        
        logger.info(f"Saved {saved_count} frames to {output_dir}")
    
    def get_video_info(self) -> Dict[str, Any]:
        """Get video information"""
        return {
            'path': self.video_path,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration': self.duration,
            'duration_formatted': str(datetime.timedelta(seconds=self.duration))
        }
    
    def __del__(self):
        """Cleanup video capture"""
        if hasattr(self, 'cap'):
            self.cap.release()

def extract_frames_from_video(video_path: str, output_dir: str, 
                            frame_interval: int = 30, max_frames: Optional[int] = None):
    """
    Extract frames from video file
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        frame_interval: Extract every Nth frame
        max_frames: Maximum number of frames to extract
    """
    processor = VideoProcessor(video_path)
    processor.save_frames(output_dir, frame_interval)
    
    if max_frames:
        # Limit number of saved frames
        saved_frames = list(Path(output_dir).glob("*.jpg"))
        if len(saved_frames) > max_frames:
            for frame_file in saved_frames[max_frames:]:
                frame_file.unlink()
    
    return processor.get_video_info()
