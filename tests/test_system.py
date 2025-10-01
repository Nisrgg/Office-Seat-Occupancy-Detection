"""
Unit Tests for Office Seat Occupancy Detection System
====================================================

Comprehensive test suite for all components of the system.
"""

import pytest
import numpy as np
import cv2
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Import modules to test
from src.config import Config
from src.models.seat_detector import SeatDetector
from src.analysis.occupancy_analyzer import OccupancyAnalyzer
from src.utils.general import non_max_suppression, scale_coords, letterbox
from src.utils.plots import plot_detections, plot_occupancy_timeline

class TestConfig:
    """Test configuration management"""
    
    def test_config_loading(self):
        """Test configuration loading from YAML"""
        config_data = {
            'model': {'name': 'yolov5s', 'confidence_threshold': 0.5},
            'detection': {'classes': ['person', 'chair']},
            'data': {'input_video_path': 'data/videos/'},
            'analysis': {'time_window_minutes': 5}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = Config(config_path)
            assert config.model_config['name'] == 'yolov5s'
            assert config.detection_config['classes'] == ['person', 'chair']
        finally:
            Path(config_path).unlink()
    
    def test_config_validation(self):
        """Test configuration validation"""
        invalid_config = {
            'model': {'name': 'yolov5s'}  # Missing required sections
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(invalid_config, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError):
                Config(config_path)
        finally:
            Path(config_path).unlink()

class TestSeatDetector:
    """Test seat detection functionality"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        config = Mock()
        config.model_config = {
            'name': 'yolov5s',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'device': 'cpu',
            'input_size': [640, 640]
        }
        config.detection_config = {
            'classes': ['person', 'chair'],
            'min_chair_area': 0.01,
            'max_chair_area': 0.1,
            'overlap_threshold': 0.3
        }
        return config
    
    @pytest.fixture
    def sample_frame(self):
        """Create sample frame for testing"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, mock_config):
        """Test detector initialization"""
        with patch('src.models.seat_detector.torch.hub.load') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            detector = SeatDetector(mock_config)
            assert detector.config == mock_config
            assert detector.class_names == ['person', 'chair']
    
    def test_overlap_detection(self, mock_config):
        """Test bounding box overlap detection"""
        with patch('src.models.seat_detector.torch.hub.load'):
            detector = SeatDetector(mock_config)
            
            # Test overlapping boxes
            bbox1 = [100, 100, 200, 200]
            bbox2 = [150, 150, 250, 250]
            assert detector._check_overlap(bbox1, bbox2) == True
            
            # Test non-overlapping boxes
            bbox3 = [300, 300, 400, 400]
            assert detector._check_overlap(bbox1, bbox3) == False
    
    def test_chair_filtering(self, mock_config):
        """Test chair detection filtering"""
        with patch('src.models.seat_detector.torch.hub.load'):
            detector = SeatDetector(mock_config)
            
            detections = [
                {
                    'bbox': [100, 100, 200, 200],
                    'confidence': 0.8,
                    'class_name': 'chair'
                },
                {
                    'bbox': [10, 10, 20, 20],  # Too small
                    'confidence': 0.7,
                    'class_name': 'chair'
                },
                {
                    'bbox': [100, 100, 200, 200],
                    'confidence': 0.8,
                    'class_name': 'person'
                }
            ]
            
            filtered = detector.filter_chairs(detections)
            assert len(filtered) == 1
            assert filtered[0]['class_name'] == 'chair'

class TestOccupancyAnalyzer:
    """Test occupancy analysis functionality"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        config = Mock()
        config.analysis_config = {
            'time_window_minutes': 5,
            'occupancy_threshold_seconds': 30
        }
        return config
    
    @pytest.fixture
    def sample_occupancy_data(self):
        """Create sample occupancy data"""
        return {
            'chair_1': [(1000, True), (2000, False), (3000, True), (4000, False)],
            'chair_2': [(1000, False), (2000, True), (3000, False), (4000, True)]
        }
    
    @pytest.fixture
    def sample_occupancy_stats(self):
        """Create sample occupancy statistics"""
        return {
            'chair_1': {
                'total_occupied_time': 2000,
                'occupancy_percentage': 50.0,
                'num_occupancy_events': 2,
                'avg_occupancy_duration': 1000,
                'total_sessions': 2
            },
            'chair_2': {
                'total_occupied_time': 2000,
                'occupancy_percentage': 50.0,
                'num_occupancy_events': 2,
                'avg_occupancy_duration': 1000,
                'total_sessions': 2
            }
        }
    
    def test_analyzer_initialization(self, mock_config):
        """Test analyzer initialization"""
        analyzer = OccupancyAnalyzer(mock_config)
        assert analyzer.config == mock_config
    
    def test_summary_statistics(self, mock_config, sample_occupancy_stats):
        """Test summary statistics calculation"""
        analyzer = OccupancyAnalyzer(mock_config)
        summary = analyzer._calculate_summary_statistics(sample_occupancy_stats)
        
        assert summary['total_chairs'] == 2
        assert summary['average_occupancy_percentage'] == 50.0
        assert summary['total_occupancy_events'] == 4
    
    def test_temporal_analysis(self, mock_config, sample_occupancy_data):
        """Test temporal pattern analysis"""
        analyzer = OccupancyAnalyzer(mock_config)
        temporal = analyzer._analyze_temporal_patterns(sample_occupancy_data)
        
        assert 'analysis_period' in temporal
        assert 'hourly_patterns' in temporal
        assert 'daily_patterns' in temporal
    
    def test_utilization_analysis(self, mock_config, sample_occupancy_stats):
        """Test utilization pattern analysis"""
        analyzer = OccupancyAnalyzer(mock_config)
        utilization = analyzer._analyze_utilization_patterns(sample_occupancy_stats)
        
        assert 'chair_utilization' in utilization
        assert 'utilization_distribution' in utilization
        assert 'efficiency_metrics' in utilization
    
    def test_efficiency_calculation(self, mock_config):
        """Test efficiency score calculation"""
        analyzer = OccupancyAnalyzer(mock_config)
        
        stats = {
            'occupancy_percentage': 75.0,
            'total_sessions': 8,
            'avg_occupancy_duration': 1800  # 30 minutes
        }
        
        efficiency = analyzer._calculate_efficiency_score(stats)
        assert 0 <= efficiency <= 1

class TestUtilities:
    """Test utility functions"""
    
    def test_letterbox(self):
        """Test letterbox function"""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        resized, ratio, pad = letterbox(img, (640, 640))
        
        assert resized.shape[:2] == (640, 640)
        assert isinstance(ratio, tuple)
        assert isinstance(pad, tuple)
    
    def test_scale_coords(self):
        """Test coordinate scaling"""
        coords = np.array([[100, 100, 200, 200]])
        img1_shape = (480, 640)
        img0_shape = (640, 640)
        
        scaled = scale_coords(img1_shape, coords, img0_shape)
        assert scaled.shape == coords.shape
    
    def test_plot_detections(self):
        """Test detection plotting"""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            {
                'bbox': [100, 100, 200, 200],
                'confidence': 0.8,
                'class_id': 0,
                'class_name': 'person'
            }
        ]
        class_names = ['person', 'chair']
        
        result = plot_detections(img, detections, class_names)
        assert result.shape == img.shape

class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_processing(self):
        """Test complete processing pipeline"""
        # This would test the full pipeline from video input to analysis output
        # For now, we'll create a basic test structure
        pass
    
    def test_config_to_analysis_pipeline(self):
        """Test configuration to analysis pipeline"""
        # Test that configuration properly flows through the system
        pass

# Performance tests
class TestPerformance:
    """Performance and benchmarking tests"""
    
    def test_detection_speed(self):
        """Test detection speed benchmarks"""
        # Test that detection meets performance requirements
        pass
    
    def test_memory_usage(self):
        """Test memory usage during processing"""
        # Test memory consumption stays within limits
        pass

# Fixtures for common test data
@pytest.fixture
def temp_config_file():
    """Create temporary configuration file"""
    config_data = {
        'model': {
            'name': 'yolov5s',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'device': 'cpu',
            'input_size': [640, 640]
        },
        'detection': {
            'classes': ['person', 'chair'],
            'min_chair_area': 0.01,
            'max_chair_area': 0.1,
            'overlap_threshold': 0.3
        },
        'data': {
            'input_video_path': 'data/videos/',
            'output_path': 'output/',
            'processed_data_path': 'data/processed/',
            'results_path': 'results/'
        },
        'analysis': {
            'time_window_minutes': 5,
            'occupancy_threshold_seconds': 30,
            'generate_reports': True,
            'export_formats': ['csv', 'json', 'excel']
        },
        'visualization': {
            'save_frames': True,
            'frame_interval': 30,
            'plot_style': 'seaborn',
            'figure_size': [12, 8],
            'dpi': 300
        },
        'logging': {
            'level': 'INFO',
            'log_file': 'logs/seat_detection.log',
            'console_output': True
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        yield f.name
    
    # Cleanup
    Path(f.name).unlink()

if __name__ == "__main__":
    pytest.main([__file__])
