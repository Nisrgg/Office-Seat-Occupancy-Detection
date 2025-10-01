"""
Office Seat Occupancy Detection System - Main Application
========================================================

Main application script for running the seat occupancy detection system.
"""

import cv2
import argparse
import logging
import sys
from pathlib import Path
import time
import signal
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.models.seat_detector import SeatDetector
from src.analysis.occupancy_analyzer import OccupancyAnalyzer
from src.utils.plots import plot_detections, create_summary_dashboard, save_results_to_excel
from src.utils.datasets import VideoProcessor

logger = logging.getLogger(__name__)

class SeatOccupancyApp:
    """Main application class for seat occupancy detection"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the application
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.detector = SeatDetector(self.config)
        self.analyzer = OccupancyAnalyzer(self.config)
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("SeatOccupancyApp initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def process_video(self, video_path: str, output_dir: str = "output", 
                     save_frames: bool = False, display: bool = True) -> Dict[str, Any]:
        """
        Process a video file for seat occupancy detection
        
        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            save_frames: Whether to save processed frames
            display: Whether to display frames during processing
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing video: {video_path}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        frame_count = 0
        detections_data = []
        start_time = time.time()
        
        self.running = True
        
        try:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                result = self.detector.process_frame(frame)
                detections_data.append(result)
                
                # Create visualization
                if display or save_frames:
                    vis_frame = self._create_visualization(frame, result)
                    
                    if display:
                        cv2.imshow('Seat Occupancy Detection', vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    if save_frames and frame_count % self.config.visualization_config.get('frame_interval', 30) == 0:
                        frame_file = output_path / f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(frame_file), vis_frame)
                
                # Progress logging
                if frame_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    fps_current = frame_count / elapsed_time
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), "
                              f"FPS: {fps_current:.1f}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Calculate final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        # Get occupancy statistics
        occupancy_stats = self.detector.get_occupancy_statistics()
        
        # Perform analysis
        analysis_results = self.analyzer.analyze_occupancy_data(
            self.detector.occupancy_history, 
            occupancy_stats
        )
        
        # Save results
        self._save_results(output_path, detections_data, occupancy_stats, analysis_results)
        
        # Generate report
        report = self.analyzer.generate_report(analysis_results)
        report_file = output_path / "analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Processing completed. Results saved to {output_path}")
        
        return {
            'total_frames': frame_count,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'occupancy_stats': occupancy_stats,
            'analysis_results': analysis_results,
            'output_path': str(output_path)
        }
    
    def _create_visualization(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Create visualization frame with detections and tracking info"""
        vis_frame = frame.copy()
        
        # Draw detections
        detections = result['detections']
        class_names = self.config.detection_config['classes']
        
        for det in detections:
            bbox = det['bbox']
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            color = (0, 255, 0) if det['class_name'] == 'person' else (255, 0, 0)
            
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(vis_frame, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw tracking results
        tracking_results = result['tracking_results']
        for chair_id, tracking in tracking_results.items():
            bbox = tracking['bbox']
            status = "Occupied" if tracking['occupied'] else "Empty"
            color = (0, 0, 255) if tracking['occupied'] else (0, 255, 255)
            
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            cv2.putText(vis_frame, f"{chair_id}: {status}", 
                       (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add performance info
        perf = result['performance']
        info_text = f"FPS: {perf['fps']:.1f} | Frame: {result['frame_number']}"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def _save_results(self, output_path: Path, detections_data: List[Dict], 
                     occupancy_stats: Dict, analysis_results: Dict):
        """Save all results to files"""
        
        # Save detector results
        self.detector.save_results(str(output_path))
        
        # Save analysis results
        self.analyzer.export_analysis_results(analysis_results, str(output_path))
        
        # Save detailed detections
        detections_file = output_path / "detections_data.json"
        import json
        with open(detections_file, 'w') as f:
            json.dump(detections_data, f, indent=2, default=str)
        
        # Save to Excel
        excel_file = output_path / "occupancy_results.xlsx"
        save_results_to_excel(occupancy_stats, self.detector.occupancy_history, str(excel_file))
        
        # Create visualizations
        self._create_visualizations(output_path, occupancy_stats, analysis_results)
    
    def _create_visualizations(self, output_path: Path, occupancy_stats: Dict, 
                              analysis_results: Dict):
        """Create visualization plots"""
        try:
            import matplotlib.pyplot as plt
            
            # Timeline plot
            timeline_file = output_path / "occupancy_timeline.png"
            fig = plot_occupancy_timeline(self.detector.occupancy_history)
            fig.savefig(timeline_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Statistics plot
            stats_file = output_path / "occupancy_statistics.png"
            fig = plot_occupancy_statistics(occupancy_stats)
            fig.savefig(stats_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Dashboard
            dashboard_file = output_path / "analysis_dashboard.png"
            fig = create_summary_dashboard(self.detector.occupancy_history, occupancy_stats)
            fig.savefig(dashboard_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info("Visualizations created successfully")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping visualizations")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def process_realtime(self, camera_index: int = 0, output_dir: str = "output"):
        """
        Process real-time video from camera
        
        Args:
            camera_index: Camera index
            output_dir: Output directory for results
        """
        logger.info(f"Starting real-time processing from camera {camera_index}")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.running = True
        frame_count = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                result = self.detector.process_frame(frame)
                
                # Create visualization
                vis_frame = self._create_visualization(frame, result)
                
                # Display
                cv2.imshow('Real-time Seat Occupancy Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Save frame periodically
                if frame_count % 300 == 0:  # Save every 300 frames
                    frame_file = output_path / f"realtime_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_file), vis_frame)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Save final results
            occupancy_stats = self.detector.get_occupancy_statistics()
            analysis_results = self.analyzer.analyze_occupancy_data(
                self.detector.occupancy_history, occupancy_stats
            )
            
            self._save_results(output_path, [], occupancy_stats, analysis_results)
            
            logger.info(f"Real-time processing completed. Results saved to {output_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Office Seat Occupancy Detection System')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for real-time processing')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--save-frames', action='store_true', help='Save processed frames')
    parser.add_argument('--no-display', action='store_true', help='Disable frame display')
    parser.add_argument('--realtime', action='store_true', help='Process real-time camera feed')
    
    args = parser.parse_args()
    
    try:
        # Initialize application
        app = SeatOccupancyApp(args.config)
        
        if args.realtime:
            # Real-time processing
            app.process_realtime(args.camera, args.output)
        elif args.video:
            # Video file processing
            app.process_video(
                args.video, 
                args.output, 
                args.save_frames, 
                not args.no_display
            )
        else:
            print("Please specify either --video or --realtime")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
