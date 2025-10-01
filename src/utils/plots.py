"""
Visualization Utilities
=======================

Utilities for plotting, visualization, and result display.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

# Color palette for different classes
COLORS = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (128, 0, 128),   # Purple
    (255, 165, 0),   # Orange
    (0, 128, 0),     # Dark Green
    (128, 128, 128)  # Gray
]

def color_list():
    """Return list of colors for visualization"""
    return COLORS

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    Plot one bounding box on image
    
    Args:
        x: Bounding box coordinates [x1, y1, x2, y2]
        img: Image array
        color: Box color
        label: Box label
        line_thickness: Line thickness
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_detections(image: np.ndarray, detections: List[Dict], 
                   class_names: List[str], confidence_threshold: float = 0.5) -> np.ndarray:
    """
    Plot detections on image
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        class_names: List of class names
        confidence_threshold: Confidence threshold for display
    
    Returns:
        Image with detections plotted
    """
    img = image.copy()
    
    for det in detections:
        if det['confidence'] < confidence_threshold:
            continue
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = det['bbox']
        
        # Get class name and confidence
        class_name = class_names[det['class_id']] if det['class_id'] < len(class_names) else f"Class_{det['class_id']}"
        label = f"{class_name}: {det['confidence']:.2f}"
        
        # Choose color based on class
        color = COLORS[det['class_id'] % len(COLORS)]
        
        # Plot bounding box
        plot_one_box([x1, y1, x2, y2], img, color=color, label=label)
    
    return img

def plot_occupancy_timeline(occupancy_data: Dict[str, List], 
                           time_window_minutes: int = 60,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot occupancy timeline for chairs
    
    Args:
        occupancy_data: Dictionary with chair occupancy data
        time_window_minutes: Time window for plot
        save_path: Path to save plot
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Prepare data for plotting
    chair_ids = list(occupancy_data.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(chair_ids)))
    
    for i, (chair_id, data) in enumerate(occupancy_data.items()):
        if not data:
            continue
        
        # Convert timestamps to datetime objects
        timestamps = [datetime.fromtimestamp(ts) for ts, occupied in data]
        occupied_status = [1 if occupied else 0 for ts, occupied in data]
        
        # Plot occupancy status
        ax.plot(timestamps, occupied_status, 
               label=f'Chair {chair_id}', 
               color=colors[i], 
               linewidth=2, 
               marker='o', 
               markersize=4)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Occupancy Status')
    ax.set_title('Chair Occupancy Timeline')
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Empty', 'Occupied'])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Timeline plot saved to {save_path}")
    
    return fig

def plot_occupancy_statistics(occupancy_stats: Dict[str, Dict], 
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot occupancy statistics
    
    Args:
        occupancy_stats: Dictionary with occupancy statistics
        save_path: Path to save plot
    
    Returns:
        Matplotlib figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    chair_ids = list(occupancy_stats.keys())
    
    # 1. Total occupancy time
    total_times = [stats['total_occupied_time'] for stats in occupancy_stats.values()]
    ax1.bar(chair_ids, total_times, color='skyblue', alpha=0.7)
    ax1.set_title('Total Occupied Time by Chair')
    ax1.set_xlabel('Chair ID')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Occupancy percentage
    occupancy_percentages = [stats['occupancy_percentage'] for stats in occupancy_stats.values()]
    ax2.bar(chair_ids, occupancy_percentages, color='lightcoral', alpha=0.7)
    ax2.set_title('Occupancy Percentage by Chair')
    ax2.set_xlabel('Chair ID')
    ax2.set_ylabel('Percentage (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Number of occupancy events
    num_events = [stats['num_occupancy_events'] for stats in occupancy_stats.values()]
    ax3.bar(chair_ids, num_events, color='lightgreen', alpha=0.7)
    ax3.set_title('Number of Occupancy Events by Chair')
    ax3.set_xlabel('Chair ID')
    ax3.set_ylabel('Number of Events')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Average occupancy duration
    avg_durations = [stats['avg_occupancy_duration'] for stats in occupancy_stats.values()]
    ax4.bar(chair_ids, avg_durations, color='gold', alpha=0.7)
    ax4.set_title('Average Occupancy Duration by Chair')
    ax4.set_xlabel('Chair ID')
    ax4.set_ylabel('Duration (seconds)')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Statistics plot saved to {save_path}")
    
    return fig

def plot_heatmap(occupancy_matrix: np.ndarray, 
                chair_ids: List[str], 
                time_labels: List[str],
                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot occupancy heatmap
    
    Args:
        occupancy_matrix: 2D array with occupancy data
        chair_ids: List of chair IDs
        time_labels: List of time labels
        save_path: Path to save plot
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(occupancy_matrix, 
                xticklabels=time_labels,
                yticklabels=chair_ids,
                cmap='RdYlGn',
                cbar_kws={'label': 'Occupancy Status'},
                ax=ax)
    
    ax.set_title('Chair Occupancy Heatmap')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chair ID')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to {save_path}")
    
    return fig

def create_summary_dashboard(occupancy_data: Dict[str, List], 
                           occupancy_stats: Dict[str, Dict],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive summary dashboard
    
    Args:
        occupancy_data: Raw occupancy data
        occupancy_stats: Processed statistics
        save_path: Path to save dashboard
    
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Timeline plot (top row, spans 3 columns)
    ax1 = fig.add_subplot(gs[0, :3])
    plot_occupancy_timeline(occupancy_data, ax=ax1)
    
    # 2. Summary statistics (top right)
    ax2 = fig.add_subplot(gs[0, 3])
    total_chairs = len(occupancy_stats)
    total_time = sum(stats['total_occupied_time'] for stats in occupancy_stats.values())
    avg_occupancy = np.mean([stats['occupancy_percentage'] for stats in occupancy_stats.values()])
    
    ax2.text(0.1, 0.8, f"Total Chairs: {total_chairs}", fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.6, f"Total Occupied Time: {total_time:.1f}s", fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.4, f"Avg Occupancy: {avg_occupancy:.1f}%", fontsize=12, transform=ax2.transAxes)
    ax2.set_title('Summary Statistics')
    ax2.axis('off')
    
    # 3. Occupancy distribution (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    occupancy_percentages = [stats['occupancy_percentage'] for stats in occupancy_stats.values()]
    ax3.hist(occupancy_percentages, bins=10, alpha=0.7, color='skyblue')
    ax3.set_title('Occupancy Distribution')
    ax3.set_xlabel('Occupancy %')
    ax3.set_ylabel('Frequency')
    
    # 4. Peak hours analysis (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    # This would require time-based analysis of occupancy data
    ax4.text(0.5, 0.5, 'Peak Hours\nAnalysis\n(To be implemented)', 
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Peak Hours')
    ax4.axis('off')
    
    # 5. Utilization efficiency (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    utilization = [stats['occupancy_percentage'] for stats in occupancy_stats.values()]
    ax5.pie(utilization, labels=list(occupancy_stats.keys()), autopct='%1.1f%%')
    ax5.set_title('Utilization Distribution')
    
    # 6. Time series analysis (bottom row, spans 4 columns)
    ax6 = fig.add_subplot(gs[2, :])
    # Aggregate occupancy over time
    all_timestamps = []
    for data in occupancy_data.values():
        all_timestamps.extend([ts for ts, occupied in data])
    
    if all_timestamps:
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        time_bins = np.linspace(min_time, max_time, 50)
        
        # Calculate occupancy rate for each time bin
        occupancy_rates = []
        for i in range(len(time_bins) - 1):
            bin_start, bin_end = time_bins[i], time_bins[i + 1]
            occupied_in_bin = 0
            total_in_bin = 0
            
            for chair_data in occupancy_data.values():
                for ts, occupied in chair_data:
                    if bin_start <= ts <= bin_end:
                        total_in_bin += 1
                        if occupied:
                            occupied_in_bin += 1
            
            rate = occupied_in_bin / total_in_bin if total_in_bin > 0 else 0
            occupancy_rates.append(rate)
        
        bin_centers = [(time_bins[i] + time_bins[i + 1]) / 2 for i in range(len(time_bins) - 1)]
        bin_labels = [datetime.fromtimestamp(ts).strftime('%H:%M') for ts in bin_centers]
        
        ax6.plot(bin_labels, occupancy_rates, marker='o', linewidth=2)
        ax6.set_title('Overall Occupancy Rate Over Time')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Occupancy Rate')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Office Seat Occupancy Analysis Dashboard', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {save_path}")
    
    return fig

def save_results_to_excel(occupancy_stats: Dict[str, Dict], 
                         occupancy_data: Dict[str, List],
                         output_path: str):
    """
    Save results to Excel file
    
    Args:
        occupancy_stats: Processed statistics
        occupancy_data: Raw occupancy data
        output_path: Output file path
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary statistics sheet
        stats_df = pd.DataFrame(occupancy_stats).T
        stats_df.to_excel(writer, sheet_name='Summary Statistics')
        
        # Detailed data sheet
        detailed_data = []
        for chair_id, data in occupancy_data.items():
            for timestamp, occupied in data:
                detailed_data.append({
                    'Chair_ID': chair_id,
                    'Timestamp': datetime.fromtimestamp(timestamp),
                    'Occupied': occupied
                })
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Detailed Data', index=False)
    
    logger.info(f"Results saved to Excel file: {output_path}")

def create_video_with_detections(input_video_path: str, 
                                detections_data: List[Dict],
                                output_video_path: str,
                                class_names: List[str] = None):
    """
    Create output video with detection overlays
    
    Args:
        input_video_path: Path to input video
        detections_data: List of detection data per frame
        output_video_path: Path to output video
        class_names: List of class names
    """
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add detections for this frame
        if frame_idx < len(detections_data):
            frame_detections = detections_data[frame_idx]
            frame = plot_detections(frame, frame_detections, class_names or [])
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    logger.info(f"Output video saved to: {output_video_path}")
