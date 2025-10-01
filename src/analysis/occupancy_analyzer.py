"""
Data Analysis Module
====================

Comprehensive data analysis and statistical processing for seat occupancy data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class OccupancyAnalyzer:
    """Analyze seat occupancy data and generate insights"""
    
    def __init__(self, config):
        """
        Initialize occupancy analyzer
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.analysis_config = config.analysis_config
        logger.info("OccupancyAnalyzer initialized")
    
    def analyze_occupancy_data(self, occupancy_history: Dict[str, List], 
                             occupancy_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of occupancy data
        
        Args:
            occupancy_history: Raw occupancy history data
            occupancy_stats: Processed occupancy statistics
            
        Returns:
            Dictionary with analysis results
        """
        analysis_results = {
            'summary_statistics': self._calculate_summary_statistics(occupancy_stats),
            'temporal_analysis': self._analyze_temporal_patterns(occupancy_history),
            'utilization_analysis': self._analyze_utilization_patterns(occupancy_stats),
            'efficiency_metrics': self._calculate_efficiency_metrics(occupancy_stats),
            'peak_hours_analysis': self._analyze_peak_hours(occupancy_history),
            'correlation_analysis': self._analyze_correlations(occupancy_history)
        }
        
        return analysis_results
    
    def _calculate_summary_statistics(self, occupancy_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate overall summary statistics"""
        if not occupancy_stats:
            return {}
        
        total_chairs = len(occupancy_stats)
        total_occupied_time = sum(stats['total_occupied_time'] for stats in occupancy_stats.values())
        avg_occupancy_percentage = np.mean([stats['occupancy_percentage'] for stats in occupancy_stats.values()])
        total_events = sum(stats['num_occupancy_events'] for stats in occupancy_stats.values())
        
        # Calculate utilization distribution
        occupancy_percentages = [stats['occupancy_percentage'] for stats in occupancy_stats.values()]
        utilization_distribution = {
            'high_utilization': len([p for p in occupancy_percentages if p > 70]),
            'medium_utilization': len([p for p in occupancy_percentages if 30 <= p <= 70]),
            'low_utilization': len([p for p in occupancy_percentages if p < 30])
        }
        
        return {
            'total_chairs': total_chairs,
            'total_occupied_time': total_occupied_time,
            'average_occupancy_percentage': avg_occupancy_percentage,
            'total_occupancy_events': total_events,
            'utilization_distribution': utilization_distribution,
            'most_utilized_chair': max(occupancy_stats.items(), key=lambda x: x[1]['occupancy_percentage'])[0] if occupancy_stats else None,
            'least_utilized_chair': min(occupancy_stats.items(), key=lambda x: x[1]['occupancy_percentage'])[0] if occupancy_stats else None
        }
    
    def _analyze_temporal_patterns(self, occupancy_history: Dict[str, List]) -> Dict[str, Any]:
        """Analyze temporal patterns in occupancy data"""
        if not occupancy_history:
            return {}
        
        # Aggregate all timestamps
        all_timestamps = []
        for chair_history in occupancy_history.values():
            all_timestamps.extend([timestamp for timestamp, _ in chair_history])
        
        if not all_timestamps:
            return {}
        
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        total_duration = max_time - min_time
        
        # Analyze hourly patterns
        hourly_occupancy = self._calculate_hourly_occupancy(occupancy_history)
        
        # Analyze daily patterns
        daily_patterns = self._calculate_daily_patterns(occupancy_history)
        
        return {
            'analysis_period': {
                'start_time': datetime.fromtimestamp(min_time).isoformat(),
                'end_time': datetime.fromtimestamp(max_time).isoformat(),
                'total_duration_hours': total_duration / 3600
            },
            'hourly_patterns': hourly_occupancy,
            'daily_patterns': daily_patterns,
            'temporal_trends': self._detect_temporal_trends(occupancy_history)
        }
    
    def _calculate_hourly_occupancy(self, occupancy_history: Dict[str, List]) -> Dict[str, Any]:
        """Calculate hourly occupancy patterns"""
        hourly_data = {}
        
        for hour in range(24):
            hourly_data[str(hour)] = {
                'total_occupancy_time': 0,
                'num_sessions': 0,
                'avg_session_duration': 0
            }
        
        for chair_id, history in occupancy_history.items():
            if not history:
                continue
            
            # Group by hour
            hour_sessions = {}
            current_session_start = None
            
            for timestamp, occupied in history:
                hour = datetime.fromtimestamp(timestamp).hour
                
                if occupied and current_session_start is None:
                    current_session_start = timestamp
                elif not occupied and current_session_start is not None:
                    session_duration = timestamp - current_session_start
                    if hour not in hour_sessions:
                        hour_sessions[hour] = []
                    hour_sessions[hour].append(session_duration)
                    current_session_start = None
            
            # Handle ongoing session
            if current_session_start is not None:
                session_duration = time.time() - current_session_start
                hour = datetime.fromtimestamp(current_session_start).hour
                if hour not in hour_sessions:
                    hour_sessions[hour] = []
                hour_sessions[hour].append(session_duration)
            
            # Aggregate data
            for hour, sessions in hour_sessions.items():
                hourly_data[str(hour)]['total_occupancy_time'] += sum(sessions)
                hourly_data[str(hour)]['num_sessions'] += len(sessions)
                hourly_data[str(hour)]['avg_session_duration'] = np.mean(sessions) if sessions else 0
        
        return hourly_data
    
    def _calculate_daily_patterns(self, occupancy_history: Dict[str, List]) -> Dict[str, Any]:
        """Calculate daily occupancy patterns"""
        daily_data = {}
        
        for chair_id, history in occupancy_history.items():
            if not history:
                continue
            
            for timestamp, occupied in history:
                date = datetime.fromtimestamp(timestamp).date()
                date_str = date.isoformat()
                
                if date_str not in daily_data:
                    daily_data[date_str] = {
                        'total_chairs': 0,
                        'total_occupancy_time': 0,
                        'peak_hour': None,
                        'utilization_rate': 0
                    }
                
                if occupied:
                    daily_data[date_str]['total_occupancy_time'] += 1  # Assuming 1-second intervals
        
        return daily_data
    
    def _detect_temporal_trends(self, occupancy_history: Dict[str, List]) -> Dict[str, Any]:
        """Detect temporal trends in occupancy data"""
        trends = {
            'increasing_trend': False,
            'decreasing_trend': False,
            'peak_periods': [],
            'low_periods': []
        }
        
        # Analyze occupancy over time windows
        time_windows = self._create_time_windows(occupancy_history)
        
        if len(time_windows) >= 3:
            occupancy_rates = [window['occupancy_rate'] for window in time_windows]
            
            # Simple trend detection using linear regression
            x = np.arange(len(occupancy_rates))
            slope = np.polyfit(x, occupancy_rates, 1)[0]
            
            trends['increasing_trend'] = slope > 0.01
            trends['decreasing_trend'] = slope < -0.01
            
            # Find peak and low periods
            avg_rate = np.mean(occupancy_rates)
            trends['peak_periods'] = [i for i, rate in enumerate(occupancy_rates) if rate > avg_rate * 1.2]
            trends['low_periods'] = [i for i, rate in enumerate(occupancy_rates) if rate < avg_rate * 0.8]
        
        return trends
    
    def _create_time_windows(self, occupancy_history: Dict[str, List], 
                           window_size_minutes: int = 60) -> List[Dict[str, Any]]:
        """Create time windows for trend analysis"""
        windows = []
        
        # Get all timestamps
        all_timestamps = []
        for history in occupancy_history.values():
            all_timestamps.extend([timestamp for timestamp, _ in history])
        
        if not all_timestamps:
            return windows
        
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        window_size_seconds = window_size_minutes * 60
        
        current_time = min_time
        while current_time < max_time:
            window_end = current_time + window_size_seconds
            
            # Count occupancy in this window
            total_time = 0
            occupied_time = 0
            
            for chair_history in occupancy_history.values():
                for timestamp, occupied in chair_history:
                    if current_time <= timestamp < window_end:
                        total_time += 1
                        if occupied:
                            occupied_time += 1
            
            occupancy_rate = occupied_time / total_time if total_time > 0 else 0
            
            windows.append({
                'start_time': current_time,
                'end_time': window_end,
                'occupancy_rate': occupancy_rate,
                'total_time': total_time,
                'occupied_time': occupied_time
            })
            
            current_time = window_end
        
        return windows
    
    def _analyze_utilization_patterns(self, occupancy_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze utilization patterns across chairs"""
        if not occupancy_stats:
            return {}
        
        utilization_data = {
            'chair_utilization': {},
            'utilization_distribution': {},
            'efficiency_metrics': {}
        }
        
        # Individual chair utilization
        for chair_id, stats in occupancy_stats.items():
            utilization_data['chair_utilization'][chair_id] = {
                'occupancy_percentage': stats['occupancy_percentage'],
                'total_sessions': stats['total_sessions'],
                'avg_session_duration': stats['avg_occupancy_duration'],
                'efficiency_score': self._calculate_efficiency_score(stats)
            }
        
        # Utilization distribution
        occupancy_percentages = [stats['occupancy_percentage'] for stats in occupancy_stats.values()]
        utilization_data['utilization_distribution'] = {
            'mean': np.mean(occupancy_percentages),
            'median': np.median(occupancy_percentages),
            'std': np.std(occupancy_percentages),
            'min': np.min(occupancy_percentages),
            'max': np.max(occupancy_percentages),
            'quartiles': np.percentile(occupancy_percentages, [25, 50, 75])
        }
        
        return utilization_data
    
    def _calculate_efficiency_score(self, stats: Dict[str, Any]) -> float:
        """Calculate efficiency score for a chair"""
        # Simple efficiency score based on occupancy percentage and session patterns
        occupancy_score = min(stats['occupancy_percentage'] / 100, 1.0)
        session_score = min(stats['total_sessions'] / 10, 1.0)  # Normalize to max 10 sessions
        duration_score = min(stats['avg_occupancy_duration'] / 3600, 1.0)  # Normalize to max 1 hour
        
        efficiency_score = (occupancy_score * 0.5 + session_score * 0.3 + duration_score * 0.2)
        return efficiency_score
    
    def _analyze_peak_hours(self, occupancy_history: Dict[str, List]) -> Dict[str, Any]:
        """Analyze peak hours and usage patterns"""
        hourly_occupancy = {}
        
        # Initialize hourly data
        for hour in range(24):
            hourly_occupancy[hour] = {'total_time': 0, 'occupied_time': 0}
        
        # Aggregate data by hour
        for chair_history in occupancy_history.values():
            for timestamp, occupied in history:
                hour = datetime.fromtimestamp(timestamp).hour
                hourly_occupancy[hour]['total_time'] += 1
                if occupied:
                    hourly_occupancy[hour]['occupied_time'] += 1
        
        # Calculate occupancy rates
        hourly_rates = {}
        for hour, data in hourly_occupancy.items():
            rate = data['occupied_time'] / data['total_time'] if data['total_time'] > 0 else 0
            hourly_rates[hour] = rate
        
        # Find peak hours
        sorted_hours = sorted(hourly_rates.items(), key=lambda x: x[1], reverse=True)
        peak_hours = sorted_hours[:3]  # Top 3 peak hours
        low_hours = sorted_hours[-3:]  # Bottom 3 low hours
        
        return {
            'hourly_occupancy_rates': hourly_rates,
            'peak_hours': [{'hour': hour, 'rate': rate} for hour, rate in peak_hours],
            'low_hours': [{'hour': hour, 'rate': rate} for hour, rate in low_hours],
            'peak_hour': peak_hours[0][0] if peak_hours else None,
            'lowest_hour': low_hours[0][0] if low_hours else None
        }
    
    def _analyze_correlations(self, occupancy_history: Dict[str, List]) -> Dict[str, Any]:
        """Analyze correlations between different chairs"""
        if len(occupancy_history) < 2:
            return {}
        
        # Create time series data for each chair
        chair_series = {}
        for chair_id, history in occupancy_history.items():
            if not history:
                continue
            
            # Convert to time series
            timestamps = [timestamp for timestamp, _ in history]
            occupancy_values = [1 if occupied else 0 for _, occupied in history]
            
            chair_series[chair_id] = {
                'timestamps': timestamps,
                'occupancy': occupancy_values
            }
        
        # Calculate correlations between chairs
        correlations = {}
        chair_ids = list(chair_series.keys())
        
        for i, chair1 in enumerate(chair_ids):
            for chair2 in chair_ids[i+1:]:
                # Align time series and calculate correlation
                correlation = self._calculate_time_series_correlation(
                    chair_series[chair1], chair_series[chair2]
                )
                correlations[f"{chair1}_{chair2}"] = correlation
        
        return {
            'chair_correlations': correlations,
            'high_correlation_pairs': [
                pair for pair, corr in correlations.items() if abs(corr) > 0.7
            ],
            'low_correlation_pairs': [
                pair for pair, corr in correlations.items() if abs(corr) < 0.3
            ]
        }
    
    def _calculate_time_series_correlation(self, series1: Dict, series2: Dict) -> float:
        """Calculate correlation between two time series"""
        # Simple correlation calculation
        # In a real implementation, you'd want to properly align the time series
        try:
            correlation = np.corrcoef(series1['occupancy'], series2['occupancy'])[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_efficiency_metrics(self, occupancy_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate efficiency and optimization metrics"""
        if not occupancy_stats:
            return {}
        
        total_chairs = len(occupancy_stats)
        total_occupied_time = sum(stats['total_occupied_time'] for stats in occupancy_stats.values())
        avg_occupancy = np.mean([stats['occupancy_percentage'] for stats in occupancy_stats.values()])
        
        # Calculate space utilization efficiency
        utilization_efficiency = avg_occupancy / 100
        
        # Calculate session efficiency (how well chairs are being used)
        avg_sessions = np.mean([stats['total_sessions'] for stats in occupancy_stats.values()])
        session_efficiency = min(avg_sessions / 20, 1.0)  # Normalize to max 20 sessions
        
        # Calculate duration efficiency
        avg_duration = np.mean([stats['avg_occupancy_duration'] for stats in occupancy_stats.values()])
        duration_efficiency = min(avg_duration / 3600, 1.0)  # Normalize to max 1 hour
        
        # Overall efficiency score
        overall_efficiency = (utilization_efficiency * 0.4 + 
                            session_efficiency * 0.3 + 
                            duration_efficiency * 0.3)
        
        return {
            'utilization_efficiency': utilization_efficiency,
            'session_efficiency': session_efficiency,
            'duration_efficiency': duration_efficiency,
            'overall_efficiency': overall_efficiency,
            'optimization_potential': 1.0 - overall_efficiency,
            'recommended_actions': self._generate_recommendations(occupancy_stats)
        }
    
    def _generate_recommendations(self, occupancy_stats: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if not occupancy_stats:
            return recommendations
        
        avg_occupancy = np.mean([stats['occupancy_percentage'] for stats in occupancy_stats.values()])
        
        if avg_occupancy < 30:
            recommendations.append("Consider reducing the number of chairs - low utilization detected")
        elif avg_occupancy > 80:
            recommendations.append("Consider adding more chairs - high utilization detected")
        
        # Check for underutilized chairs
        underutilized = [chair_id for chair_id, stats in occupancy_stats.items() 
                        if stats['occupancy_percentage'] < 20]
        if underutilized:
            recommendations.append(f"Review positioning of chairs: {', '.join(underutilized)}")
        
        # Check for overutilized chairs
        overutilized = [chair_id for chair_id, stats in occupancy_stats.items() 
                       if stats['occupancy_percentage'] > 90]
        if overutilized:
            recommendations.append(f"Consider adding chairs near: {', '.join(overutilized)}")
        
        return recommendations
    
    def export_analysis_results(self, analysis_results: Dict[str, Any], 
                              output_path: str):
        """Export analysis results to various formats"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to JSON
        json_file = output_dir / 'analysis_results.json'
        with open(json_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Export summary statistics to CSV
        if 'summary_statistics' in analysis_results:
            summary_df = pd.DataFrame([analysis_results['summary_statistics']])
            csv_file = output_dir / 'summary_statistics.csv'
            summary_df.to_csv(csv_file, index=False)
        
        # Export utilization data to CSV
        if 'utilization_analysis' in analysis_results:
            utilization_data = analysis_results['utilization_analysis']
            if 'chair_utilization' in utilization_data:
                chair_df = pd.DataFrame(utilization_data['chair_utilization']).T
                chair_csv = output_dir / 'chair_utilization.csv'
                chair_df.to_csv(chair_csv)
        
        logger.info(f"Analysis results exported to {output_path}")
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a comprehensive text report"""
        report = []
        report.append("=" * 60)
        report.append("OFFICE SEAT OCCUPANCY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary Statistics
        if 'summary_statistics' in analysis_results:
            summary = analysis_results['summary_statistics']
            report.append("SUMMARY STATISTICS")
            report.append("-" * 20)
            report.append(f"Total Chairs: {summary.get('total_chairs', 'N/A')}")
            report.append(f"Total Occupied Time: {summary.get('total_occupied_time', 0):.2f} seconds")
            report.append(f"Average Occupancy: {summary.get('average_occupancy_percentage', 0):.2f}%")
            report.append(f"Total Events: {summary.get('total_occupancy_events', 0)}")
            report.append("")
        
        # Utilization Analysis
        if 'utilization_analysis' in analysis_results:
            util = analysis_results['utilization_analysis']
            if 'utilization_distribution' in util:
                dist = util['utilization_distribution']
                report.append("UTILIZATION DISTRIBUTION")
                report.append("-" * 25)
                report.append(f"Mean Utilization: {dist.get('mean', 0):.2f}%")
                report.append(f"Median Utilization: {dist.get('median', 0):.2f}%")
                report.append(f"Standard Deviation: {dist.get('std', 0):.2f}%")
                report.append("")
        
        # Peak Hours Analysis
        if 'peak_hours_analysis' in analysis_results:
            peak = analysis_results['peak_hours_analysis']
            report.append("PEAK HOURS ANALYSIS")
            report.append("-" * 20)
            if 'peak_hours' in peak:
                report.append("Top 3 Peak Hours:")
                for hour_data in peak['peak_hours']:
                    report.append(f"  Hour {hour_data['hour']}: {hour_data['rate']:.2f} occupancy rate")
            report.append("")
        
        # Recommendations
        if 'efficiency_metrics' in analysis_results:
            efficiency = analysis_results['efficiency_metrics']
            if 'recommended_actions' in efficiency:
                report.append("RECOMMENDATIONS")
                report.append("-" * 15)
                for i, rec in enumerate(efficiency['recommended_actions'], 1):
                    report.append(f"{i}. {rec}")
                report.append("")
        
        report.append("=" * 60)
        report.append("End of Report")
        report.append("=" * 60)
        
        return "\n".join(report)
