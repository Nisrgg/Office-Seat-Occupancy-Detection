"""
Configuration Management Module
==============================

Handles loading and validation of configuration parameters for the seat detection system.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        required_sections = ['model', 'detection', 'data', 'analysis']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Create logs directory if it doesn't exist
        log_file = log_config.get('log_file', 'logs/seat_detection.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler() if log_config.get('console_output', True) else logging.NullHandler()
            ]
        )
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config['model']
    
    @property
    def detection_config(self) -> Dict[str, Any]:
        """Get detection configuration"""
        return self.config['detection']
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config['data']
    
    @property
    def analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration"""
        return self.config['analysis']
    
    @property
    def visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        return self.config.get('visualization', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, key: str, value: Any):
        """Update configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save configuration to file"""
        save_path = path or self.config_path
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
