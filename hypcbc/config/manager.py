from __future__ import annotations
from typing import Dict, Any, List, TypeVar, Optional, Union
from pathlib import Path
import yaml
import os

# Import own scripts
from hypcbc.config.base import BaseConfig
from hypcbc.config.experiment import ExperimentConfig


T = TypeVar('T', bound='BaseConfig')


class ConfigManager:
    """Manages loading, saving, and merging of configurations."""
    
    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML file with environment variable substitution."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            content = f.read()
        
        # Environment variable substitution
        content = os.path.expandvars(content)
        
        return yaml.safe_load(content)
    
    @staticmethod
    def save_yaml(config: BaseConfig, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
    
    @staticmethod
    def load_experiment_config(config_path: Union[str, Path]) -> ExperimentConfig:
        """Load complete experiment configuration from YAML."""
        config_data = ConfigManager.load_yaml(config_path)
        return ExperimentConfig.from_dict(config_data)
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        def _deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = _deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return _deep_merge(base_config, override_config)
    
    @staticmethod
    def load_with_overrides(base_config_path: Union[str, Path],
                           override_paths: Optional[List[Union[str, Path]]] = None,
                           cli_overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
        """Load configuration with multiple override levels."""
        # Load base config
        config_data = ConfigManager.load_yaml(base_config_path)
        
        # Apply file overrides
        if override_paths:
            for override_path in override_paths:
                override_data = ConfigManager.load_yaml(override_path)
                config_data = ConfigManager.merge_configs(config_data, override_data)
        
        # Apply CLI overrides
        if cli_overrides:
            config_data = ConfigManager.merge_configs(config_data, cli_overrides)
        
        return ExperimentConfig.from_dict(config_data)