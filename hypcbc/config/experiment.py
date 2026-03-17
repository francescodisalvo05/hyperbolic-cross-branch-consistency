from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Optional

# Import own scripts
from hypcbc.config.optimizer import OptimizerConfig
from hypcbc.config.trainer import TrainerConfig
from hypcbc.config.model import ModelConfig
from hypcbc.config.data import DataConfig
from hypcbc.config.base import BaseConfig


T = TypeVar('T', bound='BaseConfig')


@dataclass
class ExperimentConfig(BaseConfig):
    """Main experiment configuration combining all components."""
    data: DataConfig
    model: ModelConfig
    optimizer: Optional[OptimizerConfig] = None
    trainer: Optional[TrainerConfig] = None
    
    # Experiment metadata
    experiment_name: str = "default_experiment"
    description: str = ""