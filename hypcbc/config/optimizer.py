from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Optional

# Import own scripts
from hypcbc.config.base import BaseConfig


T = TypeVar('T', bound='BaseConfig')


@dataclass
class OptimizerConfig(BaseConfig):
    """Optimizer configuration."""
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.1
    
    # Scheduler
    scheduler: Optional[str] = "cosine"
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    
    def validate(self) -> None:
        """Validate optimizer configuration."""
        if self.lr <= 0:
            raise ValueError("learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.name.lower() != 'adamw':
            raise ValueError(f"optimizer `{self.name.lower()}` not supported.")