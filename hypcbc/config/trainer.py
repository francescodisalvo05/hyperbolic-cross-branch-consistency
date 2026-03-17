from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Optional, Union
from pathlib import Path

# Import own scripts
from hypcbc.config.base import BaseConfig


T = TypeVar('T', bound='BaseConfig')


@dataclass
class TrainerConfig(BaseConfig):
    """Training configuration."""
    max_epochs: int = 100

    # Loss
    loss: str = "ce"
    loss_uses_domain: bool = False
    loss_uses_features: bool = False
    
    # Monitoring performance
    monitor: str = "loss"
    monitor_mode: str = "min"

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Path
    output_root: Path = Path("./assets/output")
    
    # Runtime
    seed: int = 42
    device: Union[int, str] = "cuda"
    use_wandb: bool = True
    skip_train: Optional[float] = None


    # == Loss hparams == #
    
    # Distillation (Ours)
    dist_lam: Optional[float] = None
    dist_temp: Optional[float] = None

    # IRM
    irm_lambda: Optional[int] = None
    irm_anneal_iters: Optional[int] = None

    # GroupDRO
    gdro_eta: Optional[float] = None

    # VREx
    vrex_lambda: Optional[float] = None
    vrex_anneal_iters: Optional[int] = None

    # MMD and CORAL
    mmd_gamma: Optional[float] = None

    # Ablation
    eval_domain_accuracy: Optional[bool] = None
    
    def validate(self) -> None:
        """Validate trainer configuration."""
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.min_delta < 0.0:
            raise ValueError(f"min_delta must be positive")
        valid_modes = ["min", "max"]
        if self.monitor_mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        
        