from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Optional
from pathlib import Path

# Import own scripts
from hypcbc.config.base import BaseConfig


T = TypeVar('T', bound='BaseConfig')


@dataclass
class DataConfig(BaseConfig):
    """Data loading and preprocessing configuration."""
    # Dataset specifics
    dataset: str
    mode: str = "raw"       
    batch_size: int = 512
    target_domain: Optional[str] = None

    # Image or label transform
    augmentation: Optional[str] = None
    augmentation_factor: Optional[int] = None
    data_root: Path = Path("./assets/data")
    database_root: Path = Path("./assets/database")
    
    # Dataloading
    num_workers: int = 4
    drop_last: bool = False
    
    def validate(self) -> None:
        """Validate data configuration."""

        # Dataset specifics and Dataloading
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.mode not in ["raw", "emb"]:
            raise ValueError("mode must be 'raw' (to load images) or 'emb' (to load embeddings)")
        
        # Validate augmentation options
        valid_augs = ["augmix", "randaugment", "augmedmnistc", None]
        if self.augmentation not in valid_augs:
            raise ValueError(f"augmentation must be one of {valid_augs}")
        elif self.augmentation and self.mode == 'raw' and self.augmentation_factor is None:
            raise ValueError(f"you need to set augmentation_factor")