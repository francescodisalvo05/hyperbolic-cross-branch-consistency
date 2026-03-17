from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Optional

# Import own scripts
from hypcbc.config.base import BaseConfig


T = TypeVar('T', bound='BaseConfig')


@dataclass
class ModelConfig(BaseConfig):
    """Model backbone configuration."""
    backbone_id: str = "dinov2_small"
    freeze_backbone: bool = True
    device: Optional[str] = None

    # Head(s)
    manifold: str = "hyp"
    branch1_dim: int = 128
    branch2_dim: Optional[int] = None
    num_classes: Optional[int] = None

    # Hyperbolic hparams
    hyp_curvature: float = 1.0
    hyp_clip_r: float = 1.0
    
    # Feature extraction
    extract_projections: bool = False


    def validate(self) -> None:
        """Validate backbone configuration."""
        if not self.backbone_id:
            raise ValueError("backbone name cannot be empty")
        if self.hyp_curvature <= 0:
            raise ValueError("hyp_curvature must be positive")
        if self.hyp_clip_r <= 0:
            raise ValueError("clip_r must be positive")

