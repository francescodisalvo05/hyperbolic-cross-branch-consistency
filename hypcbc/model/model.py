from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision import transforms

# Import own scripts
from hypcbc.model.registry import FEATURE_DIMENSION, MODEL_EXTENDED
from hypcbc.hyptorch.hyper_nets import HyperbolicMLR
from hypcbc.config.model import ModelConfig
import hypcbc.hyptorch.nn as hypnn


class ModelModule(nn.Module):
    """ModelModule class for hyperbolic domain generalization."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.manifold = config.manifold
        
        # Properties that will be set during setup
        self._backbone: Optional[nn.Module] = None
        self._transform: Optional[transforms.Compose] = None

        # Single branch setup
        self._hidden: Optional[nn.Module] = None
        self._head: Optional[nn.Module] = None
        
        # Two-branches setup
        self._hidden_branch1: Optional[nn.Module] = None
        self._head_branch1: Optional[nn.Module] = None
        self._hidden_branch2: Optional[nn.Module] = None
        self._head_branch2: Optional[nn.Module] = None
        
        # Forward function based on setup
        self._forward_fn: Optional[Callable] = None
        
        # Validate configuration
        self._validate_config()

    @property
    def backbone(self) -> nn.Module:
        """Get the backbone model. Raises error if not set up."""
        if self._backbone is None:
            raise RuntimeError("ModelModule not set up")
        return self._backbone
    
    @property
    def transform(self) -> nn.Module:
        """Get the backbone transform. Raises error if not set up."""
        if self._transform is None:
            raise RuntimeError("ModelModule not set up")
        return self._transform

    @property
    def backbone_id(self) -> str:
        """Get the backbone string from registry."""
        try:
            return MODEL_EXTENDED[self.config.backbone_id]
        except KeyError:
            raise ValueError(f"unknown backbone `{self.config.backbone_id}`")

    @property
    def feature_dim(self) -> int:
        """Get the feature dimension from registry."""
        backbone_key = self.config.backbone_id.split("_")[1]
        try:
            return FEATURE_DIMENSION[backbone_key]
        except KeyError:
            raise ValueError(f"unknown backbone feature dimension for `{backbone_key}`")

    def _validate_config(self) -> None:
        """Validate the model configuration."""
        if self.config.backbone_id not in MODEL_EXTENDED:
            raise ValueError(f"unknown backbone '{self.config.backbone_id}'")

        if not self.manifold:
            raise ValueError("manifold must be specified")
        
        if self.manifold not in ["euc", "hyp"]:
            raise ValueError(f"unsupported manifold '{self.manifold}', must be 'euc' or 'hyp'")
        
        if not hasattr(self.config, 'branch1_dim') or self.config.branch1_dim is None:
            raise ValueError("branch1_dim must be specified")
        
        if self.config.branch1_dim <= 0:
            raise ValueError("branch1_dim must be positive")
        
        if hasattr(self.config, 'branch2_dim') and self.config.branch2_dim is not None:
            if self.config.branch2_dim <= 0:
                raise ValueError("branch2_dim must be positive if specified")

    def setup(self) -> None:
        """Set up the model components."""
        # Load backbone
        self._backbone = timm.create_model(
            self.backbone_id, 
            pretrained=True, 
            num_classes=0
        )

        # Freeze backbone if requested
        if self.config.freeze_backbone:
            self._freeze_backbone()

        # Load the backbone-specific transform
        data_config = timm.data.resolve_model_data_config(self._backbone)
        self._transform = timm.data.create_transform(**data_config, is_training=False)

        # Build heads
        self._build_heads()
        
        # Select forward function (based on single or two-branch)
        self._forward_fn = self._select_forward()

    def _freeze_backbone(self) -> None:
        """Freeze the backbone parameters."""
        if self._backbone is None:
            raise RuntimeError("ModelModule not set up. Call setup() first.")
            
        for param in self._backbone.parameters():
            param.requires_grad = False

    def _build_heads(self) -> None:
        """Build the projection heads based on branch configuration."""
        if self.config.branch2_dim is not None:
            # Dual branch setup
            self._hidden_branch1, self._head_branch1 = self._build_hidden_and_head(
                self.config.branch1_dim
            )
            self._hidden_branch2, self._head_branch2 = self._build_hidden_and_head(
                self.config.branch2_dim
            )
        else:
            # Single branch setup
            self._hidden, self._head = self._build_hidden_and_head(
                self.config.branch1_dim
            )

    def _build_hidden_and_head(self, dim: int) -> Tuple[nn.Module, nn.Module]:
        """Build hidden projection and classification head for given dimension."""
        
        if self.manifold == "euc":
            hidden = nn.Linear(self.feature_dim, dim)
            head = nn.Linear(dim, self.config.num_classes)
            
        elif self.manifold == "hyp":
            hidden = nn.Sequential(
                nn.Linear(self.feature_dim, dim),
                hypnn.ToPoincare(
                    c=self.config.hyp_curvature,
                    ball_dim=dim,
                    riemannian=False,
                    clip_r=self.config.hyp_clip_r
                )
            )
            head = HyperbolicMLR(
                ball_dim=dim, 
                n_classes=self.config.num_classes, 
                c=self.config.hyp_curvature
            )
        else:
            raise ValueError(f"Unsupported manifold base '{self.manifold}'")

        return hidden, head

    def _select_forward(
        self,
    ) -> Callable[[torch.Tensor, bool], Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Select the appropriate forward function based on branch configuration."""
        if self.config.branch2_dim is not None:
            return self._forward_dual_branch
        else:
            return self._forward_single_branch

    def forward(
        self, x: torch.Tensor, return_proj: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model."""
        if self._forward_fn is None:
            raise RuntimeError("ModelModule not set up. Call setup() first.")

        if not self.config.freeze_backbone:
            feats = self.backbone(x)  # If fine-tune the backbone (need `raw` datasets!)
        else:
            feats = x                 # Otherwise, assuming `emb` datasets

        # Use the selected forward function
        return self._forward_fn(feats, return_proj)

    def _forward_single_branch(self, feats: torch.Tensor, return_proj: bool = False) -> torch.Tensor:
        """Forward pass for single branch (non-2b) manifolds."""
        if self._hidden is None:
            raise RuntimeError("Single branch components not initialized")
            
        z = self._hidden(feats)
        
        # Return projected embeddings 
        if return_proj:
            return self._head(z), z
        
        return self._head(z)

    def _forward_dual_branch(
        self, feats: torch.Tensor, return_proj: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for dual branch configuration."""
        if self._hidden_branch1 is None or self._hidden_branch2 is None:
            raise RuntimeError("Dual branch components not initialized")
            
        z_branch1 = self._hidden_branch1(feats)
        z_branch2 = self._hidden_branch2(feats)

        # Return projected embeddings from both branches
        if return_proj:
            logits = self._head_branch1(z_branch1), self._head_branch2(z_branch2)
            z = z_branch1, z_branch2
            return logits, z

        return self._head_branch1(z_branch1), self._head_branch2(z_branch2)

    @torch.no_grad()
    def extract_features(
        self,
        dataloader: DataLoader,
        device: Union[str, torch.device] = "cuda"
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Extract features from the model using a dataloader to create an embedding-based database."""
        if self._backbone is None:
            raise RuntimeError("ModelModule not set up. Call setup() first.")
            
        self.eval()
        self.to(device)

        feats_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        domains_list: List[torch.Tensor] = []

        for batch in tqdm(dataloader, desc="Extracting features"):
            # Handle different batch formats
            if len(batch) == 3:
                x, y, d = batch
                domains_list.append(d.squeeze(0).reshape(-1, 1))
            elif len(batch) == 2:
                x, y = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            # Move to device
            x = x.to(device, non_blocking=True)
            
            # Extract backbone features
            out = self.backbone(x)

            # Store results
            feats_list.append(out.cpu())
            labels_list.append(y.squeeze(0).cpu())

        # Concatenate all features and labels
        feats = torch.cat(feats_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        # Return with or without domains
        if domains_list:
            domains = torch.cat(domains_list, dim=0)
            return feats, labels, domains

        return feats, labels
