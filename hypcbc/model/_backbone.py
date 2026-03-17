from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from types import SimpleNamespace


class Backbone(nn.Module):
    """Wrap a **timm** backbone together with its preprocessing transform."""

    # ------------------------------------------------------------------
    # lookup tables
    # ------------------------------------------------------------------
    _NAME_MAP: Dict[str, str] = {
        "dino_small": "vit_small_patch16_224.dino",
        "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "deit3_small": "deit3_small_patch16_224.fb_in22k_ft_in1k",
        "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
        "gigapath": "hf_hub:prov-gigapath/prov-gigapath",
    }

    _FEATURE_DIM: Dict[str, int] = {
        "small": 384,
        "base": 768,
    }

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        backbone_config: SimpleNamespace,
        trainable: bool = False
    ) -> None:
        """TBD
        """
        super().__init__()
        self.full_name = self.resolve_full_name(backbone_config.name)

        # build model ------------------------------------------------------
        self.model = timm.create_model(
            self.full_name,
            pretrained = True,
            num_classes = 0,
        )

        # freeze if requested ---------------------------------------------
        if not trainable:
            self.freeze()

        # canonical transform ---------------------------------------------
        data_cfg = timm.data.resolve_model_data_config(self.model)
        self.transform_fn = timm.data.create_transform(**data_cfg, is_training=False)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Plain forward: returns features of shape *(B, feature_dim)*."""
        return self.model(x)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        return self._infer_feature_dim(self.name)

    @property
    def transform(self) -> Callable:
        return self._transform

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # embedding / feature extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(
        self,
        dataloader: DataLoader,
        device: Union[str, torch.device] = "cuda:0",
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Mirror of your original `extract_features` helper.

        Returns
        -------
        (embeddings, labels) *or* (embeddings, labels, domains)
        depending on whether the dataloader provides domain metadata.
        """
        from tqdm.auto import tqdm  # import lazily to avoid unnecessary dep

        self.eval()
        device = torch.device(device)
        self.to(device)

        embs: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        domains: List[torch.Tensor] = []

        for batch in tqdm(dataloader, desc="Extracting features"):
            # ---------------------------------------------------------
            # unpack batch
            # ---------------------------------------------------------
            if len(batch) == 3:
                inputs, targets, domain = batch  # type: ignore[misc]
                domains.append(domain.squeeze(0).reshape(-1, 1))
            else:
                inputs, targets = batch  # type: ignore[misc]

            # forward -------------------------------------------------------
            inputs = inputs.to(device, non_blocking=True)
            feats = self(inputs).cpu()

            embs.append(feats)
            labels.append(targets.squeeze(0).cpu())

        # ------------------------------------------------------------------
        # stack & return
        # ------------------------------------------------------------------
        embeddings = torch.cat(embs, dim=0)
        labels_t   = torch.cat(labels, dim=0)

        if domains:
            domains_t = torch.cat(domains, dim=0)
            return embeddings, labels_t, domains_t
        
        return embeddings, labels_t
    

    # ------------------------------------------------------------------
    # static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def resolve_full_name(name: str) -> str:
        try:
            return Backbone._NAME_MAP[name]
        except KeyError as err:
            raise ValueError(f"Backbone shorthand '{name}' not recognised.") from err