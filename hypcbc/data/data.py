from __future__ import annotations
from typing import Dict, Optional
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

# Import own scripts
from hypcbc.data.transformbuilder import build_transforms
from hypcbc.data.databuilder import build_dataset
from hypcbc.data.registry import DATASET_CLASSES
from hypcbc.config.data import DataConfig


class DataModule:
    """DataModule class."""
    def __init__(self, config: DataConfig, backbone: str, backbone_transform: transforms.Compose):
        self.config = config
        self.backbone = backbone
        self.backbone_transform = backbone_transform

        self._datasets: Optional[Dict[str, torch.utils.data.Dataset]] = {}
        self._loaders: Optional[Dict[str, DataLoader]] = {}
        self._transforms: Optional[Dict[str, Optional[transforms.Compose]]] = {}

    @property
    def loaders(self) -> Dict[str, DataLoader]:
        """Get the dataset loaders. Raises error if not set up."""
        if self._loaders is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self._loaders

    @property
    def num_classes(self) -> int:
        """Get the number of classes. Raises error if dataset is not found."""
        try:
            return DATASET_CLASSES[self.config.dataset]
        except KeyError:
            raise ValueError(f"unknown dataset '{self.config.dataset}'. Valid options: {list(DATASET_CLASSES.keys())}")

    def setup(self) -> None:
        """Set up the data components."""
        # Transform functions
        self._transforms = build_transforms(
            mode=self.config.mode,
            backbone_transform=self.backbone_transform,
            augmentation=self.config.augmentation,
            dataset_name=self.config.dataset
        )

        # Initialize datasets
        self._datasets = build_dataset(
            config=self.config,
            backbone=self.backbone,
            transforms=self._transforms
        )

        # Define dataloaders
        self._loaders = self._build_dataloaders()
        

    def _build_dataloaders(self) -> Dict[str, DataLoader]:
        """Define dataloaders for each of the available data splits."""
        loaders = {}
        for split, dataset in self._datasets.items():
            if isinstance(dataset, torch.utils.data.Dataset):
                shuffle = (split == "train" and self.config.mode == "emb")
                loaders[split] = DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    shuffle=shuffle,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.num_workers > 0,
                    drop_last=(split == "train")
                )
        return loaders