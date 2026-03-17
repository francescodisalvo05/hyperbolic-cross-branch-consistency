from torch.utils.data import Dataset
from typing import Tuple

import torch


class Camelyon17Wrapper(Dataset):
    """Wrap WILDS Camelyon17 samples as `(image, label, domain)` tuples."""

    def __init__(self, wilds_dataset: Dataset) -> None:
        """Store a WILDS subset (train, val, or test)."""
        self.wilds_dataset = wilds_dataset

    def __len__(self) -> int:
        return len(self.wilds_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        x, y, z = self.wilds_dataset[idx]  # Original triplet (image, label, metadata)
        domain = z[0]                      # Extract hospital ID from metadata
        return x, y, domain
