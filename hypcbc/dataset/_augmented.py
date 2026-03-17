from typing import Any, Tuple

import torch


class AugmentedDatasetWrapper(torch.utils.data.Dataset):
    """Repeat each base sample `n_augments` times."""

    def __init__(self, dataset: torch.utils.data.Dataset, n_augments: int = 1) -> None:
        self.dataset = dataset
        self.n_augments = n_augments

    def __len__(self) -> int:
        return len(self.dataset) * self.n_augments

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        base_idx = idx % len(self.dataset)
        img, label = self.dataset[base_idx][:2]  # Support (img, label) or (img, label, domain)
        rest = self.dataset[base_idx][2:]        # domain if available
        return (img, label, *rest) if rest else (img, label)
