from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Tuple, Union
from pathlib import Path

import numpy as np
import torch


class EMBDataset(Dataset):
    """Dataset wrapper for precomputed embeddings saved as `.pt` files."""

    def __init__(self, root: Union[str, Path], split: str, augmentation: str = None) -> None:
        super(EMBDataset, self).__init__()
        self.root = root
        self.split = split
        self.augmentation = augmentation

        # Validate inputs
        self._validate_inputs()

        # Load and filter data
        data = self._load_data()

        if len(data) == 2:
            self.embeddings, self.labels = data
            self.domains = None
        else:
            self.embeddings, self.labels, self.domains = data
            self.domains = self.domains.reshape(-1,)

        # Enforce them to be 1D
        self.labels = self.labels.reshape(-1,)

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory {self.root} does not exist")
        
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")
            
        if self.augmentation and self.split != 'train':
            raise ValueError(f"Augmentation `{self.augmentation}` cannot be applied to split: `{self.split}`")

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, np.ndarray], 
                                             Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        x = self.embeddings[idx]
        y = self.labels[idx]

        if self.domains is not None:
            d = self.domains[idx]
            return x, y, d
        else:
            return x, y

    def _load_data(self) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Load data from the split file selected by current options."""
        filename = self._get_filename()
        filepath = self.root / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file {filepath} does not exist")
        
        data = torch.load(filepath, weights_only=True)
        
        if len(data) == 2:
            embeddings, labels = data
            embeddings = embeddings.numpy()
            labels = labels.numpy()
            return embeddings, labels
        else:
            embeddings, labels, domain = data
            embeddings = embeddings.numpy()
            labels = labels.numpy()
            domain = domain.numpy()
            return embeddings, labels, domain
    
    def _get_filename(self) -> str:
        """Generate filename from split and augmentation settings."""
        if self.augmentation and self.split == 'train':
            return f'{self.split}_{self.augmentation}.pt'
        else:
            return f'{self.split}.pt'


class EMBDatasetID(Dataset):
    """Create in-distribution train/val/test splits from combined embedding files."""

    def __init__(self, root: Union[Path, str], split: str) -> None:
        super(EMBDatasetID, self).__init__()
        self.root = root
        self.split = split

        # Validate inputs
        self._validate_inputs()

        # Load and filter data
        self.embeddings, self.labels, self.domains = self._load_data()

        # Enforce them to be 1D
        self.labels = self.labels.reshape(-1,)
        self.domains = self.domains.reshape(-1,)


    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory {self.root} does not exist")
        
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.embeddings[idx]
        y = self.labels[idx]
        d = self.domains[idx]
        return x, y, d

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load embeddings and generate deterministic ID train/val/test splits."""

        train_size, val_size, test_size = 0.7, 0.1, 0.2
        seed = 342234324
        
        # Load .pt files (assumed to contain NumPy arrays)
        x_tr, y_tr, d_tr = torch.load(self.root / 'train.pt')  # all are np.ndarrays
        x_va, y_va, d_va = torch.load(self.root / 'val.pt')
        x_te, y_te, d_te = torch.load(self.root / 'test.pt')

        # Concatenate all
        x_all = np.concatenate([x_tr, x_va, x_te], axis=0)
        y_all = np.concatenate([y_tr, y_va, y_te], axis=0)
        d_all = np.concatenate([d_tr, d_va, d_te], axis=0)

        # First split: train (70%) and temp (30%)
        x_train, x_temp, y_train, y_temp, d_train, d_temp = train_test_split(
            x_all, y_all, d_all,
            test_size=(1 - train_size),
            stratify=y_all,
            random_state=seed
        )

        # Second split: val (10%) and test (20%) from temp
        # Compute relative proportions from temp
        val_ratio = val_size / (val_size + test_size)  # 0.1 / 0.3 ≈ 0.333
        x_val, x_test, y_val, y_test, d_val, d_test = train_test_split(
            x_temp, y_temp, d_temp,
            test_size=(1 - val_ratio),
            stratify=y_temp,
            random_state=seed
        )

        if self.split == 'train':
            return x_train, y_train, d_train,
        elif self.split == 'val':
            return x_val, y_val, d_val
        else:
            return x_test, y_test, d_test
