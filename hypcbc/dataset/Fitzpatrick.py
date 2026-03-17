from sklearn.model_selection import train_test_split
from typing import Optional, Union, Tuple
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

import torchvision.transforms as transforms
import numpy as np
import torch


class Fitzpatrick17k(Dataset):
    """
    Fitzpatrick17k dataset for dermatological image classification with skin tone analysis.
    
    The dataset contains dermatological images labeled with:
    - Disease labels (3 or 9 categories)
    - Fitzpatrick skin type scales (grouped as 1-2, 3-4, 5-6)
    """

    # Class constants
    VALID_LABEL_PARTITIONS = [3, 9]
    VALID_SPLITS = ['all', 'train', 'val', 'test']
    FITZPATRICK_MAPPING = {
        '1': '12', '2': '12',  # Light skin
        '3': '34', '4': '34',    
        '5': '56', '6': '56'   # Dark skin
    }

    def __init__(
        self, 
        root: Union[str, Path],
        split: str = 'train',
        label_partition: int = 3,
        target_domain: Optional[str] = 'id',
        transform: Optional[transforms.Compose] = None
    ) -> None:

        # Store parameters
        self.root = Path(root)
        self.split = split
        self.label_partition = label_partition
        self.target_domain = target_domain
        self.transform = transform

        # Validate inputs
        self._validate_inputs()

        # Initialize data containers
        self.fpaths = []
        self.labels = []
        self.fitz_scales = []
        
        # Load and process data
        fpaths, labels, fitz_scales = self._load_dataset()

        # Split data by domain 
        if self.target_domain in ['12','34','56']:
            self.fpaths, self.labels, self.fitz_scales = self._split_by_domain(fpaths, labels, fitz_scales)
        elif self.target_domain == 'id':
            self.fpaths, self.labels, self.fitz_scales = self._split_id(fpaths, labels, fitz_scales)

    def _validate_inputs(self):
        """Validate all input parameters"""
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory {self.root} does not exist")
        
        if self.label_partition not in self.VALID_LABEL_PARTITIONS:
            raise ValueError(f"label_partition must be one of {self.VALID_LABEL_PARTITIONS}, "
                        f"got {self.label_partition}")
        
        if self.split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{self.split}'")
        
        # Check for required files
        metadata_path = self.root / 'fitzpatrick17k.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    def _load_dataset(self) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Load image paths and labels from CSV metadata file"""
        metadata_path = self.root / 'fitzpatrick17k.csv'

        # Read CSV with pandas for better handling
        import pandas as pd
        df = pd.read_csv(metadata_path)
        required_columns = ['md5hash', 'fitzpatrick_scale']

        # Add label column based on partition
        if self.label_partition == 3:
            label_col = 'three_partition_label'
        elif self.label_partition == 9:
            label_col = 'nine_partition_label'

        required_columns.append(label_col)

        filepaths, labels_str, fitz_scales = [], [], []
        data_dir = self.root / 'data'

        for _, row in df.iterrows():
            # Check if image file exists
            img_path = data_dir / f"{row['md5hash']}.jpg"
            if not img_path.exists():
                continue
            
            # Process Fitzpatrick scale
            fitz_scale_str = str(row['fitzpatrick_scale'])
            if fitz_scale_str not in self.FITZPATRICK_MAPPING:
                continue  # Skip invalid/missing scales
            
            # Collect valid data
            filepaths.append(str(img_path))
            labels_str.append(str(row[label_col]))
            fitz_scales.append(self.FITZPATRICK_MAPPING[fitz_scale_str])
        
        # Create label mapping
        unique_labels = sorted(set(labels_str))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.class_names = unique_labels
        
        # Convert string labels to indices
        labels = [self.label_map[label] for label in labels_str]
        
        return np.array(filepaths), np.array(labels), np.array(fitz_scales)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Get a single from the dataset"""
        img = Image.open(self.fpaths[index]).convert("RGB")
        label = self.labels[index]
        fs = self.fitz_scales[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, fs

    def __len__(self) -> int:
        return len(self.fpaths)
    
    def _split_by_domain(self, embeddings, labels, domains) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data based on target domain"""

        domains_int = domains.copy().astype(np.int64)

        # Split based on target domain
        if self.split == 'test':
            mask = domains_int == int(self.target_domain)
            return embeddings[mask], labels[mask], domains_int[mask]
        else:
            # Use non-target domains for train/val
            mask = domains_int != int(self.target_domain)
            trainval_embeddings = embeddings[mask]
            trainval_labels = labels[mask]
            trainval_domains = domains_int[mask]

            # Further split into train/val
            train_emb, val_emb, train_labels, val_labels, train_domains, val_domains = train_test_split(
                trainval_embeddings, trainval_labels, trainval_domains,
                stratify=trainval_labels,
                test_size=0.2,
                random_state=342234324 # seed independent
            )

            if self.split == 'train':
                return train_emb, train_labels, train_domains
            else:  # val
                return val_emb, val_labels, val_domains

    def _split_id(self, embeddings, labels, domains) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ID data split, stratified by labels"""

        domains_int = domains.copy().astype(np.int64)

        # Split into trainval and test
        train_emb, test_emb, train_labels, test_labels, train_domains, test_domains = train_test_split(
            embeddings, labels, domains_int,
            stratify=labels,
            test_size=0.2,
            random_state=342234324 # seed independent
        )

        # Further split into train/val
        train_emb, val_emb, train_labels, val_labels, train_domains, val_domains = train_test_split(
            train_emb, train_labels, train_domains,
            stratify=train_labels,
            test_size=0.1 / 0.7,
            random_state=342234324 # seed independent
        )
        
        if self.split == 'train':
            return train_emb, train_labels, train_domains
        elif self.split == 'val':
            return val_emb, val_labels, val_domains
        elif self.split == 'test':
            return test_emb, test_labels, test_domains