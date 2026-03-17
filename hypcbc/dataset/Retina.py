from typing import Optional, Union, Tuple
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import torch


class RetinaDataset(Dataset):
    VALID_SPLITS = ['train', 'val', 'test']
    DATASET_SPLITS = {'train': ['aptos', 'deepdr'], 'val': ['idrid'], 'test': ['messidor']}
    DATASET_IDS = {'aptos': 0, 'deepdr': 1, 'idrid': 2, 'messidor': 3}

    def __init__(
        self, 
        root: Union[str, Path],
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ) -> None:

        # Store parameters
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Validate inputs
        self._validate_inputs()

        # Initialize containers
        self.fpaths, self.labels, self.domains = [], [], []
        
        # Load data based on selected split
        self._load_dataset()


    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory {self.root} does not exist")

        if self.split not in self.VALID_SPLITS:
            raise ValueError("Wrong split.")

    def _load_dataset(self) -> None:
        """Load split-specific samples into internal path/label/domain lists."""
        datasets = self.DATASET_SPLITS[self.split]

        for dataset_str in datasets:
            if dataset_str == 'aptos':
                self._load_aptos()
            elif dataset_str == 'deepdr':
                self._load_deepdr()
            elif dataset_str == 'idrid':
                self._load_idrid()
            elif dataset_str == 'messidor':
                self._load_messidor()
        

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Return one sample as `(image, label, domain)`."""
        img = Image.open(self.fpaths[index]).convert("RGB")
        label = self.labels[index]
        domain = self.domains[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, domain

    def __len__(self) -> int:
        return len(self.fpaths)
    

    def _load_aptos(self) -> None:
        """
        Loads the entire APTOS dataset (train + val + test) into one dataset,
        for cross-validation splitting later.
        """
        # Define all splits
        splits_info = [
            {
                'csv': self.root / 'aptos' / 'train_1.csv',
                'img_dir': self.root / 'aptos' / 'train_images' / 'train_images' # Default dataset structure
            },
            {
                'csv': self.root / 'aptos' / 'valid.csv',
                'img_dir': self.root / 'aptos' / 'val_images' / 'val_images'
            },
            {
                'csv': self.root / 'aptos' / 'test.csv',
                'img_dir': self.root / 'aptos' / 'test_images' / 'test_images'
            }
        ]

        # Iterate over each split, load data
        for split in splits_info:
            csv_path = split['csv']
            img_dir = split['img_dir']

            # Load the CSV
            df = pd.read_csv(csv_path)

            # Create file paths, labels, domains
            for _, row in df.iterrows():
                img_file = row['id_code']
                label = row['diagnosis']

                fpath = img_dir / f'{img_file}.png'

                if not fpath.is_file():
                    print(f"Not found, skipped: {fpath}")
                    continue

                self.fpaths.append(fpath)
                self.labels.append(label)
                self.domains.append(self.DATASET_IDS['aptos'])

    def _load_deepdr(self) -> None:
        """
        Loads the DeepDR dataset using 'patient_DR_Level' as the label.
        """
        csv_path = self.root / 'deepdr' / 'regular-fundus-training.csv'

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            img_rel_path = row['image_path'].replace("\\", "/")  
            fpath = self.root / 'deepdr' / img_rel_path[1:]
            
            label = row['patient_DR_Level']

            if np.isnan(label):
                print(f"NaN label, skipped: {fpath}")
                continue

            label = int(label)

            if not fpath.is_file():
                print(f"Not found, skipped: {fpath}")
                continue

            self.fpaths.append(fpath)
            self.labels.append(label)
            self.domains.append(self.DATASET_IDS['deepdr'])
    
    def _load_idrid(self) -> None:
        """
        Loads the IDRiD dataset into self.fpaths, self.labels, self.domains
        """
        csv_path = self.root / 'idrid' / 'idrid_labels.csv'
        img_dir = self.root / 'idrid' / 'Imagenes' / 'Imagenes'

        # Load CSV
        df = pd.read_csv(csv_path)

        # Use only id_code and diagnosis
        for _, row in df.iterrows():
            img_file = row['id_code'] + ".jpg"  # because files are IDRiD_001.jpg etc
            label = row['diagnosis']
            fpath = img_dir / img_file

            if not fpath.is_file():
                print(f"Not found, skipped: {fpath}")
                continue

            self.fpaths.append(fpath)
            self.labels.append(label)
            self.domains.append(self.DATASET_IDS['idrid'])

    def _load_messidor(self) -> None:
        """
        Loads the Messidor dataset into self.fpaths, self.labels, self.domains

        4 Images do not contain label
        Nan label `(nan)`, skipped: 20060411_58550_0200_PP.png
        Nan label `(nan)`, skipped: IM002385.JPG
        Nan label `(nan)`, skipped: IMAGES/IM003718.JPG
        Nan label `(nan)`, skipped: M004176.JPG
        """
        csv_path = self.root / 'messidor' / 'messidor_data.csv'
        img_dir = self.root / 'messidor' / 'IMAGES'

        # Load CSV
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            img_file = row['image_id']  # already has .png
            label = row['adjudicated_dr_grade']
            fpath = img_dir / img_file.replace(".jpg", ".JPG")  # csv has .jpg but files are uppercase

            if np.isnan(label):
                print(f"Nan label `({label})`, skipped: {fpath}")
                continue

            label = int(label)

            if not fpath.is_file():
                print(f"Not found, skipped: {fpath}")
                continue

            if label not in set([0, 1, 2, 3, 4]):
                print(f"Wrong label `({label})`, skipped: {fpath}")
                continue
            
            self.fpaths.append(fpath)
            self.labels.append(label)
            self.domains.append(self.DATASET_IDS['messidor'])