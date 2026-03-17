from torchvision import transforms
from typing import Dict, Optional
from pathlib import Path
import torch

# Import own scripts
from hypcbc.config.data import DataConfig


def build_dataset(
    config: DataConfig,
    backbone: str,
    transforms: Dict[str, Optional[transforms.Compose]]
) -> Dict[str, torch.utils.data.Dataset]:
    """Build `train`/`val`/`test` datasets for embedding or raw-image mode."""

    # === Embedding datasets === #
    if config.mode == "emb":
        from hypcbc.dataset.Database import EMBDataset, EMBDatasetID

        def _make_path(ds_key: str) -> Path:
            # For domain-specific runs, embeddings are stored in a domain-suffixed directory.
            if config.target_domain:
                return config.database_root / f"{ds_key}_{config.target_domain}" / backbone
            return config.database_root / ds_key / backbone

        # Handle in-distribution versions of Camelyon17 and Retina
        # i.e., stack train/val/test (inter-institutional) and create an ID-split domain-wise
        if config.dataset in ("camelyon17_id", "retina_id"):
            # Use the default database root
            root = config.database_root / config.dataset.replace("_id", "") / backbone
            return {s: EMBDatasetID(root=root, split=s) for s in ("train", "val", "test")}

        # Default path handling for MedMNIST and OOD settings.
        root = _make_path(config.dataset)
        return {s: EMBDataset(root=root, split=s, augmentation=config.augmentation if s == "train" else None) for s in ("train", "val", "test")}

    # === Raw datasets === #

    # Initialize raw datasets
    datasets = {}

    # MedMNIST
    if config.dataset in {
        "breastmnist", "retinamnist", "pneumoniamnist", "dermamnist", "bloodmnist",
        "organcmnist", "organsmnist", "organamnist", "pathmnist", "octmnist", "tissuemnist"
    }:
        from medmnist import INFO
        import medmnist
        info = INFO[config.dataset]
        ds_class = getattr(medmnist, info["python_class"])
        root = config.data_root / "medmnist"
        kwargs = dict(as_rgb=True, size=224, mmap_mode="r")
        datasets = {
            split: _prepare_medmnist(ds_class, root, split, transforms["train" if split == "train" else "val"], **kwargs)
            for split in ("train", "val", "test")
        }

    # Camelyon17-Wilds
    elif config.dataset == "camelyon17":
        from wilds import get_dataset as wilds_get
        from hypcbc.dataset.Camelyon import Camelyon17Wrapper
        raw = wilds_get("camelyon17", root_dir=str(config.data_root / "camelyon17"))
        datasets = {
            split: Camelyon17Wrapper(raw.get_subset(split, transform=transforms["train" if split == "train" else "val"]))
            for split in ("train", "val", "test")
        }

    # Retina dataset (train=[APTOS Dataset, DeepDR], val=[IDRID], test=[MESSIDOR])
    elif config.dataset == "retina":
        from hypcbc.dataset.Retina import RetinaDataset
        root = config.data_root / "retina"
        datasets = {
            split: RetinaDataset(root, split, transforms["train" if split == "train" else "val"])
            for split in ("train", "val", "test")
        }

    # Fitzpatrick17k
    elif config.dataset.startswith("fitzpatrick17k"):
        from hypcbc.dataset.Fitzpatrick import Fitzpatrick17k
        root = config.data_root / "fitzpatrick17k"
        target = config.dataset.split("_")[1]    # 'id', '12', '34', '56'
        datasets = {
            split: Fitzpatrick17k(
                root, split=split, target_domain=target,
                transform=transforms["train" if split == "train" else "val"]
            ) for split in ("train", "val", "test")
        }
    
    else:
        raise ValueError(
            f"Unknown dataset '{config.dataset}'. Supported groups: "
            "MedMNIST variants, camelyon17, retina, fitzpatrick17k_*."
        )

    if config.augmentation:
        from hypcbc.dataset._augmented import AugmentedDatasetWrapper
        datasets["train"] = AugmentedDatasetWrapper(datasets["train"], n_augments=config.augmentation_factor)

    return datasets


def _prepare_medmnist(
    cls,
    root: Path,
    split: str,
    transform: Optional[transforms.Compose],
    **kwargs
) -> torch.utils.data.Dataset:
    """Load one MedMNIST split and flatten labels to shape `(N,)`."""
    ds = cls(root=str(root), split=split, transform=transform, **kwargs)
    ds.labels = ds.labels.reshape(-1,)
    return ds
