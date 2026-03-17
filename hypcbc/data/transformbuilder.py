from torchvision import transforms
from typing import Optional, Dict


def build_transforms(
    mode: str,
    backbone_transform: transforms.Compose,
    augmentation: Optional[str],
    dataset_name: str
) -> Dict[str, Optional[transforms.Compose]]:
    """Build train/val transforms from mode, backbone transform, and augmentation."""

    # Embeddings do not require transform
    if mode == "emb":
        return {"train": None, "val": None}

    # Default augmentation
    if augmentation is None:
        return {"train": backbone_transform, "val": backbone_transform}

    # Keep backbone resize/crop and normalization stack, and inject augmentation in between.
    resize, crop, *rest = backbone_transform.transforms

    # === Augmentations === #
    if augmentation == "augmix":
        train_tfms = transforms.Compose([
            resize, crop,
            transforms.AugMix(severity=3, mixture_width=3, alpha=0.2),
            *rest,
        ])
    elif augmentation == "randaugment":
        train_tfms = transforms.Compose([
            resize, crop,
            transforms.RandAugment(num_ops=1),
            *rest,
        ])
    elif augmentation == "augmedmnistc":
        from medmnistc.augmentation import AugMedMNISTC
        from medmnistc.corruptions.registry import CORRUPTIONS_DS

        if dataset_name.startswith("fitz"):
            medmnist_equivalent = "dermamnist"
        elif dataset_name.startswith("camelyon"):
            medmnist_equivalent = "pathmnist"
        elif dataset_name.startswith("retina"):
            medmnist_equivalent = "retinamnist"
        else:
            raise ValueError(
                "Dataset not supported for AugMedMNISTC. "
                f"Got '{dataset_name}', expected prefixes: fitz*, camelyon*, retina*."
            )

        corrs = CORRUPTIONS_DS[medmnist_equivalent].copy()
        # Drop motion blur to avoid expensive per-sample latency during training.
        corrs.pop("motion_blur", None)
        train_tfms = transforms.Compose([
            resize, crop,
            AugMedMNISTC(corrs),
            *rest,
        ])
    else:
        raise ValueError(
            f"Augmentation '{augmentation}' not supported. "
            "Valid options: augmix, randaugment, augmedmnistc."
        )

    return {"train": train_tfms, "val": backbone_transform}
