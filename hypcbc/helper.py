from typing import Any, Dict, List

import json
import os
import numpy as np
import random
import torch

# === Config Helpers === #
def parse_cli_overrides(override_strings: List[str]) -> Dict[str, Any]:
    """Parse CLI override strings into nested dictionary."""
    overrides = {}

    for override_str in override_strings:
        if '=' not in override_str:
            raise ValueError(f"Invalid override format: {override_str}. Use key=value")

        key, value = override_str.split('=', 1)

        # Parse the value
        try:
            # Try to parse as JSON first (handles booleans, numbers, lists, etc.)
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            # If JSON parsing fails, treat as string
            parsed_value = value

        # Handle nested keys (e.g., "data.batch_size=32")
        keys = key.split('.')
        current_dict = overrides

        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]

        current_dict[keys[-1]] = parsed_value

    return overrides


def print_config(config: Dict):
    print('\n' + '=' * 50)
    print('FINAL CONFIGURATION')
    print('=' * 50)
    import yaml
    print(yaml.dump(config.to_dict(), default_flow_style = False))
    print('=' * 50)


# === Reproducibility === #
def seed_everything(seed: int) -> None:
    """
    Set random seed to ensure reproducibility.

    :param seed: random seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed to: {seed}.")


# === Run Naming === #
def get_run_id(config: dict) -> str:
    """Create a unique identifier for the current run.
    This will be used for wandb, ckpts, and final logs.

    :param config: Configuration dictionary.
    :return: Unique identifier for the current run.
    """
    
    # Dataset-id
    if config.data.target_domain is None:
        dataset_id = config.data.dataset
    else:
        dataset_id = f'{config.data.dataset}_{str(config.data.target_domain)}'

    # Dimension id
    if config.model.branch2_dim:
        dimensions_id = f'{config.model.branch1_dim}-{config.model.branch2_dim}'
    else:
        dimensions_id = f'{config.model.branch1_dim}'

    # Loss id
    if config.trainer.loss == 'dist':
        loss_id = f'dist-{config.trainer.dist_lam}l-{config.trainer.dist_temp}t'
    else:
        loss_id = config.trainer.loss

    if config.model.manifold == 'hyp' and config.model.hyp_curvature != 1.0:
        manifold_id = f'hyp-{config.model.hyp_curvature}'
    else:
        manifold_id = config.model.manifold

    # Compose ID
    name = '_'.join([
        dataset_id,
        manifold_id,
        dimensions_id,
        loss_id,
        config.model.backbone_id,
        config.data.augmentation if config.data.augmentation is not None else 'noaug',
        f"{config.optimizer.lr}lr",
        f"{config.data.batch_size}bs",
        f"{config.trainer.seed}s"
    ])
    print(f"ID of the current run: {name}.")
    return name


# === Metrics === #
class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count