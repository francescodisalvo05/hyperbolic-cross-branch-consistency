from pathlib import Path
import argparse
import torch

# Import own scripts
from hypcbc.helper import parse_cli_overrides, print_config
from hypcbc.helper import seed_everything
from hypcbc.data.registry import DATASET_CLASSES
from hypcbc.config.manager import ConfigManager
from hypcbc.model.model import ModelModule
from hypcbc.data.data import DataModule


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Precompute and store embedding databases from YAML configs.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            'Examples:\n'
            '  uv run hypcbc-preprocess --config config/create_db.yaml\n'
            '  uv run hypcbc-preprocess --config config/create_db.yaml --set data.dataset=camelyon17\n'
            '  uv run hypcbc-preprocess --config config/create_db.yaml --set model.backbone_id=dinov2_small'
        )
    )

    parser.add_argument(
        '--config', type=Path, required=True,
        help='Path to base configuration file (YAML)'
    )
    parser.add_argument(
        '--override', action='append', type=Path, default=[],
        help='Path to override configuration file (repeatable)'
    )
    parser.add_argument(
        '--set',  action='append', default=[],
        metavar='KEY=VALUE',
        help='Inline override (repeatable), e.g., --set data.batch_size=32'
    )
    parser.add_argument(
        '--print-config', action='store_true',
        help='Print final merged configuration and exit'
    )
    args = parser.parse_args()
    return args


def main():
    # === Init run === #
    args = parse_arguments()

    # Load overrides from CLI
    cli_overrides = parse_cli_overrides(args.set) if args.set else None
    
    # Create ConfigManager class based on YAML + CLI
    config = ConfigManager.load_with_overrides(
        base_config_path=args.config,
        override_paths=args.override if args.override else None,
        cli_overrides=cli_overrides
    )

    # Print config, if requested
    if args.print_config:
        print_config(config)
        return
    
    # To create augmented embeddings, we do utilize an independent seed
    # Note: In such cases, we store a larger augmented set (5x original size) 
    seed_everything(62347625)

    # Set device
    device = config.model.device

    # === Load model === #
    print(f"Loading backbone: {config.model.backbone_id}")
    model = ModelModule(config = config.model)
    config.model.num_classes = DATASET_CLASSES[config.data.dataset]
    model.setup()  
    model.to(device)


    # === Load dataset === #
    print(f"Loading dataset: {config.data.dataset}")
    data = DataModule(
        config = config.data,
        backbone = config.model.backbone_id,
        backbone_transform = model.transform
    )
    
    data.setup()
    dataloaders = data.loaders
    print(f"Dataset loaded with {len(dataloaders)} splits = {list(dataloaders.keys())}")
    

    # === Create embedding database === #
    print("Setting up feature database...")
    db_path: Path = config.data.database_root / config.data.dataset / config.model.backbone_id
    db_path.mkdir(parents=True, exist_ok=True)

    for split, loader in dataloaders.items():

        if config.data.augmentation and split != 'train':
            continue

        if config.data.augmentation:
            filepath = db_path / f'{split}_{config.data.augmentation}.pt'
        else:
            filepath = db_path / f'{split}.pt'

        features_data = model.extract_features(loader, device=device)
        torch.save(features_data, filepath)
        print(f"Features saved to {filepath}")

    print("Feature database ready!")
    # === Finished preprocessing === #


if __name__ == '__main__':
    main()
    
        

    
