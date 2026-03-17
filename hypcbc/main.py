from pathlib import Path
import argparse

# Import own scripts
from hypcbc.helper import parse_cli_overrides, print_config
from hypcbc.helper import seed_everything, get_run_id
from hypcbc.data.registry import DATASET_CLASSES
from hypcbc.config.manager import ConfigManager
from hypcbc.model.trainer import TrainerModule
from hypcbc.model.model import ModelModule
from hypcbc.data.data import DataModule


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Train and evaluate experiments from YAML configs.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            'Examples:\n'
            '  uv run hypcbc-train --config config/baseline.yaml\n'
            '  uv run hypcbc-train --config config/baseline.yaml --override config/methods/ce.yaml\n'
            '  uv run hypcbc-train --config config/baseline.yaml --set optimizer.lr=0.0003'
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
        help='Inline override (repeatable), e.g., --set data.batch_size=64'
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
    
    # Set random seed for reproducibility
    seed_everything(config.trainer.seed)

    # Device
    device = config.trainer.device

    # === Load model === #
    print(f"Loading model...")
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


    # === Train and Evaluate === #
    if config.data.mode != 'emb':
        raise ValueError("`data.mode=emb` is not supported with main.py")
    
    run_id = get_run_id(config)

    trainer = TrainerModule(
        config = config.trainer,
        config_optim = config.optimizer,
        model_module = model,
        data_module = data,
        experiment_name = config.experiment_name, 
        run_id = run_id
    )

    print(f"Initializing trainer")      
    trainer.setup()

    # Train
    if not config.trainer.skip_train:
        print(f"Training...")      
        trainer.train()

    print(f"Evaluating...")      
    
    # Eval on the val and test set
    trainer.evaluate_only(ckpt_id = 'best')
    print(f"Evaluation completed")

    if config.trainer.eval_domain_accuracy:
        print(f"Evaluating domain accuracy...")
        trainer.evaluate_domain_acc()


        


if __name__ == '__main__':
    main()

    

    
        

    
