from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import own scripts
from hypcbc.config.trainer import TrainerConfig
from hypcbc.config.optimizer import OptimizerConfig
from hypcbc.model.model import ModelModule
from hypcbc.data.data import DataModule
from hypcbc.helper import AverageMeter
from hypcbc.model.loss import CrossEntropyLoss


class TrainerModule:
    """TrainerModule class for training and evaluation."""
    
    def __init__(
        self,
        config: TrainerConfig,
        config_optim: OptimizerConfig,
        model_module: ModelModule,
        data_module: DataModule,
        experiment_name: str,
        run_id: str
    ):
        self.config = config
        self.config_optim = config_optim
        self.model_module = model_module
        self.data_module = data_module
        self.run_id = run_id
        
        # Training components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler] = None
        self.criterion: Optional[nn.Module] = None
        self.ce_loss = CrossEntropyLoss()  # Used for val
        
        # Training state
        self.best_epoch = 0
        self.best_model: Optional[nn.Module] = None
        self.best_val_metric = np.double('inf') if config.monitor_mode == 'min' else np.double('-inf')
        
        # Paths
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.output_dir = Path(config.output_root / experiment_name / run_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self) -> None:
        """Initialize training components."""
        # Setup model
        self.model = self.model_module.to(self.config.device)

        # Setup optimizer
        self.optimizer = self._init_optimizer()

        # Setup scheduler
        self.scheduler = self._init_scheduler()
        
        # Setup loss function
        self.criterion = self._init_criterion()

    def train(self) -> None:
        """Train until convergence."""
            
        # Initialize wandb
        if self.config.use_wandb:
            self._init_wandb()
        
        for epoch in range(self.config.max_epochs):
            print(f"Epoch {epoch+1}/{self.config.max_epochs}")

            # Train one epoch
            train_metrics = self.train_one_epoch()
            
            # Evaluate
            val_metrics = self.evaluate(self.data_module.loaders['val'], store_csv=False)
            
            # Manage best model
            # Update global vars: best_val_metric, best_model, best_epoch
            # Eventually store the ckpt (if best)
            self.manage_best_model(epoch, val_metrics)
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Early stopping check
            if (epoch - self.best_epoch) >= self.config.patience:
                print("Early stopping!")
                break
        
        # Save final model
        self._save_final_model(epoch)
        
        # Log best results
        if self.config.use_wandb:
            wandb.log({
                'best_epoch': self.best_epoch + 1, 
                f'best_val_{self.config.monitor}': self.best_val_metric
            })

    def train_one_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        # Track loss values
        loss_meters: Dict[str, AverageMeter] = {}

        self.model.train()
        
        for batch in tqdm(self.data_module.loaders['train']):
            if len(batch) == 2:
                input, target = batch[:2]  # Handles both (input, target) and (input, target, domain)
            else:
                input, target, domain = batch
                domain = domain.to(self.config.device)

            input, target = input.to(self.config.device), target.to(self.config.device)
            
            # Handle both single branch and multi-branch
            # For multi-branch, returns a tuple     
            # If `return_proj == True`, returns a tuple with logits, features
            # If `return_proj == True` and it's multi-branch, logits and features are also tuples
            # .. containing both branch data
            output = self.model(input, return_proj=self.config.loss_uses_features) 
            
            if self.config.loss_uses_domain:
                target = (domain, target)
                
            loss, branch_losses = self.criterion(output, target)
            
            # Dynamically track all losses returned
            for key, value in branch_losses.items():
                if key not in loss_meters:
                    loss_meters[key] = AverageMeter()
                loss_meters[key].update(value)

            from hypcbc.model.loss import DANNLoss
            if not (isinstance(self.criterion, DANNLoss) and self.criterion.is_disc_step):
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                
        if self.scheduler:
            self.scheduler.step()

        # Prepare output
        return {name: meter.average for name, meter in loss_meters.items()}

    def evaluate(self, dataloader: DataLoader, store_csv: bool = False) -> Tuple[Dict[str, float], Dict[str, List]]:
        """Evaluate model on given dataloader."""
        # Track loss values
        loss_meters: Dict[str, AverageMeter] = {}
        # Init output information
        domains, predictions, references, max_probabilities, all_probabilities = [], [], [], [], []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:

                # Extract batch information and put them into `device`
                if len(batch) == 2:
                    input, target = batch
                    domains.extend([0] * len(input))  # Default domain index
                else:
                    input, target, domain = batch
                    domain = domain.to(self.config.device)
                    domains.extend(domain.cpu().tolist())

                input, target = input.to(self.config.device), target.to(self.config.device)

                # Inference - handle both single branch and multi-branch
                # For multi-branch, returns a tuple (branch1_logits, branch2_logits)
                output = self.model(input)

                # Compute CE loss
                _, branch_losses = self.ce_loss(output, target)
                
                # Dynamically track all losses returned
                for key, value in branch_losses.items():
                    if key not in loss_meters:
                        loss_meters[key] = AverageMeter()
                    loss_meters[key].update(value)

                # If two-branches, classify using the first one
                class_logits = output if not isinstance(output, tuple) else output[0]

                # Compute probabilities and predictions
                probs = F.softmax(class_logits, dim=1)
                max_prob, pred = torch.max(probs, dim=1)

                # Store batch results
                predictions.extend(pred.cpu().tolist())
                references.extend(target.cpu().tolist())
                max_probabilities.extend(max_prob.cpu().tolist())
                all_probabilities.extend(probs.cpu().tolist())

        # Compute metrics
        accuracy = accuracy_score(references, predictions)
        bacc = balanced_accuracy_score(references, predictions)
        all_probabilities = np.array(all_probabilities)

        # Calculate AUC
        if len(set(references)) == 2:
            positive_class_probs = [prob[1] for prob in all_probabilities]
            auc_score = roc_auc_score(references, positive_class_probs)
        else:
            from sklearn.preprocessing import label_binarize
            references_binarized = label_binarize(references, classes=range(len(set(references))))  # Binarize labels for multi-class
            auc_score = roc_auc_score(references_binarized, all_probabilities, multi_class='ovr')

        # Group information
        eval_info = {
            'accuracy': accuracy, 
            'bacc': bacc, 
            'auc': auc_score, 
            **{name: meter.average for name, meter in loss_meters.items()}
        }
        
        store_info = {
            'domains': domains, 
            'labels': references, 
            'predictions': predictions, 
            'probabilities': max_probabilities
        }

        if store_csv:
            return eval_info, store_info
        else:
            return eval_info

    def store_predictions(self, predictions_info: Dict[str, List], filename: str) -> None:
        """Store predictions in CSV format."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            for idx in range(len(predictions_info['labels'])):
                line = ','.join([
                    str(predictions_info['domains'][idx]),
                    str(predictions_info['labels'][idx]),
                    str(int(predictions_info['predictions'][idx])),
                    f"{predictions_info['probabilities'][idx]:.4f}"
                ])
                f.write(line + '\n')

    def manage_best_model(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        """Handle the best model checkpoint."""
        current_metric = val_metrics[self.config.monitor]
        
        if self.config.monitor_mode == "min":
            is_best = current_metric < self.best_val_metric
        else:
            is_best = current_metric > self.best_val_metric
            
        if is_best:
            print(f"New best model found with val {self.config.monitor}: {current_metric:.4f}")
            self.best_val_metric = current_metric
            self.best_model = self.model
            self.best_epoch = epoch
            
            # Save best model checkpoint
            self.save_ckpt("best_model.pt", epoch)

    def save_ckpt(self, filename: str, epoch: int) -> None:
        """Save model checkpoint."""
        filepath = self.output_dir / filename
        
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'opt_state_dict': self.optimizer.state_dict(),
        }

        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(state, filepath)

    def evaluate_only(self, ckpt_id: str = "best") -> None:
        """Evaluate only mode - load best model and test."""
        assert ckpt_id in ['best', 'last']
        
        best_ckpt_path = self.output_dir / f"{ckpt_id}_model.pt"
        
        if not best_ckpt_path.exists():
            raise FileNotFoundError(f"Best model checkpoint not found at {best_ckpt_path}")
        
        # Load best model
        checkpoint = torch.load(best_ckpt_path, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        # Evaluate on validation and test sets
        train_metrics, train_info = self.evaluate(self.data_module.loaders['train'], store_csv=True)
        val_metrics, val_info = self.evaluate(self.data_module.loaders['val'], store_csv=True)
        test_metrics, test_info = self.evaluate(self.data_module.loaders['test'], store_csv=True)
        
        print(f'Val accuracy: {val_metrics["accuracy"]:.6f}')
        print(f'Val balanced accuracy: {val_metrics["bacc"]:.6f}')
        print(f'Test accuracy: {test_metrics["accuracy"]:.6f}')
        print(f'Test balanced accuracy: {test_metrics["bacc"]:.6f}')
        
        # Store predictions
        self.store_predictions(train_info, "best_train_info.csv")
        self.store_predictions(val_info, "best_val_info.csv")
        self.store_predictions(test_info, "best_test_info.csv")
        
        # Store embeddings if requested
        if self.model_module.config.extract_projections:
            self._extract_and_store_embeddings()
        
        # Save results
        self._save_results(train_metrics, val_metrics, test_metrics)

    def evaluate_domain_acc(self, classifier: str = 'linear') -> None:
        """Evaluate domain accuracy on train embeddings.

        `classifier` options:
        - `linear`: Logistic Regression (default)
        - `nonlinear`: MLP classifier
        - `both`: run and store both variants
        """
        from sklearn.preprocessing import LabelEncoder

        if classifier not in {'linear', 'nonlinear', 'both'}:
            raise ValueError(
                f"Unsupported classifier '{classifier}'. "
                "Valid options: 'linear', 'nonlinear', 'both'."
            )

        # Load embeddings
        train_emb_path = self.output_dir / 'embeddings' / 'best_train_embeddings.pt'
        db_train = torch.load(train_emb_path, weights_only=False)
        x_train, d_train = db_train[0]['branch1'], db_train[2]

        test_emb_path = self.output_dir / 'embeddings' / 'best_test_embeddings.pt'
        db_test = torch.load(test_emb_path, weights_only=False)
        x_test, d_test = db_test[0]['branch1'], db_test[2]

        d_train = d_train.reshape(-1,)
        d_test = d_test.reshape(-1,)

        # If domain labels are not ordered integers, encode them.
        if 0 not in set(d_test):
            le = LabelEncoder()
            d_train = le.fit_transform(d_train)
            d_test = le.transform(d_test)

        results = {}
        classifiers_to_run = ['linear', 'nonlinear'] if classifier == 'both' else [classifier]

        for clf_name in classifiers_to_run:
            if clf_name == 'linear':
                y_pred, y_pred_proba = self._domain_classifier_linear(x_train, d_train, x_test)
            else:
                y_pred, y_pred_proba = self._domain_classifier_nonlinear(x_train, d_train, x_test)

            acc, bacc, auc = self._compute_domain_metrics(d_test, y_pred, y_pred_proba)
            results[f'{clf_name}_domain_acc'] = acc
            results[f'{clf_name}_domain_bacc'] = bacc
            results[f'{clf_name}_domain_auc'] = auc

        # Store results
        results['domain_classifier_mode'] = classifier
        results_file = self.output_dir / 'results_domain_classification.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        print('Domain Accuracy results stored at: ', results_file)

    def _domain_classifier_linear(
        self,
        x_train: np.ndarray,
        d_train: np.ndarray,
        x_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linear domain classifier using Logistic Regression."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        pipeline.fit(x_train, d_train)
        y_pred = pipeline.predict(x_test)
        y_pred_proba = pipeline.predict_proba(x_test)
        return y_pred, y_pred_proba

    def _domain_classifier_nonlinear(
        self,
        x_train: np.ndarray,
        d_train: np.ndarray,
        x_test: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-4,
        batch_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Non-linear domain classifier using a small MLP."""

        class DomainMLP(nn.Module):
            def __init__(self, input_dim: int, num_classes: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert to tensors
        x_train_t = torch.tensor(x_train, dtype=torch.float32)
        d_train_t = torch.tensor(d_train, dtype=torch.long)
        x_test_t = torch.tensor(x_test, dtype=torch.float32)

        input_dim = x_train_t.shape[1]
        num_classes = int(d_train_t.max().item()) + 1

        model = DomainMLP(input_dim, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(x_train_t, d_train_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Train
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(x_test_t.to(device))
            y_pred_proba = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = y_pred_proba.argmax(axis=1)

        return y_pred, y_pred_proba

    def _compute_domain_metrics(
        self,
        d_test: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute accuracy, balanced accuracy, and ROC-AUC for domain classification."""
        from sklearn.preprocessing import label_binarize

        acc = accuracy_score(d_test, y_pred)
        bacc = balanced_accuracy_score(d_test, y_pred)

        if len(set(d_test)) == 2:
            auc = roc_auc_score(d_test, y_pred_proba[:, 1])
        else:
            d_test_binarized = label_binarize(d_test, classes=range(len(set(d_test))))
            auc = roc_auc_score(d_test_binarized, y_pred_proba, multi_class='ovr')

        return acc, bacc, auc
             
    def _init_optimizer(self) -> optim.Optimizer:
        """Initialize the optimizer"""

        if self.config_optim.name == "adamw":
            return optim.AdamW(
                self.model.parameters(), 
                lr=self.config_optim.lr
            )
        else:
            raise ValueError(f"optimizer `{self.config_optim.name}` not supported")

    def _init_scheduler(self) -> optim.lr_scheduler:
        """Initialize the scheduler"""
        if self.config_optim.scheduler and self.config_optim.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.max_epochs,
                eta_min=self.config_optim.min_lr
            )
        else:
            raise ValueError(f"scheduler `{self.config_optim.scheduler}` not supported")
    
    def _init_criterion(self) -> nn.Module:
        from hypcbc.data.registry import DATASET_TRAIN_DOMAINS

        if self.config.loss == "ce":
            from hypcbc.model.loss import CrossEntropyLoss
            return CrossEntropyLoss()
        
        elif self.config.loss == "dist" and self.config.dist_lam and self.config.dist_temp:
            from hypcbc.model.loss import DistillationLoss
            return DistillationLoss(
                lambda_=self.config.dist_lam,
                temperature=self.config.dist_temp
            )
        
        elif self.config.loss == "irm" and self.config.irm_lambda and self.config.irm_anneal_iters:
            from hypcbc.model.loss import IRMLoss
            if self.model_module._hidden_branch2:
                raise ValueError("IRM doesn't support a multi-branch strategy.")

            return IRMLoss(
                lambda_ = self.config.irm_lambda,
                anneal_iters = self.config.irm_anneal_iters
            )
        
        elif self.config.loss == "gdro":
            from hypcbc.model.loss import GroupDROLoss
            if self.model_module._hidden_branch2:
                raise ValueError("GroupDRO doesn't support a multi-branch strategy.")
            
            return GroupDROLoss(
                n_groups = DATASET_TRAIN_DOMAINS[self.data_module.config.dataset],
                eta = self.config.gdro_eta
            )

        elif self.config.loss == "vrex" and self.config.vrex_lambda and self.config.vrex_anneal_iters:
            from hypcbc.model.loss import VRExLoss

            if self.model_module._hidden_branch2:
                raise ValueError("VREx doesn't support a multi-branch strategy.")
            
            return VRExLoss(
                lambda_= self.config.vrex_lambda,
                anneal_iters=self.config.vrex_anneal_iters
            )
        
        elif self.config.loss == "coral" and self.config.mmd_gamma:
            from hypcbc.model.loss import CORALLoss

            if self.model_module._hidden_branch2:
                raise ValueError("CORAL doesn't support a multi-branch strategy.")
            
            return CORALLoss(gamma = self.config.mmd_gamma)
        
        elif self.config.loss == "mmd" and self.config.mmd_gamma:
            from hypcbc.model.loss import MMDLoss

            if self.model_module._hidden_branch2:
                raise ValueError("MMD doesn't support a multi-branch strategy.")
            
            return MMDLoss(gamma = self.config.mmd_gamma)
        
        elif self.config.loss == "dann":
            from hypcbc.model.loss import DANNLoss

            if self.model_module._hidden_branch2:
                raise ValueError("DANN doesn't support a multi-branch strategy.")
            
            # data_module, num_classes
            from hypcbc.data.registry import DATASET_TRAIN_DOMAINS_FULL, DATASET_CLASSES
            dataset_str = self.data_module.config.dataset
            return DANNLoss(domains = DATASET_TRAIN_DOMAINS_FULL[dataset_str], 
                            num_classes = DATASET_CLASSES[dataset_str])
        
        elif self.config.loss == "cdann":
            from hypcbc.model.loss import DANNLoss

            if self.model_module._hidden_branch2:
                raise ValueError("CDANN doesn't support a multi-branch strategy.")
            
            # data_module, num_classes
            from hypcbc.data.registry import DATASET_TRAIN_DOMAINS_FULL, DATASET_CLASSES
            dataset_str = self.data_module.config.dataset
            return DANNLoss(domains = DATASET_TRAIN_DOMAINS_FULL[dataset_str], 
                            num_classes = DATASET_CLASSES[dataset_str],
                            conditional = True,
                            class_balance = True)

        else:
            raise ValueError(f"loss `{self.config.loss}` not supported")

    def _reset_optimizer_and_scheduler(self) -> None:
        """Reset optimizer and scheduler (IRM annealing jump-safe)"""
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        print("Optimizer and scheduler reset after IRM penalty annealing.")

    def _extract_and_store_embeddings(self) -> None:
        """Extract and store feature embeddings."""
        embeddings_dir = self.output_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)

        for split in ['train', 'val', 'test']:
            if split in self.data_module.loaders:
                embeddings, labels, domains = self._extract_features(
                    self.data_module.loaders[split],
                    two_branch= (self.model_module._hidden_branch2 is not None)
                )
                
                filepath = embeddings_dir / f"best_{split}_embeddings.pt"
                torch.save((embeddings, labels, domains), filepath)
                print(f"Embeddings saved at: {filepath}")
        
    def _extract_features(self, dataloader: DataLoader, two_branch: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract feature embeddings from model."""
        
        embeddings_dict = {'branch1': []}
        if two_branch:
            embeddings_dict['branch2'] = []
        
        labels, domains = [], []
        self.model.eval()


        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if len(batch) == 2:
                    input, target = batch
                    domains.extend([idx] * len(input)) # placeholder
                else:
                    input, target, domain = batch
                    domains.extend(domain.cpu().numpy())

                input, target = input.to(self.config.device), target.to(self.config.device)
                _, z = self.model(input, return_proj=True)

                if two_branch:
                    embeddings_dict['branch1'].extend(z[0].cpu().numpy())
                    embeddings_dict['branch2'].extend(z[1].cpu().numpy())
                else:
                    embeddings_dict['branch1'].extend(z.cpu().numpy())

                labels.extend(target.cpu().numpy())
        
        # Clean output
        output = {k: np.array(v) for k, v in embeddings_dict.items()}
        return output, np.array(labels), np.array(domains)
    
    def _init_wandb(self) -> None:
        """Initialize wandb logging."""
        exp_name = str(self.output_dir.parent.name)
        run_name = str(self.output_dir.name)
        
        wandb.init(
            project="hypDIST",
            name=f'exp_{exp_name}__{run_name}',
            config=self.config.__dict__,
            mode="online"
        )

    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log training metrics."""
        # Prepare log stats
        log_stats = {
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }

        # Log to wandb
        if self.config.use_wandb:
            wandb.log(log_stats)

        # Console log
        row_log = f"Epoch {str(epoch+1).zfill(3)}: " + "\t".join([f"{k}:{v:.6f}" for k, v in log_stats.items()])
        print(row_log)

        # File log
        log_file = self.output_dir / "log.txt"
        with open(log_file, 'a') as f:
            f.write(row_log + "\n")

    def _save_final_model(self, epoch: int) -> None:
        """Save the final model."""
        self.save_ckpt("last_model.pt", epoch)
        print(f"Final model saved at: {self.output_dir / 'last_model.pt'}")

    def _save_results(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> None:
        """Save final results to JSON."""
        results = {
            'accuracy': test_metrics["accuracy"],
            'balanced_accuracy': test_metrics["bacc"],
            'auc': test_metrics["auc"],
            'val_accuracy': val_metrics["accuracy"],
            'val_balanced_accuracy': val_metrics["bacc"],
            'val_auc': val_metrics["auc"],
            'train_accuracy': train_metrics["accuracy"],
            'train_balanced_accuracy': train_metrics["bacc"],
            'train_auc': train_metrics["auc"],
        }
        
        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)    
        print(f'Final log stored at: {results_file}')
