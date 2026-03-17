import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch import nn
from typing import Dict, Tuple


class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError


class CrossEntropyLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, output, target) -> Tuple[torch.Tensor, Dict[str, float]]:
        metrics = {}

        if not isinstance(output, tuple):  # single-branch
            logits = output
            loss = F.cross_entropy(logits, target)
            metrics["loss_branch1"] = loss.item()
            metrics["loss"] = loss.item()
            return loss, metrics

        # two-branch
        logits_branch1, logits_branch2 = output
        ce_branch1 = F.cross_entropy(logits_branch1, target)
        ce_branch2 = F.cross_entropy(logits_branch2, target)
        total_loss = ce_branch1 + ce_branch2

        metrics["ce_branch1"] = ce_branch1.item()
        metrics["ce_branch2"] = ce_branch2.item()
        metrics["loss"] = total_loss

        return total_loss, metrics


class DistillationLoss(BaseLoss):
    def __init__(self, lambda_: float = 1.0, temperature: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
        self.temperature = temperature

    def forward(self, output, target) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits_branch1, logits_branch2 = output

        ce_branch1 = F.cross_entropy(logits_branch1, target)
        ce_branch2 = F.cross_entropy(logits_branch2, target)

        # Soft distillation
        T = self.temperature
        soft_teacher = F.softmax(logits_branch2 / T, dim=-1)
        soft_student = F.log_softmax(logits_branch1 / T, dim=-1)
        
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

        total_loss = ce_branch1 + ce_branch2 + self.lambda_ * kd_loss

        return total_loss, {
            "ce_branch1": ce_branch1.item(),
            "ce_branch2": ce_branch2.item(),
            "kd_loss": kd_loss.item(),
            "loss": total_loss.item()
        }


class IRMLoss(BaseLoss):
    def __init__(self, lambda_: float = 1.0, anneal_iters: int = 500):
        super().__init__()
        self.initial_lambda = lambda_
        self.lambda_ = 1.0  # will be annealed
        self.anneal_iters = anneal_iters
        self.step = 0  # updated externally (from Trainer)

    @staticmethod
    def _compute_irm_penalty(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute IRM penalty: squared gradient norm w.r.t. dummy scale"""
        scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
        loss = F.cross_entropy(logits * scale, targets)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def forward(self, output, target) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute IRM loss using domain labels"""

        domains, labels = target
        unique_domains = torch.unique(domains)
        
        total_ce = 0.0
        total_penalty = 0.0
        n_envs = len(unique_domains)

        for d in unique_domains:
            idx = domains == d
            if idx.sum() < 2:         # IRM penalty needs at least 2 samples
                continue
            env_logits = output[idx]  # logits
            env_targets = labels[idx] # labels

            total_ce += F.cross_entropy(env_logits, env_targets)
            
            # === Only compute IRM penalty during training ===
            if self.training:
                total_penalty += self._compute_irm_penalty(env_logits, env_targets)


        # Average over environments
        total_ce /= n_envs
        total_penalty /= n_envs

        # Update annealing
        if self.step >= self.anneal_iters:
            penalty_weight = self.initial_lambda
        else:
            penalty_weight = 1.0

        self.step += 1
        total_loss = total_ce + penalty_weight * total_penalty

        return total_loss, {
            "ce_loss": total_ce.item(),
            "irm_penalty": total_penalty.item(),
            "irm_weight": penalty_weight,
            "loss": total_loss.item()
        }



class GroupDROLoss(BaseLoss):
    def __init__(self, n_groups: int, eta: float = 0.01):
        """
        Args:
            n_groups: Number of unique training domains
            eta: Learning rate for updating group weights
        """
        super().__init__()
        self.n_groups = n_groups
        self.eta = eta

        # Initialized lazily at first forward pass (so it moves with device)
        self.q = None

    def forward(self, output, target) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        domains, labels = target

        device = output.device
        unique_groups = torch.unique(domains)

        # === Initialize group weights on first pass ===
        if self.q is None:
            self.q = torch.ones(self.n_groups, device=device) / self.n_groups

        # === Compute per-group losses ===
        group_losses = torch.zeros(self.n_groups, device=device)
        group_counts = torch.zeros(self.n_groups, device=device)

        #print(unique_groups)

        for domain_id, g in enumerate(unique_groups):
            idx = (domains == g)
            if idx.sum() > 0:
                group_losses[domain_id] = F.cross_entropy(output[idx], labels[idx])
                group_counts[domain_id] = idx.sum()

        #print(group_losses)

        # import sys
        # sys.exit(0)

        # === Only update weights during training ===
        # q[g] ← q[g] * exp(η * loss_g)
        self.q *= torch.exp(self.eta * group_losses.detach())
        self.q /= self.q.sum()  # normalize

        # Final weighted loss
        loss = torch.dot(self.q, group_losses)

        metrics = {
            "groupdro_loss": loss.item(),
            "loss": loss.item(),
        }

        return loss, metrics


class VRExLoss(BaseLoss):
    def __init__(self, lambda_: float = 1.0, anneal_iters: int = 500):
        super().__init__()
        self.initial_lambda = lambda_
        self.lambda_ = 1.0
        self.anneal_iters = anneal_iters
        self.step = 0

    def forward(self, output, target) -> Tuple[torch.Tensor, Dict[str, float]]:
        domains, labels = target
        unique_domains = torch.unique(domains)
        
        per_env_losses = []
        metrics = {}

        # Compute cross-entropy per environment
        for d in unique_domains:
            idx = domains == d
            if idx.sum() > 0:
                env_loss = F.cross_entropy(output[idx], labels[idx])
                per_env_losses.append(env_loss)
                metrics[f"domain_{d.item()}_loss"] = env_loss.item()

        losses_tensor = torch.stack(per_env_losses)
        mean_loss = losses_tensor.mean()
        penalty = ((losses_tensor - mean_loss) ** 2).mean()

        # Annealing
        if self.step >= self.anneal_iters:
            penalty_weight = self.initial_lambda
        else:
            penalty_weight = 1.0

        self.step += 1
        total_loss = mean_loss + penalty_weight * penalty

        metrics.update({
            "vrex_mean_loss": mean_loss.item(),
            "vrex_penalty": penalty.item(),
            "vrex_weight": penalty_weight,
            "loss": total_loss.item()
        })

        return total_loss, metrics



class MMDLoss(BaseLoss):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def _my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min(1e-30)

    def _gaussian_kernel(self, x, y, gammas=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self._my_cdist(x, y)
        K = torch.zeros_like(D)
        for g in gammas:
            K += torch.exp(-g * D)
        return K

    def _mmd(self, x, y):
        Kxx = self._gaussian_kernel(x, x).mean()
        Kyy = self._gaussian_kernel(y, y).mean()
        Kxy = self._gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def forward(self, output, target) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Expects `output = (features, logits)`"""
        logits, features = output
        domains, labels = target
        unique_domains = torch.unique(domains)

        ce_loss = 0.0
        penalty = 0.0
        domain_features = []
        domain_labels = []

        for d in unique_domains:
            idx = domains == d
            if idx.sum() < 2:
                continue
            domain_features.append(features[idx])
            domain_labels.append((logits[idx], labels[idx]))

        n_envs = len(domain_features)

        for logit, label in domain_labels:
            ce_loss += F.cross_entropy(logit, label)

        if n_envs >= 2:
            pair_count = 0
            for i in range(n_envs):
                for j in range(i + 1, n_envs):
                    penalty += self._mmd(domain_features[i], domain_features[j])
                    pair_count += 1
            penalty /= pair_count
        else:
            penalty = torch.tensor(0.0, device=features.device)

        ce_loss /= n_envs
        total_loss = ce_loss + self.gamma * penalty

        return total_loss, {
            "ce_loss": ce_loss.item(),
            "mmd_penalty": penalty.item(),
            "loss": total_loss.item()
        }


class CORALLoss(BaseLoss):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def _coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = cent_x.t() @ cent_x / (len(x) - 1)
        cova_y = cent_y.t() @ cent_y / (len(y) - 1)
        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()
        return mean_diff + cova_diff

    def forward(self, output, target) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits, features = output
        domains, labels = target
        unique_domains = torch.unique(domains)

        ce_loss = 0.0
        penalty = 0.0
        domain_features = []
        domain_labels = []

        for d in unique_domains:
            idx = domains == d
            if idx.sum() < 2:
                continue
            domain_features.append(features[idx])
            domain_labels.append((logits[idx], labels[idx]))

        n_envs = len(domain_features)

        for logit, label in domain_labels:
            ce_loss += F.cross_entropy(logit, label)

        if n_envs >= 2:
            pair_count = 0
            for i in range(n_envs):
                for j in range(i + 1, n_envs):
                    penalty += self._coral(domain_features[i], domain_features[j])
                    pair_count += 1
            penalty /= pair_count
        else:
            penalty = torch.tensor(0.0, device=features.device)

        ce_loss /= n_envs
        total_loss = ce_loss + self.gamma * penalty

        return total_loss, {
            "ce_loss": ce_loss.item(),
            "coral_penalty": penalty.item(),
            "loss": total_loss.item()
        }

class DANNLoss(BaseLoss):
    def __init__(self,
                 domains: list,
                 num_classes: int,
                 feature_dim: int = 128,
                 lambda_: float = 1.0,
                 grad_penalty: float = 0.0,
                 lr_d: float = 5e-5,
                 weight_decay_d: float = 0.0,
                 beta1: float = 0.5,
                 conditional: bool = False,
                 class_balance: bool = False,
                 d_steps_per_g_step: int = 1):
        super().__init__()
        self.step = 0

        self.num_classes = num_classes
        self.num_domains = len(domains)

        # Set domain lookup
        # Create mapping tensor: original_id → new_id
        max_domain_id = max(domains)
        if max_domain_id >= len(domains):
            self._map_domains = True 
            self._domain_lookup = torch.full((max_domain_id + 1,), -1, dtype=torch.long, device='cuda') # << automize
            for new_id, orig_id in enumerate(domains):
                self._domain_lookup[orig_id] = new_id
        else:
            self._map_domains = False


        self.lambda_ = lambda_
        self.grad_penalty = grad_penalty
        self.conditional = conditional
        self.class_balance = class_balance
        self.d_steps_per_g_step = d_steps_per_g_step

        # Discriminator is just one linear layer, as the main branch
        self.discriminator = nn.Linear(feature_dim, self.num_domains)
        self.class_embeddings = (
            nn.Embedding(num_classes, feature_dim).cuda()
            if conditional else None
        )

        self.discriminator = self.discriminator.cuda()
        self.class_embeddings = self.class_embeddings.cuda() if self.class_embeddings else None

        self.disc_opt = torch.optim.Adam(
            list(self.discriminator.parameters()) +
            (list(self.class_embeddings.parameters()) if conditional else []),
            lr=lr_d,
            weight_decay=weight_decay_d,
            betas=(beta1, 0.9)
        )

    def forward(self,
                output: Tuple[torch.Tensor, torch.Tensor],
                target: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:

        logits, features = output
        domains, labels = target
        device = features.device

        #print(torch.unique(labels), self.num_classes)
        #print(torch.unique(domains), self.num_domains)

        if self._map_domains:
            domains = self._domain_lookup[domains]

        self.is_disc_step = (self.step % (1 + self.d_steps_per_g_step)) < self.d_steps_per_g_step
        self.step += 1

        # === Classifier loss ===
        clf_loss = F.cross_entropy(logits, labels)

        # === Discriminator input ===
        disc_input = features
        if self.conditional and self.class_embeddings is not None:
            disc_input = disc_input + self.class_embeddings(labels)

        # === Discriminator labels ===
        disc_labels = domains

        # === Discriminator output ===
        disc_logits = self.discriminator(disc_input)

        # === Discriminator loss ===
        if self.class_balance:
            y_counts = F.one_hot(labels, num_classes=self.class_embeddings.num_embeddings).sum(dim=0)
            weights = 1. / (y_counts[labels] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_logits, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_logits, disc_labels)

        # === Gradient penalty ===
        if self.grad_penalty > 0.:
            input_grad = autograd.grad(
                disc_loss, [disc_input], create_graph=True, retain_graph=True
            )[0]
            grad_penalty = (input_grad**2).sum(dim=1).mean()
            disc_loss += self.grad_penalty * grad_penalty
        else:
            grad_penalty = torch.tensor(0.0, device=device)

        # === Adversarial update strategy ===
        if self.is_disc_step:
            self.disc_opt.zero_grad() # << move outside?
            disc_loss.backward() # # < < move outside?
            self.disc_opt.step() # << move outside?
            return disc_loss.detach(), {
                "disc_loss": disc_loss.item(),
                "grad_penalty": grad_penalty.item(),
                #"step_type": "disc"
            }
        else:
            total_loss = clf_loss - self.lambda_ * disc_loss
            return total_loss, {
                "clf_loss": clf_loss.item(),
                "adv_disc_loss": disc_loss.item(),
                "lambda": self.lambda_,
                "loss": total_loss.item(),
                #"step_type": "gen"
            }
