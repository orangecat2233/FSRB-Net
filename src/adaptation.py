

import torch
import torch.nn as nn

class MMDLoss(nn.Module):

    def __init__(
        self,
        bandwidth: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.bandwidth = bandwidth
        self.eps = eps

    @staticmethod
    def _flatten_features(
        features: torch.Tensor,
    ) -> torch.Tensor:

        if not torch.is_tensor(features):
            raise TypeError(
                "MMD  torch.Tensor,"
                f":{type(features).__name__}"
            )

        if features.ndim < 2:
            raise RuntimeError(
                "MMD ,"
                f":{tuple(features.shape)}"
            )

        if features.ndim > 2:
            features = torch.flatten(
                features,
                start_dim=1,
            )

        return features

    @staticmethod
    def _pairwise_squared_distance(
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:

        x_norm = torch.sum(
            x * x,
            dim=1,
            keepdim=True,
        )

        y_norm = torch.sum(
            y * y,
            dim=1,
            keepdim=True,
        ).transpose(0, 1)

        distance = (
            x_norm
            + y_norm
            - 2.0 * torch.matmul(
                x,
                y.transpose(0, 1),
            )
        )

                      
        distance = torch.clamp(
            distance,
            min=0.0,
        )

        return distance

    def _estimate_bandwidth(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:

        if self.bandwidth is not None:

            bandwidth = torch.as_tensor(
                self.bandwidth,
                dtype=source_features.dtype,
                device=source_features.device,
            )

            return torch.clamp(
                bandwidth,
                min=self.eps,
            )

                                                              
                         
                                                              

        combined_features = torch.cat(
            [
                source_features,
                target_features,
            ],
            dim=0,
        )

        distance_matrix = (
            self._pairwise_squared_distance(
                combined_features,
                combined_features,
            )
        )

        sample_count = (
            combined_features.size(0)
        )

                                                              
                   
                                                              

        if sample_count <= 1:
            raise RuntimeError(
                "MMD "
                " kernel bandwidth."
            )

        mask = ~torch.eye(
            sample_count,
            dtype=torch.bool,
            device=distance_matrix.device,
        )

        off_diagonal_distances = (
            distance_matrix[mask]
        )

                           
        bandwidth = (
            off_diagonal_distances
            .detach()
            .mean()
        )

        bandwidth = torch.clamp(
            bandwidth,
            min=self.eps,
        )

        return bandwidth

    def _rbf_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        bandwidth: torch.Tensor,
    ) -> torch.Tensor:

        distance = (
            self._pairwise_squared_distance(
                x,
                y,
            )
        )

        kernel = torch.exp(
            -distance
            / (
                2.0
                * bandwidth
                + self.eps
            )
        )

        return kernel

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:

                                                              
                 
                                                              

        source_features = (
            self._flatten_features(
                source_features
            )
        )

        target_features = (
            self._flatten_features(
                target_features
            )
        )

                                                              
                 
                                                              

        if (
            source_features.size(1)
            != target_features.size(1)
        ):
            raise RuntimeError(
                ":"
                f"source={source_features.size(1)},"
                f"target={target_features.size(1)}"
            )

        source_count = (
            source_features.size(0)
        )

        target_count = (
            target_features.size(0)
        )

        if source_count < 2:
            raise RuntimeError(
                "MMD  batch size >= 2,"
                f":{source_count}"
            )

        if target_count < 2:
            raise RuntimeError(
                "MMD  batch size >= 2,"
                f":{target_count}"
            )

                                                              
                      
                                                              

        bandwidth = (
            self._estimate_bandwidth(
                source_features,
                target_features,
            )
        )

                                                              
                            
                                                              

        kernel_ss = self._rbf_kernel(
            source_features,
            source_features,
            bandwidth,
        )

        kernel_tt = self._rbf_kernel(
            target_features,
            target_features,
            bandwidth,
        )

        kernel_st = self._rbf_kernel(
            source_features,
            target_features,
            bandwidth,
        )

                                                              
                                   
         
                           
                                                              

        source_term = (
            (
                kernel_ss.sum()
                - torch.diagonal(
                    kernel_ss
                ).sum()
            )
            / (
                source_count
                * (source_count - 1)
            )
        )

        target_term = (
            (
                kernel_tt.sum()
                - torch.diagonal(
                    kernel_tt
                ).sum()
            )
            / (
                target_count
                * (target_count - 1)
            )
        )

        cross_term = (
            kernel_st.mean()
        )

        mmd_loss = (
            source_term
            + target_term
            - 2.0 * cross_term
        )

        return mmd_loss

"""CORAL loss for unsupervised domain adaptation."""

import torch
import torch.nn as nn

class CORALLoss(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _flatten_features(features: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(features):
            raise TypeError(
                "CORAL input features must be torch.Tensor, "
                f"but got {type(features).__name__}."
            )

        if features.ndim < 2:
            raise RuntimeError(
                "CORAL input features must have at least 2 dimensions, "
                f"but got shape {tuple(features.shape)}."
            )

        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)

        return features

    @staticmethod
    def _covariance(features: torch.Tensor) -> torch.Tensor:
        sample_count = features.size(0)

        if sample_count < 2:
            raise RuntimeError(
                "CORAL requires batch_size >= 2, "
                f"but got {sample_count}."
            )

        centered = features - features.mean(dim=0, keepdim=True)
        covariance = (
            centered.transpose(0, 1) @ centered
        ) / (sample_count - 1)

        return covariance

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        source_features = self._flatten_features(source_features)
        target_features = self._flatten_features(target_features)

        if source_features.size(1) != target_features.size(1):
            raise RuntimeError(
                "Source and target feature dimensions do not match: "
                f"source={source_features.size(1)}, "
                f"target={target_features.size(1)}."
            )

        source_covariance = self._covariance(source_features)
        target_covariance = self._covariance(target_features)

        feature_dim = source_features.size(1)
        covariance_difference = source_covariance - target_covariance

        coral_loss = covariance_difference.pow(2).sum() / (
            4.0 * feature_dim * feature_dim
        )

        return coral_loss

"""Multi-Kernel Maximum Mean Discrepancy (MK-MMD) loss."""

from typing import Iterable, Optional

import torch
import torch.nn as nn

class MKMMDLoss(nn.Module):

    def __init__(
        self,
        kernel_multipliers: Iterable[float] = (
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
        ),
        bandwidth: Optional[float] = None,
        eps: float = 1e-8,
    ):
        super().__init__()

        kernel_multipliers = tuple(
            float(value) for value in kernel_multipliers
        )

        if len(kernel_multipliers) == 0:
            raise ValueError("kernel_multipliers cannot be empty.")

        if any(value <= 0 for value in kernel_multipliers):
            raise ValueError(
                "All kernel multipliers must be > 0."
            )

        self.kernel_multipliers = kernel_multipliers
        self.bandwidth = bandwidth
        self.eps = eps

    @staticmethod
    def _flatten_features(
        features: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(features):
            raise TypeError(
                "MK-MMD input features must be torch.Tensor, "
                f"but got {type(features).__name__}."
            )

        if features.ndim < 2:
            raise RuntimeError(
                "MK-MMD input features must have at least 2 dimensions, "
                f"but got shape {tuple(features.shape)}."
            )

        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)

        return features

    @staticmethod
    def _pairwise_squared_distance(
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = torch.sum(x * x, dim=1, keepdim=True)
        y_norm = torch.sum(
            y * y, dim=1, keepdim=True
        ).transpose(0, 1)

        distance = (
            x_norm
            + y_norm
            - 2.0 * torch.matmul(x, y.transpose(0, 1))
        )

        return torch.clamp(distance, min=0.0)

    def _estimate_base_bandwidth(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.bandwidth is not None:
            bandwidth = torch.as_tensor(
                self.bandwidth,
                dtype=source_features.dtype,
                device=source_features.device,
            )
            return torch.clamp(
                bandwidth,
                min=self.eps,
            )

        combined_features = torch.cat(
            [source_features, target_features],
            dim=0,
        )

        distance_matrix = self._pairwise_squared_distance(
            combined_features,
            combined_features,
        )

        sample_count = combined_features.size(0)

        if sample_count <= 1:
            raise RuntimeError(
                "MK-MMD requires at least two joint samples "
                "to estimate kernel bandwidth."
            )

        off_diagonal_mask = ~torch.eye(
            sample_count,
            dtype=torch.bool,
            device=distance_matrix.device,
        )

        off_diagonal_distances = distance_matrix[
            off_diagonal_mask
        ]

        base_bandwidth = (
            off_diagonal_distances.detach().mean()
        )

        return torch.clamp(
            base_bandwidth,
            min=self.eps,
        )

    def _multi_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        base_bandwidth: torch.Tensor,
    ) -> torch.Tensor:
        distance = self._pairwise_squared_distance(x, y)

        kernels = []

        for multiplier in self.kernel_multipliers:
            current_bandwidth = (
                base_bandwidth * multiplier
            )

            current_kernel = torch.exp(
                -distance
                / (
                    2.0 * current_bandwidth
                    + self.eps
                )
            )

            kernels.append(current_kernel)

        return torch.stack(kernels, dim=0).mean(dim=0)

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        source_features = self._flatten_features(
            source_features
        )
        target_features = self._flatten_features(
            target_features
        )

        if source_features.size(1) != target_features.size(1):
            raise RuntimeError(
                "Source and target feature dimensions do not match: "
                f"source={source_features.size(1)}, "
                f"target={target_features.size(1)}."
            )

        source_count = source_features.size(0)
        target_count = target_features.size(0)

        if source_count < 2:
            raise RuntimeError(
                "MK-MMD requires source batch_size >= 2, "
                f"but got {source_count}."
            )

        if target_count < 2:
            raise RuntimeError(
                "MK-MMD requires target batch_size >= 2, "
                f"but got {target_count}."
            )

        base_bandwidth = self._estimate_base_bandwidth(
            source_features,
            target_features,
        )

        kernel_ss = self._multi_kernel(
            source_features,
            source_features,
            base_bandwidth,
        )
        kernel_tt = self._multi_kernel(
            target_features,
            target_features,
            base_bandwidth,
        )
        kernel_st = self._multi_kernel(
            source_features,
            target_features,
            base_bandwidth,
        )

        source_term = (
            kernel_ss.sum()
            - torch.diagonal(kernel_ss).sum()
        ) / (
            source_count * (source_count - 1)
        )

        target_term = (
            kernel_tt.sum()
            - torch.diagonal(kernel_tt).sum()
        ) / (
            target_count * (target_count - 1)
        )

        cross_term = kernel_st.mean()

        return (
            source_term
            + target_term
            - 2.0 * cross_term
        )

"""Stable DWJDA objective for DWDAN (Li et al., EAAI 2026).

The paper describes two central ideas in DWJDA:
1) class-aware weighting to reduce the influence of random class-count imbalance
   inside a mini-batch during conditional alignment;
2) A-distance based dynamic weighting between marginal and conditional alignment.

This implementation preserves those mechanisms in a controlled reimplementation.
For numerical stability under the project's 10-class / batch-size-40 MR-STFT-RGB
protocol, target pseudo-labels are confidence-filtered and the final adaptation
strength is scheduled by the trainer rather than applied at full strength from
batch 1.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8

def _pairwise_sq_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    dist = x2 + y2 - 2.0 * (x @ y.T)
    return dist.clamp_min(0.0)

def _multi_kernel_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> torch.Tensor:
    dist = _pairwise_sq_distance(x, y)

    with torch.no_grad():
        positive = dist[dist > 0]
        if positive.numel() > 0:
            bandwidth = positive.median()
        else:
            bandwidth = dist.new_tensor(1.0)
        bandwidth = bandwidth.clamp_min(1e-4)

    center = kernel_num // 2
    kernels = []
    for i in range(kernel_num):
        bw = bandwidth * (kernel_mul ** (i - center))
        kernels.append(torch.exp(-dist / (2.0 * bw + _EPS)))

    return sum(kernels) / float(kernel_num)

def _weighted_mmd(
    source: torch.Tensor,
    target: torch.Tensor,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> torch.Tensor:
    if source.numel() == 0 or target.numel() == 0:
        return source.new_zeros(())

    source = F.normalize(source, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)

    ns = source.size(0)
    nt = target.size(0)

    if source_weights is None:
        source_weights = source.new_ones(ns)
    if target_weights is None:
        target_weights = target.new_ones(nt)

    source_weights = source_weights.clamp_min(0.0)
    target_weights = target_weights.clamp_min(0.0)

    source_weights = source_weights / (source_weights.sum() + _EPS)
    target_weights = target_weights / (target_weights.sum() + _EPS)

    k_ss = _multi_kernel_matrix(source, source, kernel_mul, kernel_num)
    k_tt = _multi_kernel_matrix(target, target, kernel_mul, kernel_num)
    k_st = _multi_kernel_matrix(source, target, kernel_mul, kernel_num)

    term_ss = source_weights @ k_ss @ source_weights
    term_tt = target_weights @ k_tt @ target_weights
    term_st = source_weights @ k_st @ target_weights

    return (term_ss + term_tt - 2.0 * term_st).clamp_min(0.0)

@torch.no_grad()
def _proxy_a_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    if source.size(0) < 2 or target.size(0) < 2:
        return source.new_zeros(())

    source = F.normalize(source.detach(), p=2, dim=1)
    target = F.normalize(target.detach(), p=2, dim=1)

    cs = source.mean(dim=0, keepdim=True)
    ct = target.mean(dim=0, keepdim=True)

    s_ds = ((source - cs) ** 2).sum(dim=1)
    s_dt = ((source - ct) ** 2).sum(dim=1)
    t_ds = ((target - cs) ** 2).sum(dim=1)
    t_dt = ((target - ct) ** 2).sum(dim=1)

    correct = (s_ds <= s_dt).float().sum() + (t_dt < t_ds).float().sum()
    accuracy = correct / float(source.size(0) + target.size(0))

                                                                              
    accuracy = torch.maximum(accuracy, 1.0 - accuracy)
    error = (1.0 - accuracy).clamp(0.0, 0.5)
    return (2.0 * (1.0 - 2.0 * error)).clamp(0.0, 2.0)

class DWDANLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        confidence_threshold: float = 0.60,
        min_class_samples: int = 2,
        min_dynamic_weight: float = 0.10,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.kernel_mul = float(kernel_mul)
        self.kernel_num = int(kernel_num)
        self.confidence_threshold = float(confidence_threshold)
        self.min_class_samples = int(min_class_samples)
        self.min_dynamic_weight = float(min_dynamic_weight)

    def _conditional_alignment(
        self,
        source_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_features: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        target_prob = F.softmax(target_logits.detach(), dim=1)
        target_conf, target_pseudo = target_prob.max(dim=1)

        class_losses = []
        class_raw_weights = []
        class_a_distances = []

        for class_index in range(self.num_classes):
            source_mask = source_labels == class_index
            target_mask = (
                (target_pseudo == class_index)
                & (target_conf >= self.confidence_threshold)
            )

            ns = int(source_mask.sum().item())
            nt = int(target_mask.sum().item())

            if ns < self.min_class_samples or nt < self.min_class_samples:
                continue

            source_c = source_features[source_mask]
            target_c = target_features[target_mask]

                                                                               
            source_w = source_c.new_ones(ns)
            target_w = target_conf[target_mask].to(target_c.dtype)

            class_loss = _weighted_mmd(
                source_c,
                target_c,
                source_weights=source_w,
                target_weights=target_w,
                kernel_mul=self.kernel_mul,
                kernel_num=self.kernel_num,
            )

                                                                              
                                                                              
            class_weight = source_c.new_tensor(
                1.0 / float(ns + nt)
            )

            class_losses.append(class_loss)
            class_raw_weights.append(class_weight)
            class_a_distances.append(_proxy_a_distance(source_c, target_c))

        if len(class_losses) == 0:
            zero = source_features.new_zeros(())
            return zero, zero, zero

        losses = torch.stack(class_losses)
        weights = torch.stack(class_raw_weights)
        weights = weights / (weights.sum() + _EPS)

        conditional_loss = (weights * losses).sum()
        conditional_a = torch.stack(class_a_distances).mean()
        valid_class_count = source_features.new_tensor(float(len(class_losses)))

        return conditional_loss, conditional_a, valid_class_count

    def forward(
        self,
        source_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_features: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        marginal_loss = _weighted_mmd(
            source_features,
            target_features,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
        )

        conditional_loss, conditional_a, valid_class_count = (
            self._conditional_alignment(
                source_features=source_features,
                source_labels=source_labels,
                target_features=target_features,
                target_logits=target_logits,
            )
        )

        marginal_a = _proxy_a_distance(source_features, target_features)

        with torch.no_grad():
            if valid_class_count.item() <= 0:
                conditional_weight = source_features.new_tensor(0.0)
            else:
                conditional_weight = conditional_a / (
                    marginal_a + conditional_a + _EPS
                )

            lower = self.min_dynamic_weight
            upper = 1.0 - self.min_dynamic_weight

            if valid_class_count.item() > 0:
                conditional_weight = conditional_weight.clamp(lower, upper)
            else:
                conditional_weight = conditional_weight.clamp(0.0, upper)

            marginal_weight = 1.0 - conditional_weight

        joint_loss = (
            marginal_weight * marginal_loss
            + conditional_weight * conditional_loss
        )

        return {
            "joint_loss": joint_loss,
            "marginal_loss": marginal_loss,
            "conditional_loss": conditional_loss,
            "marginal_weight": marginal_weight,
            "conditional_weight": conditional_weight,
            "marginal_a_distance": marginal_a,
            "conditional_a_distance": conditional_a,
            "valid_class_count": valid_class_count,
        }

"""Domain discriminator and gradient reversal layer for DANN."""

import torch
import torch.nn as nn
from torch.autograd import Function

class _GradientReversalFunction(Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float):
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            -ctx.coeff * grad_output,
            None,
        )

def gradient_reverse(
    features: torch.Tensor,
    coeff: float = 1.0,
) -> torch.Tensor:

    return _GradientReversalFunction.apply(
        features,
        coeff,
    )

class DomainDiscriminator(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be > 0."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(
                hidden_dim,
                2,
            ),
        )

    @staticmethod
    def _flatten_features(
        features: torch.Tensor,
    ) -> torch.Tensor:

        if not torch.is_tensor(features):
            raise TypeError(
                "DomainDiscriminator input must be torch.Tensor, "
                f"but got {type(features).__name__}."
            )

        if features.ndim < 2:
            raise RuntimeError(
                "DomainDiscriminator input must have at least "
                "2 dimensions, "
                f"but got shape {tuple(features.shape)}."
            )

        if features.ndim > 2:
            features = torch.flatten(
                features,
                start_dim=1,
            )

        return features

    def forward(
        self,
        features: torch.Tensor,
        grl_coeff: float = 1.0,
    ) -> torch.Tensor:

        features = self._flatten_features(
            features
        )

        reversed_features = gradient_reverse(
            features,
            coeff=grl_coeff,
        )

        domain_logits = self.classifier(
            reversed_features
        )

        return domain_logits