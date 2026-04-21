"""
2-layer MLP probe with Hewitt & Liang control task validation
==============================================================
Given residual-stream activations h_l in R^d at layer l and target first-token
labels y in {1..V}, train a 2-layer MLP

    logits = W_2 @ GELU(W_1 @ LN(h)) + b_2     (~1024 hidden units)

with cross-entropy loss. Selectivity = real_top10 - control_top10, where the
control probe is trained on labels deterministically shuffled per unique input
(Hewitt & Liang 2019). High selectivity means the probe found genuine input ->
output structure rather than memorizing.

Train/test split is by unique input (not by example), so the probe must
generalize across inputs it never saw, matching Hewitt & Liang's protocol.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MLPDecoderConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLPProbe module
# ---------------------------------------------------------------------------
class MLPProbe(nn.Module):
    """
    2-layer MLP: residual-stream -> hidden (GELU) -> vocab logits.

    Parameters of note:
      - input_layernorm: applies LayerNorm to the input first (stabilizes
        training given residual streams have very different magnitudes
        across layers/models).
      - dropout between hidden and output.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        vocab_size: int,
        dropout: float = 0.1,
        input_layernorm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.in_norm = nn.LayerNorm(d_model) if input_layernorm else nn.Identity()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # Standard small-init for the output layer (vocab is large, default
        # init produces large logits and slow learning).
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(h)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Probe training results
# ---------------------------------------------------------------------------
@dataclass
class ProbeResult:
    """Result of training one probe (real or control) at one layer."""
    task: str
    layer_1idx: int
    condition: str            # "real" or "control"
    n_train: int
    n_test: int
    train_top1: float
    train_top10: float
    test_top1: float
    test_top5: float
    test_top10: float
    final_train_loss: float
    final_test_loss: float
    n_epochs: int


@dataclass
class LayerComparison:
    """Real vs control results at one (task, layer) -- what we report."""
    task: str
    layer_1idx: int
    real_test_top1: float
    real_test_top5: float
    real_test_top10: float
    control_test_top1: float
    control_test_top5: float
    control_test_top10: float
    selectivity_top1: float
    selectivity_top5: float
    selectivity_top10: float
    n_test: int


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def _stack_task_data(
    task_cache: Dict,
    layer_1idx: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Pool activations across the 8 templates of one task at one layer.

    Returns:
      X: [N_total, d_model] activations on `device` (fp32 for training)
      y: [N_total] long token-id labels on `device`
      inputs: list of length N_total -- the input string for each example
    """
    Xs, ys, inputs_all = [], [], []
    for tid, payload in task_cache.items():
        acts_by_layer = payload["acts_by_layer"]
        if layer_1idx not in acts_by_layer:
            continue
        X = acts_by_layer[layer_1idx]  # [N, d_model] (bf16 cpu)
        y = payload["correct_first_tokens"]
        inputs = payload["inputs"]

        # Drop examples with -1 labels (tokenization failures)
        valid = (y >= 0)
        if not valid.all():
            X = X[valid]
            y = y[valid]
            inputs = [inp for inp, ok in zip(inputs, valid.tolist()) if ok]

        Xs.append(X)
        ys.append(y)
        inputs_all.extend(inputs)

    X = torch.cat(Xs, dim=0).to(device=device, dtype=torch.float32, non_blocking=True)
    y = torch.cat(ys, dim=0).to(device=device, dtype=torch.long, non_blocking=True)
    return X, y, inputs_all


def _make_input_split(
    inputs: List[str],
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split-by-unique-input. All occurrences of the same input go to the same
    side of the split, so the probe must generalize to new inputs.

    Returns (train_idx, test_idx) into the example list.
    """
    unique_inputs = sorted(set(inputs))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_inputs)
    n_test = max(1, int(round(len(unique_inputs) * test_fraction)))
    test_inputs = set(unique_inputs[:n_test])

    inputs_arr = np.asarray(inputs)
    test_mask = np.array([inp in test_inputs for inp in inputs])
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx


def _make_control_labels(
    inputs: List[str],
    real_labels: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """
    Hewitt & Liang control task: deterministically shuffle the input -> label
    mapping. All occurrences of the same input get the same shuffled label,
    but the mapping is a random permutation of the real label set.
    """
    real_np = real_labels.cpu().numpy()
    unique_inputs = sorted(set(inputs))

    # Build the real input -> label mapping (use first occurrence)
    real_map: Dict[str, int] = {}
    for inp, lbl in zip(inputs, real_np.tolist()):
        if inp not in real_map:
            real_map[inp] = lbl

    # Shuffle the values across keys
    rng = np.random.RandomState(seed)
    keys = list(real_map.keys())
    values = [real_map[k] for k in keys]
    perm = rng.permutation(len(values))
    shuffled_map = {k: values[perm[i]] for i, k in enumerate(keys)}

    control_labels = np.array([shuffled_map[inp] for inp in inputs], dtype=np.int64)
    return torch.tensor(control_labels, dtype=torch.long, device=real_labels.device)


# ---------------------------------------------------------------------------
# Single-probe training routine
# ---------------------------------------------------------------------------
def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Top-k accuracy via topk indices."""
    if labels.numel() == 0:
        return 0.0
    topk = logits.topk(k, dim=-1).indices  # [N, k]
    hits = (topk == labels.unsqueeze(-1)).any(dim=-1).float()
    return hits.mean().item()


def _train_one_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    cfg: MLPDecoderConfig,
    vocab_size: int,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Train one MLPProbe on (X_train, y_train), evaluate on (X_test, y_test).

    Returns: (train_top1, train_top10, test_top1, test_top5, test_top10,
              final_train_loss, final_test_loss, n_epochs_used)
    """
    d_model = X_train.shape[-1]
    probe = MLPProbe(
        d_model=d_model,
        hidden_dim=cfg.hidden_dim,
        vocab_size=vocab_size,
        dropout=cfg.dropout,
        input_layernorm=cfg.input_layernorm,
    ).to(device)

    optim = torch.optim.AdamW(
        probe.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fused=device.type == "cuda",
    )

    n_steps = cfg.n_epochs if cfg.full_batch else (
        cfg.n_epochs * max(1, math.ceil(X_train.shape[0] / cfg.batch_size))
    )
    warmup_steps = max(1, int(n_steps * cfg.warmup_fraction))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    if cfg.full_batch:
        # One step per epoch, full-batch.
        for _ in range(cfg.n_epochs):
            probe.train()
            optim.zero_grad(set_to_none=True)
            logits = probe(X_train)
            loss = F.cross_entropy(logits, y_train)
            loss.backward()
            optim.step()
            scheduler.step()
    else:
        n = X_train.shape[0]
        idx = torch.randperm(n, device=device)
        for ep in range(cfg.n_epochs):
            probe.train()
            for i in range(0, n, cfg.batch_size):
                sl = idx[i:i + cfg.batch_size]
                logits = probe(X_train[sl])
                loss = F.cross_entropy(logits, y_train[sl])
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                scheduler.step()
            # reshuffle each epoch
            idx = torch.randperm(n, device=device)

    # Evaluation
    probe.eval()
    with torch.no_grad():
        train_logits = probe(X_train)
        train_loss = F.cross_entropy(train_logits, y_train).item()
        train_top1 = _topk_accuracy(train_logits, y_train, 1)
        train_top10 = _topk_accuracy(train_logits, y_train, 10)

        test_logits = probe(X_test)
        test_loss = F.cross_entropy(test_logits, y_test).item()
        test_top1 = _topk_accuracy(test_logits, y_test, 1)
        test_top5 = _topk_accuracy(test_logits, y_test, 5)
        test_top10 = _topk_accuracy(test_logits, y_test, 10)

    # Free probe memory
    del probe, optim, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (train_top1, train_top10, test_top1, test_top5, test_top10,
            train_loss, test_loss, float(cfg.n_epochs))


# ---------------------------------------------------------------------------
# Public entry: train both probes for one (task, layer)
# ---------------------------------------------------------------------------
def train_probes_for_task_layer(
    task_name: str,
    layer_1idx: int,
    task_cache: Dict,
    cfg: MLPDecoderConfig,
    vocab_size: int,
    device: torch.device,
) -> Tuple[ProbeResult, ProbeResult, LayerComparison]:
    """
    Train one real-label MLP and one control-label MLP at the given layer.
    Returns (real_result, control_result, layer_comparison).
    """
    # Stack data across templates
    X, y_real, inputs = _stack_task_data(task_cache, layer_1idx, device)

    # Train/test split by unique input
    train_idx, test_idx = _make_input_split(
        inputs, cfg.test_input_fraction, cfg.seed,
    )
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long, device=device)

    X_train = X[train_idx_t]
    X_test = X[test_idx_t]
    y_real_train = y_real[train_idx_t]
    y_real_test = y_real[test_idx_t]

    # ---- real probe ----
    (rtrn_top1, rtrn_top10, rtest_top1, rtest_top5, rtest_top10,
     rtrn_loss, rtest_loss, n_eps) = _train_one_probe(
        X_train, y_real_train, X_test, y_real_test,
        cfg=cfg, vocab_size=vocab_size, device=device,
    )
    real_result = ProbeResult(
        task=task_name, layer_1idx=layer_1idx, condition="real",
        n_train=int(X_train.shape[0]), n_test=int(X_test.shape[0]),
        train_top1=rtrn_top1, train_top10=rtrn_top10,
        test_top1=rtest_top1, test_top5=rtest_top5, test_top10=rtest_top10,
        final_train_loss=rtrn_loss, final_test_loss=rtest_loss,
        n_epochs=int(n_eps),
    )

    # ---- control probe (Hewitt & Liang shuffle) ----
    if cfg.run_control_probes:
        y_control = _make_control_labels(inputs, y_real, cfg.control_seed)
        y_ctrl_train = y_control[train_idx_t]
        y_ctrl_test = y_control[test_idx_t]
        (ctrn_top1, ctrn_top10, ctest_top1, ctest_top5, ctest_top10,
         ctrn_loss, ctest_loss, _) = _train_one_probe(
            X_train, y_ctrl_train, X_test, y_ctrl_test,
            cfg=cfg, vocab_size=vocab_size, device=device,
        )
        control_result = ProbeResult(
            task=task_name, layer_1idx=layer_1idx, condition="control",
            n_train=int(X_train.shape[0]), n_test=int(X_test.shape[0]),
            train_top1=ctrn_top1, train_top10=ctrn_top10,
            test_top1=ctest_top1, test_top5=ctest_top5, test_top10=ctest_top10,
            final_train_loss=ctrn_loss, final_test_loss=ctest_loss,
            n_epochs=int(n_eps),
        )
    else:
        control_result = ProbeResult(
            task=task_name, layer_1idx=layer_1idx, condition="control",
            n_train=int(X_train.shape[0]), n_test=int(X_test.shape[0]),
            train_top1=0.0, train_top10=0.0,
            test_top1=0.0, test_top5=0.0, test_top10=0.0,
            final_train_loss=float("nan"), final_test_loss=float("nan"),
            n_epochs=0,
        )

    comparison = LayerComparison(
        task=task_name,
        layer_1idx=layer_1idx,
        real_test_top1=real_result.test_top1,
        real_test_top5=real_result.test_top5,
        real_test_top10=real_result.test_top10,
        control_test_top1=control_result.test_top1,
        control_test_top5=control_result.test_top5,
        control_test_top10=control_result.test_top10,
        selectivity_top1=real_result.test_top1 - control_result.test_top1,
        selectivity_top5=real_result.test_top5 - control_result.test_top5,
        selectivity_top10=real_result.test_top10 - control_result.test_top10,
        n_test=real_result.n_test,
    )
    return real_result, control_result, comparison
