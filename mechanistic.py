"""
Mechanistic Interpretability Experiments (per EXPERIMENT_REDESIGN_SPEC.md §5.6)
================================================================================
All experiments here are **IID-gated**: they check IID accuracy before
running causal interventions.  Tasks below threshold are logged and skipped
for patching but still run through probing (to distinguish "not linearly
readable" from "linearly readable but not steerable").

Experiments:
  1. Linear Probing -- per-(task, template, layer) logistic regression probes
     on cached zero-shot activations, with C sweep on validation set,
     cross-template probe transfer, multi-label evaluation for ambiguous tasks,
     and readability-steerability gap computation.
  2. Activation Patching -- layer-wise causal patching (IID-gated)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wilcoxon
from tqdm import tqdm

from .config import ExperimentConfig
from .extraction import FVCollection
from .models import ModelWrapper
from .steering import IIDSummary
from .tasks import TaskSpec, get_tasks, TaskCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Linear Probing (spec Section 5.6)
# ---------------------------------------------------------------------------
@dataclass
class ProbeResult:
    """Result of linear probing at one (task, template, layer)."""
    task: str
    template_id: str
    layer: int
    accuracy: float
    n_train: int
    n_val: int
    n_test: int
    n_classes: int
    best_C: float
    eval_mode: str  # "standard", "multi_label", "binary_polarity"


@dataclass
class ProbeTransferResult:
    """Result of cross-template probe transfer (spec Section 5.6.4 point 3)."""
    task: str
    train_template: str
    eval_template: str
    layer: int
    accuracy: float


class LinearProber:
    """
    Train per-(task, template, layer) linear probes (spec Section 5.6).

    Key differences from prior version:
    - One probe per (task, template, layer), not per (task, layer)
    - 70/15/15 train/val/test split (not 70/30)
    - C sweep [0.01, 0.1, 1.0, 10.0] on validation set
    - Multi-label evaluation for ambiguous tasks (synonym, hypernym, object_color)
    - Binary polarity classification for sentiment_flip
    - Cross-template probe transfer evaluation
    - Probing uses zero-shot activations (bare template + input, no ICL context)
    """

    def __init__(self, model: ModelWrapper, config: ExperimentConfig):
        self.model = model
        self.config = config

    def _get_eval_mode(self, task_spec: TaskSpec) -> str:
        """Determine evaluation mode per spec Section 5.6.2."""
        if task_spec.name == "sentiment_flip":
            return "binary_polarity"
        if task_spec.alternative_outputs:
            return "multi_label"
        return "standard"

    def _make_polarity_labels(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Binary sentiment labels: all inputs in our data are positive sentiment."""
        return np.zeros(len(pairs), dtype=int)  # 0 = positive

    def _split_data(
        self, pairs: List[Tuple[str, str]],
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """70/15/15 train/val/test split (spec Section 5.6.3)."""
        n = len(pairs)
        n_train = int(n * self.config.science.probe_train_fraction)
        n_val = int(n * self.config.science.probe_val_fraction)
        return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]

    def probe_task_template(
        self,
        task_spec: TaskSpec,
        template_id: str,
        layers_1idx: List[int],
        cached_activations: Optional[Dict[int, torch.Tensor]] = None,
    ) -> List[ProbeResult]:
        """
        Probe for task output at each layer for one (task, template).

        Uses zero-shot activations (bare template + input, no ICL demos).
        If cached_activations is provided, skips the expensive forward pass.
        """
        template_str = task_spec.templates[template_id]
        eval_mode = self._get_eval_mode(task_spec)

        pairs = list(task_spec.pairs)
        train_pairs, val_pairs, test_pairs = self._split_data(pairs)

        if len(test_pairs) < 3:
            logger.warning(
                "Probe for '%s'/%s has only %d test examples",
                task_spec.name, template_id, len(test_pairs),
            )

        # Labels
        if eval_mode == "binary_polarity":
            y_all = np.zeros(len(pairs), dtype=int)
            n_classes = 2
            le = None
        else:
            train_labels = [out for _, out in train_pairs]
            val_labels = [out for _, out in val_pairs]
            test_labels = [out for _, out in test_pairs]
            le = LabelEncoder()
            all_labels = train_labels + val_labels + test_labels
            le.fit(all_labels)
            y_all = le.transform(all_labels)
            n_classes = len(le.classes_)

        n_train = len(train_pairs)
        n_val = len(val_pairs)
        y_train = y_all[:n_train]
        y_val = y_all[n_train:n_train + n_val]
        y_test = y_all[n_train + n_val:]

        # Get activations — prefer cached to avoid redundant forward passes
        if cached_activations is not None:
            acts = cached_activations
        else:
            # Format bare prompts (zero-shot, no ICL)
            train_prompts = [template_str.replace("{X}", inp) for inp, _ in train_pairs]
            val_prompts = [template_str.replace("{X}", inp) for inp, _ in val_pairs]
            test_prompts = [template_str.replace("{X}", inp) for inp, _ in test_pairs]
            all_prompts = train_prompts + val_prompts + test_prompts

            layers_0idx = [l - 1 for l in layers_1idx]
            logger.info(
                "Extracting probe activations for '%s'/%s (%d examples)",
                task_spec.name, template_id, len(all_prompts),
            )
            acts = self.model.get_activations(
                all_prompts, layers_0idx,
                batch_size=self.config.ops.extraction_batch_size,
            )["resid_post"]

        results = []
        C_sweep = self.config.science.probe_regularization_sweep
        max_iter = self.config.science.probe_max_iter

        for layer_1idx in layers_1idx:
            layer_0idx = layer_1idx - 1
            if layer_0idx not in acts:
                continue

            layer_acts = acts[layer_0idx].float().numpy()
            X_train = layer_acts[:n_train]
            X_val = layer_acts[n_train:n_train + n_val]
            X_test = layer_acts[n_train + n_val:]

            if eval_mode == "binary_polarity":
                results.append(ProbeResult(
                    task=task_spec.name, template_id=template_id,
                    layer=layer_1idx, accuracy=0.5,
                    n_train=n_train, n_val=n_val, n_test=len(X_test),
                    n_classes=2, best_C=1.0, eval_mode=eval_mode,
                ))
                continue

            # C sweep on validation set (spec Section 5.6.3)
            best_C = C_sweep[0]
            best_val_acc = -1.0

            for C in C_sweep:
                clf = LogisticRegression(
                    max_iter=max_iter,
                    C=C, solver="lbfgs",
                    multi_class="multinomial" if n_classes > 2 else "auto",
                )
                try:
                    clf.fit(X_train, y_train)
                    val_acc = float(clf.score(X_val, y_val))
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_C = C
                except Exception as e:
                    logger.warning(
                        "Probe C=%s failed at L%d for '%s'/%s: %s",
                        C, layer_1idx, task_spec.name, template_id, e,
                    )

            # Retrain with best C and evaluate on test set
            clf = LogisticRegression(
                max_iter=max_iter,
                C=best_C, solver="lbfgs",
                multi_class="multinomial" if n_classes > 2 else "auto",
            )
            try:
                clf.fit(X_train, y_train)

                if eval_mode == "multi_label" and task_spec.alternative_outputs:
                    preds = clf.predict(X_test)
                    correct = 0
                    for i, (inp, _) in enumerate(test_pairs[:len(preds)]):
                        pred_label = le.inverse_transform([preds[i]])[0]
                        valid = {task_spec.get_ground_truth(inp)}
                        if inp in task_spec.alternative_outputs:
                            valid.update(task_spec.alternative_outputs[inp])
                        if pred_label in valid:
                            correct += 1
                    acc = correct / len(preds) if preds.size > 0 else 0.0
                else:
                    acc = float(clf.score(X_test, y_test))
            except Exception as e:
                logger.warning(
                    "Probe failed at L%d for '%s'/%s: %s",
                    layer_1idx, task_spec.name, template_id, e,
                )
                acc = 0.0

            results.append(ProbeResult(
                task=task_spec.name, template_id=template_id,
                layer=layer_1idx, accuracy=acc,
                n_train=n_train, n_val=n_val, n_test=len(X_test),
                n_classes=n_classes, best_C=best_C, eval_mode=eval_mode,
            ))

        return results

    def cross_template_probe_transfer(
        self,
        task_spec: TaskSpec,
        layers_1idx: List[int],
        cached_activations_by_template: Dict[str, Dict[int, torch.Tensor]],
    ) -> List[ProbeTransferResult]:
        """
        Cross-template probe transfer (spec Section 5.6.4 point 3).

        Train probe on template A's activations, evaluate on template B's.
        High transfer = template-invariant representations at this layer.

        GPU path: trains all T×L probes simultaneously (one Adam run per task),
        then evaluates every (train_template, eval_template, layer) triple with
        a single vectorised einsum — replacing ~1792 sequential sklearn fits.
        """
        results = []
        pairs = list(task_spec.pairs)
        train_pairs, val_pairs, test_pairs = self._split_data(pairs)

        if self._get_eval_mode(task_spec) == "binary_polarity":
            return results  # Skip for sentiment_flip

        n_train = len(train_pairs)
        n_val   = len(val_pairs)

        train_labels = [out for _, out in train_pairs]
        val_labels   = [out for _, out in val_pairs]
        test_labels  = [out for _, out in test_pairs]

        le = LabelEncoder()
        le.fit(train_labels + val_labels + test_labels)
        y_train = le.transform(train_labels)
        y_test  = le.transform(test_labels)
        n_classes = len(le.classes_)
        nc = max(n_classes, 2)

        template_ids  = sorted(cached_activations_by_template.keys())
        n_templates   = len(template_ids)
        tid_to_idx    = {tid: i for i, tid in enumerate(template_ids)}
        n_layers      = len(layers_1idx)

        # ---- GPU fast-path ----------------------------------------
        if torch.cuda.is_available() and n_templates >= 2:
            import torch.nn as nn
            device = torch.device("cuda")

            # Build [T, L, N, D] activation tensor
            # Only include (template, layer) pairs that exist in cache
            acts_matrix = []  # list of T rows, each L-length
            valid_tids  = []
            valid_l1s   = []

            # Figure out which layers are actually cached for ALL templates
            avail_layers = []
            for l1 in layers_1idx:
                l0 = l1 - 1
                if all(l0 in cached_activations_by_template[tid]
                       for tid in template_ids):
                    avail_layers.append(l1)
            if not avail_layers:
                avail_layers = layers_1idx  # fall through to CPU

            # Stack: [T, L, N, D]
            try:
                stacked = torch.stack([
                    torch.stack([
                        (cached_activations_by_template[tid][l1 - 1]
                         if isinstance(cached_activations_by_template[tid][l1 - 1], torch.Tensor)
                         else torch.from_numpy(cached_activations_by_template[tid][l1 - 1])
                        ).float()
                        for l1 in avail_layers
                    ])                          # [L, N, D]
                    for tid in template_ids
                ]).to(device)                   # [T, L, N, D]
            except Exception:
                stacked = None  # fall through to CPU below

            if stacked is not None:
                T, L, N, D = stacked.shape
                X_tr = stacked[:, :, :n_train]           # [T, L, n_train, D]
                X_te = stacked[:, :, n_train + n_val:]   # [T, L, n_test,  D]

                ytr_gpu = torch.from_numpy(y_train).long().to(device)
                yte_gpu = torch.from_numpy(y_test).long().to(device)

                # Flat labels: [T*L*n_train]
                ytr_flat = (ytr_gpu.unsqueeze(0).unsqueeze(0)
                            .expand(T, L, -1).reshape(-1))

                # Train all T*L probes simultaneously with Adam (C=1.0 fixed)
                wd = 1.0 / (1.0 * max(n_train, 1))
                n_steps = min(self.config.science.probe_max_iter, 1000)

                X_tr_flat = X_tr.reshape(T * L, n_train, D)    # [T*L, n_train, D]
                W = torch.zeros(T * L, nc, D, device=device, requires_grad=True)
                b = torch.zeros(T * L, nc,    device=device, requires_grad=True)
                ce  = nn.CrossEntropyLoss()
                opt = torch.optim.Adam([W, b], lr=1e-2)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=n_steps, eta_min=1e-4)

                for _ in range(n_steps):
                    opt.zero_grad()
                    logits = (torch.einsum("lnf,lcf->lnc", X_tr_flat, W)
                              + b.unsqueeze(1))            # [T*L, n_train, nc]
                    loss = ce(logits.reshape(-1, nc), ytr_flat)
                    loss = loss + 0.5 * wd * (W ** 2).sum() / (T * L)
                    loss.backward()
                    opt.step()
                    sch.step()

                # Reshape weights back to [T, L, nc, D]
                W_tl = W.detach().reshape(T, L, nc, D)
                b_tl = b.detach().reshape(T, L, nc)
                del W, b, opt, sch, X_tr_flat
                torch.cuda.empty_cache()

                # Evaluate: for each train template T_i, score on all T_j test sets
                X_te_flat = X_te.reshape(T * L, -1, D)    # [T*L, n_test, D]
                n_test = X_te.shape[2]
                yte_exp = yte_gpu.unsqueeze(0).unsqueeze(0).expand(T, L, -1)  # [T, L, n_test]

                with torch.no_grad():
                    for ti, train_tid in enumerate(template_ids):
                        W_i = W_tl[ti]   # [L, nc, D]
                        b_i = b_tl[ti]   # [L, nc]
                        # Eval on ALL templates at once: [T, L, n_test, nc]
                        logits_all = (
                            torch.einsum("tlnd,lcd->tlnc", X_te, W_i)
                            + b_i[None, :, None, :]
                        )
                        preds = logits_all.argmax(dim=-1)  # [T, L, n_test]
                        accs  = (preds == yte_exp).float().mean(dim=-1)  # [T, L]

                        for tj, eval_tid in enumerate(template_ids):
                            if eval_tid == train_tid:
                                continue
                            for l_idx, layer_1idx in enumerate(avail_layers):
                                results.append(ProbeTransferResult(
                                    task=task_spec.name,
                                    train_template=train_tid,
                                    eval_template=eval_tid,
                                    layer=layer_1idx,
                                    accuracy=float(accs[tj, l_idx].item()),
                                ))

                del stacked, X_tr, X_te, W_tl, b_tl
                torch.cuda.empty_cache()
                return results

        # ---- CPU fallback (original) ----------------------------------------
        for layer_1idx in layers_1idx:
            layer_0idx = layer_1idx - 1

            for train_tid in template_ids:
                if layer_0idx not in cached_activations_by_template[train_tid]:
                    continue

                train_acts = cached_activations_by_template[train_tid][layer_0idx].float().numpy()
                X_train = train_acts[:n_train]

                clf = LogisticRegression(
                    max_iter=self.config.science.probe_max_iter,
                    C=1.0, solver="lbfgs",
                    multi_class="multinomial" if n_classes > 2 else "auto",
                )
                try:
                    clf.fit(X_train, y_train)
                except Exception:
                    continue

                for eval_tid in template_ids:
                    if eval_tid == train_tid:
                        continue
                    if layer_0idx not in cached_activations_by_template[eval_tid]:
                        continue

                    eval_acts = cached_activations_by_template[eval_tid][layer_0idx].float().numpy()
                    X_test_cross = eval_acts[n_train + n_val:]

                    try:
                        acc = float(clf.score(X_test_cross, y_test))
                    except Exception:
                        acc = 0.0

                    results.append(ProbeTransferResult(
                        task=task_spec.name, train_template=train_tid,
                        eval_template=eval_tid, layer=layer_1idx,
                        accuracy=acc,
                    ))

        return results


def _probe_single_task_template(
    task_spec: TaskSpec,
    template_id: str,
    layers_1idx: List[int],
    acts: Dict[int, torch.Tensor],
    science_params_dict: Dict,
) -> List[ProbeResult]:
    """
    Standalone probing function for one (task, template) — suitable for
    ProcessPoolExecutor dispatch.  All data passed explicitly (no model ref).
    """
    eval_mode = "binary_polarity" if task_spec.name == "sentiment_flip" else (
        "multi_label" if task_spec.alternative_outputs else "standard"
    )

    pairs = list(task_spec.pairs)
    n = len(pairs)
    train_frac = science_params_dict.get("probe_train_fraction", 0.70)
    val_frac = science_params_dict.get("probe_val_fraction", 0.15)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    if eval_mode == "binary_polarity":
        y_all = np.zeros(n, dtype=int)
        n_classes = 2
        le = None
    else:
        le = LabelEncoder()
        all_labels = [out for _, out in pairs]
        le.fit(all_labels)
        y_all = le.transform(all_labels)
        n_classes = len(le.classes_)

    y_train = y_all[:n_train]
    y_val = y_all[n_train:n_train + n_val]
    y_test = y_all[n_train + n_val:]

    C_sweep = science_params_dict.get("probe_regularization_sweep", [0.01, 0.1, 1.0, 10.0])
    max_iter = science_params_dict.get("probe_max_iter", 2000)

    # ---- GPU fast-path: batch all layers into one LBFGS call ----
    if eval_mode != "binary_polarity" and torch.cuda.is_available():
        acts_by_layer = []
        for l1 in layers_1idx:
            l0 = l1 - 1
            if l0 in acts:
                a = acts[l0]
                if isinstance(a, torch.Tensor):
                    a = a.float().numpy()
                acts_by_layer.append((l1, a))

        if acts_by_layer:
            try:
                gpu_iter = min(max_iter, 500)
                gpu_out = _probe_all_layers_gpu(
                    acts_by_layer, y_train, y_val, y_test,
                    n_train, n_val, n_classes, C_sweep, gpu_iter,
                )
                results = []
                n_test_size = len(y_test)
                for layer_1idx, best_C, te_preds in gpu_out:
                    if eval_mode == "multi_label" and task_spec.alternative_outputs and le is not None:
                        correct = 0
                        for i, (inp, _) in enumerate(test_pairs[:len(te_preds)]):
                            pred_label = le.inverse_transform([int(te_preds[i])])[0]
                            valid = {task_spec.get_ground_truth(inp)}
                            if inp in task_spec.alternative_outputs:
                                valid.update(task_spec.alternative_outputs[inp])
                            if pred_label in valid:
                                correct += 1
                        acc = correct / len(te_preds) if len(te_preds) > 0 else 0.0
                    else:
                        acc = float((te_preds == y_test).mean()) if len(te_preds) > 0 else 0.0
                    results.append(ProbeResult(
                        task=task_spec.name, template_id=template_id,
                        layer=layer_1idx, accuracy=acc,
                        n_train=n_train, n_val=n_val, n_test=n_test_size,
                        n_classes=n_classes, best_C=best_C, eval_mode=eval_mode,
                    ))
                # Handle binary_polarity layers that were skipped above
                probed_layers = {r.layer for r in results}
                for l1 in layers_1idx:
                    if l1 not in probed_layers:
                        l0 = l1 - 1
                        if l0 in acts:
                            a = acts[l0]
                            n_test_s = (len(a) - n_train - n_val)
                            results.append(ProbeResult(
                                task=task_spec.name, template_id=template_id,
                                layer=l1, accuracy=0.5,
                                n_train=n_train, n_val=n_val, n_test=n_test_s,
                                n_classes=2, best_C=1.0, eval_mode=eval_mode,
                            ))
                return results
            except Exception as e:
                logger.warning(
                    "GPU probe failed for '%s'/%s — falling back to CPU: %s",
                    task_spec.name, template_id, e,
                )

    results = []
    for layer_1idx in layers_1idx:
        layer_0idx = layer_1idx - 1
        if layer_0idx not in acts:
            continue

        layer_acts = acts[layer_0idx]
        if isinstance(layer_acts, torch.Tensor):
            layer_acts = layer_acts.float().numpy()
        X_train = layer_acts[:n_train]
        X_val = layer_acts[n_train:n_train + n_val]
        X_test = layer_acts[n_train + n_val:]

        if eval_mode == "binary_polarity":
            results.append(ProbeResult(
                task=task_spec.name, template_id=template_id,
                layer=layer_1idx, accuracy=0.5,
                n_train=n_train, n_val=n_val, n_test=len(X_test),
                n_classes=2, best_C=1.0, eval_mode=eval_mode,
            ))
            continue

        best_C = C_sweep[0]
        best_val_acc = -1.0
        for C in C_sweep:
            clf = LogisticRegression(
                max_iter=max_iter, C=C, solver="lbfgs",
                multi_class="multinomial" if n_classes > 2 else "auto",
            )
            try:
                clf.fit(X_train, y_train)
                val_acc = float(clf.score(X_val, y_val))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_C = C
            except Exception:
                pass

        clf = LogisticRegression(
            max_iter=max_iter, C=best_C, solver="lbfgs",
            multi_class="multinomial" if n_classes > 2 else "auto",
        )
        try:
            clf.fit(X_train, y_train)
            if eval_mode == "multi_label" and task_spec.alternative_outputs:
                preds = clf.predict(X_test)
                correct = 0
                for i, (inp, _) in enumerate(test_pairs[:len(preds)]):
                    pred_label = le.inverse_transform([preds[i]])[0]
                    valid = {task_spec.get_ground_truth(inp)}
                    if inp in task_spec.alternative_outputs:
                        valid.update(task_spec.alternative_outputs[inp])
                    if pred_label in valid:
                        correct += 1
                acc = correct / len(preds) if preds.size > 0 else 0.0
            else:
                acc = float(clf.score(X_test, y_test))
        except Exception:
            acc = 0.0

        results.append(ProbeResult(
            task=task_spec.name, template_id=template_id,
            layer=layer_1idx, accuracy=acc,
            n_train=n_train, n_val=n_val, n_test=len(X_test),
            n_classes=n_classes, best_C=best_C, eval_mode=eval_mode,
        ))

    return results


# ---------------------------------------------------------------------------
# GPU-batched probe — all layers in one LBFGS call
# ---------------------------------------------------------------------------

def _probe_all_layers_gpu(
    acts_by_layer: List[Tuple[int, "np.ndarray"]],
    y_train: "np.ndarray",
    y_val: "np.ndarray",
    y_test: "np.ndarray",
    n_train: int,
    n_val: int,
    n_classes: int,
    C_sweep: List[float],
    max_iter: int = 500,
) -> List[Tuple[int, float, "np.ndarray"]]:
    """
    Fit logistic probes for ALL layers simultaneously on GPU using Adam.

    All 32 layers are stacked into a single weight tensor [L, nc, D] and
    optimised in one forward/backward pass per step.  Adam is used instead
    of LBFGS to avoid the expensive strong-Wolfe line search that dominates
    wall-clock time when n_samples is small.

    Args:
        acts_by_layer: [(layer_1idx, acts_array [n_samples, d_model]), ...]
        y_train/val/test: integer label arrays
        n_train, n_val: split sizes
        n_classes: number of output classes
        C_sweep: regularisation values to try
        max_iter: Adam steps per C value (capped at 1000 for speed)

    Returns:
        [(layer_1idx, best_C, test_predictions_np), ...]  — same order as input.
    """
    import torch.nn as nn

    device = torch.device("cuda")
    n_layers = len(acts_by_layer)
    nc = max(n_classes, 2)
    n_steps = min(max_iter, 1000)  # Adam converges fast; cap for speed

    # Stack: [L, N, D]
    stacked = torch.stack([
        (torch.from_numpy(a) if isinstance(a, np.ndarray) else a.cpu()).float()
        for _, a in acts_by_layer
    ]).to(device)

    n_features = stacked.shape[2]
    X_tr = stacked[:, :n_train]                    # [L, n_train, D]
    X_v  = stacked[:, n_train:n_train + n_val]     # [L, n_val,   D]
    X_te = stacked[:, n_train + n_val:]            # [L, n_test,  D]

    ytr = torch.from_numpy(y_train).long().to(device)
    yv  = torch.from_numpy(y_val).long().to(device)
    yte = torch.from_numpy(y_test).long().to(device)

    # Expand labels for batched cross-entropy: [L*n_train]
    ytr_flat = ytr.unsqueeze(0).expand(n_layers, -1).reshape(-1)
    yv_exp   = yv.unsqueeze(0).expand(n_layers, -1)   # [L, n_val]

    ce = nn.CrossEntropyLoss()
    best_Cs    = [C_sweep[0]] * n_layers
    best_vaccs = [-1.0]       * n_layers

    def _run_adam(X_train_g, ytr_f, wd, n_lay):
        """Train all n_lay layers simultaneously with Adam."""
        W = torch.zeros(n_lay, nc, n_features, device=device, requires_grad=True)
        b = torch.zeros(n_lay, nc,              device=device, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-4)
        for _ in range(n_steps):
            opt.zero_grad()
            logits = torch.einsum("lnf,lcf->lnc", X_train_g, W) + b.unsqueeze(1)
            loss = ce(logits.reshape(-1, nc), ytr_f)
            loss = loss + 0.5 * wd * (W ** 2).sum() / n_lay
            loss.backward()
            opt.step()
            sched.step()
        return W, b

    # ---- C sweep ----
    for C in C_sweep:
        wd = 1.0 / (C * max(n_train, 1))
        W, b = _run_adam(X_tr, ytr_flat, wd, n_layers)

        with torch.no_grad():
            vl = torch.einsum("lnf,lcf->lnc", X_v, W) + b.unsqueeze(1)
            vp = vl.argmax(dim=-1)                    # [L, n_val]
            vaccs = (vp == yv_exp).float().mean(dim=1).cpu().tolist()

        for li, va in enumerate(vaccs):
            if va > best_vaccs[li]:
                best_vaccs[li] = va
                best_Cs[li] = C

        del W, b
        torch.cuda.empty_cache()

    # ---- Retrain with best C (group layers by C to minimise kernel launches) ----
    test_preds_list: List[Optional["np.ndarray"]] = [None] * n_layers

    for C in set(best_Cs):
        sub = [li for li, bc in enumerate(best_Cs) if bc == C]
        ns  = len(sub)
        wd  = 1.0 / (C * max(n_train, 1))

        X_tr_s = X_tr[sub]                            # [ns, n_train, D]
        X_te_s = X_te[sub]                            # [ns, n_test,  D]
        ytr_s  = ytr.unsqueeze(0).expand(ns, -1).reshape(-1)

        W, b = _run_adam(X_tr_s, ytr_s, wd, ns)

        with torch.no_grad():
            tl = torch.einsum("lnf,lcf->lnc", X_te_s, W) + b.unsqueeze(1)
            tp = tl.argmax(dim=-1).cpu().numpy()       # [ns, n_test]

        for sub_i, li in enumerate(sub):
            test_preds_list[li] = tp[sub_i]

        del W, b, X_tr_s, X_te_s
        torch.cuda.empty_cache()

    del stacked, X_tr, X_v, X_te
    torch.cuda.empty_cache()

    return [(acts_by_layer[li][0], best_Cs[li], test_preds_list[li])
            for li in range(n_layers)]


def compute_readability_steerability_gap(
    probe_results: List[ProbeResult],
    iid_summaries: List[IIDSummary],
) -> Dict[str, Any]:
    """
    Compute readability-steerability gap per task (spec Section 5.6.4 point 4).

    For each task, compare best probe accuracy vs best FV steering accuracy.
    """
    # Best probe accuracy per task
    probe_by_task: Dict[str, List[float]] = {}
    for pr in probe_results:
        probe_by_task.setdefault(pr.task, []).append(pr.accuracy)

    # Best steering accuracy per task
    steer_by_task: Dict[str, List[float]] = {}
    for s in iid_summaries:
        steer_by_task.setdefault(s.task, []).append(s.best_accuracy)

    gaps = {}
    for task in probe_by_task:
        best_probe = max(probe_by_task[task])
        best_steer = max(steer_by_task.get(task, [0.0]))
        gaps[task] = {
            "best_probe_accuracy": best_probe,
            "best_steering_accuracy": best_steer,
            "gap": best_probe - best_steer,
            "readable_not_steerable": best_probe > 0.30 and best_steer < 0.10,
        }

    return gaps


def wilcoxon_probe_vs_steering(
    probe_results: List[ProbeResult],
    iid_summaries: List[IIDSummary],
) -> Dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test: probe acc vs FV acc (spec Section 5.6.7 point 3).
    """
    # Build paired observations per task
    iid_lookup: Dict[Tuple[str, str], float] = {}
    for s in iid_summaries:
        iid_lookup[(s.task, s.template_id)] = s.best_accuracy

    results_by_task: Dict[str, Dict[str, List[float]]] = {}
    for pr in probe_results:
        key = (pr.task, pr.template_id)
        results_by_task.setdefault(pr.task, {}).setdefault("probe", []).append(pr.accuracy)
        steer_acc = iid_lookup.get(key, 0.0)
        results_by_task[pr.task].setdefault("steer", []).append(steer_acc)

    test_results = {}
    n_tasks = len(results_by_task)

    for task, data in results_by_task.items():
        probe_accs = np.array(data["probe"])
        steer_accs = np.array(data["steer"])

        if len(probe_accs) < 5:
            test_results[task] = {"error": "too few observations"}
            continue

        try:
            stat, p = wilcoxon(probe_accs, steer_accs)
            # Bonferroni correction
            p_corrected = min(p * n_tasks, 1.0)
            test_results[task] = {
                "statistic": float(stat),
                "p_value": float(p),
                "p_value_bonferroni": float(p_corrected),
                "n_observations": len(probe_accs),
                "mean_probe": float(np.mean(probe_accs)),
                "mean_steer": float(np.mean(steer_accs)),
            }
        except Exception as e:
            test_results[task] = {"error": str(e)}

    return test_results


# ---------------------------------------------------------------------------
# 2. Activation Patching (IID-gated)
# ---------------------------------------------------------------------------
@dataclass
class PatchingResult:
    task: str
    source_template: str
    target_template: str
    layer: int
    component: str
    recovery_accuracy: float
    n_correct: int
    n_total: int


class ActivationPatcher:
    """
    Activation patching for dissociation cases.

    HARD CONSTRAINT: Only runs on tasks/templates with IID accuracy above threshold.
    """

    def __init__(self, model: ModelWrapper, config: ExperimentConfig):
        self.model = model
        self.config = config

    def is_iid_viable(
        self,
        task: str,
        template: str,
        iid_summaries: List[IIDSummary],
    ) -> Tuple[bool, float]:
        """Check if a task/template pair passes the IID gate."""
        threshold = self.config.science.iid_accuracy_threshold
        for s in iid_summaries:
            if s.task == task and s.template_id == template:
                return s.best_accuracy >= threshold, s.best_accuracy
        return False, 0.0

    @torch.no_grad()
    def patch_layer(
        self,
        task_spec: TaskSpec,
        source_template: str,
        target_template: str,
        layer_0idx: int,
        test_examples: List[Tuple[str, str]],
        component: str = "resid_post",
    ) -> PatchingResult:
        """
        Patch activations from source template into target template run.

        Batched: processes multiple examples per forward pass for speed.
        """
        src_template_str = task_spec.templates[source_template]
        tgt_template_str = task_spec.templates[target_template]
        batch_size = self.config.ops.patching_batch_size

        if component == "resid_post":
            hook_name = self.model.resid_post_hook(layer_0idx)
        elif component == "attn_out":
            hook_name = self.model.attn_out_hook(layer_0idx)
        elif component == "mlp_out":
            hook_name = self.model.mlp_out_hook(layer_0idx)
        else:
            raise ValueError(f"Unknown component: {component}")

        correct = 0

        for i in range(0, len(test_examples), batch_size):
            batch = test_examples[i:i + batch_size]

            clean_prompts = [src_template_str.replace("{X}", inp) for inp, _ in batch]
            corrupted_prompts = [tgt_template_str.replace("{X}", inp) for inp, _ in batch]

            # -- Get clean last-position activations (batched) --
            clean_tokens = self.model.model.to_tokens(clean_prompts, prepend_bos=True)
            _, clean_cache = self.model.model.run_with_cache(
                clean_tokens.to(self.model.device),
                names_filter=[hook_name],
                return_type="logits",
            )
            # Extract last-position activation per example (handles padding)
            clean_all = clean_cache[hook_name]  # [batch, seq_len, d_model]
            # Find actual last token position for each sequence
            pad_id = self.model.tokenizer.pad_token_id
            if pad_id is not None:
                clean_mask = clean_tokens != pad_id
                clean_last_pos = clean_mask.sum(dim=1) - 1  # [batch]
            else:
                clean_last_pos = torch.full(
                    (clean_tokens.shape[0],), clean_tokens.shape[1] - 1,
                    dtype=torch.long,
                )
            clean_last_acts = torch.stack([
                clean_all[j, clean_last_pos[j], :]
                for j in range(len(clean_prompts))
            ]).cpu().float()  # [batch, d_model]
            del clean_cache

            # -- Patched forward (batched) --
            corrupted_tokens = self.model.model.to_tokens(
                corrupted_prompts, prepend_bos=True,
            )
            if pad_id is not None:
                corr_mask = corrupted_tokens != pad_id
                corr_last_pos = corr_mask.sum(dim=1) - 1
            else:
                corr_last_pos = torch.full(
                    (corrupted_tokens.shape[0],), corrupted_tokens.shape[1] - 1,
                    dtype=torch.long,
                )

            _clean_acts = clean_last_acts
            _corr_pos = corr_last_pos

            def patch_hook(acts, hook, _ca=_clean_acts, _cp=_corr_pos):
                for j in range(acts.shape[0]):
                    acts[j, _cp[j], :] = _ca[j].to(acts.dtype).to(acts.device)
                return acts

            patched_logits = self.model.model.run_with_hooks(
                corrupted_tokens.to(self.model.device),
                fwd_hooks=[(hook_name, patch_hook)],
                return_type="logits",
            )

            # Check accuracy per example in the batch
            for j, (inp, expected) in enumerate(batch):
                next_id = patched_logits[j, corr_last_pos[j], :].argmax().item()
                predicted = self.model.tokenizer.decode([next_id]).strip()
                if task_spec.check_accuracy(inp, predicted):
                    correct += 1

            del patched_logits
            if self.model.device == "cuda":
                torch.cuda.empty_cache()

        n_total = len(test_examples)
        layer_1idx = layer_0idx + 1

        return PatchingResult(
            task=task_spec.name,
            source_template=source_template,
            target_template=target_template,
            layer=layer_1idx,
            component=component,
            recovery_accuracy=correct / n_total if n_total > 0 else 0,
            n_correct=correct,
            n_total=n_total,
        )

    @torch.no_grad()
    def _get_clean_acts_all_layers(
        self,
        prompts: List[str],
        layers_0idx: List[int],
        component: str = "resid_post",
    ) -> Dict[int, torch.Tensor]:
        """
        Extract clean last-position activations at ALL layers in one forward pass.

        Returns:
            {layer_0idx: tensor [n_prompts, d_model]} on CPU.
        """
        hook_names = []
        hook_to_layer = {}
        for layer in layers_0idx:
            if component == "resid_post":
                name = self.model.resid_post_hook(layer)
            elif component == "attn_out":
                name = self.model.attn_out_hook(layer)
            elif component == "mlp_out":
                name = self.model.mlp_out_hook(layer)
            else:
                raise ValueError(f"Unknown component: {component}")
            hook_names.append(name)
            hook_to_layer[name] = layer

        batch_size = self.config.ops.patching_batch_size
        # Accumulate per layer
        layer_acts: Dict[int, List[torch.Tensor]] = {l: [] for l in layers_0idx}

        pad_id = self.model.tokenizer.pad_token_id

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tokens = self.model.model.to_tokens(batch, prepend_bos=True)
            _, cache = self.model.model.run_with_cache(
                tokens.to(self.model.device),
                names_filter=hook_names,
                return_type="logits",
            )

            # Find last token positions
            if pad_id is not None:
                mask = tokens != pad_id
                last_pos = mask.sum(dim=1) - 1
            else:
                last_pos = torch.full(
                    (tokens.shape[0],), tokens.shape[1] - 1,
                    dtype=torch.long,
                )

            for name, layer in hook_to_layer.items():
                all_acts = cache[name]  # [batch, seq, d_model]
                batch_last = torch.stack([
                    all_acts[j, last_pos[j], :] for j in range(len(batch))
                ]).cpu().float()
                layer_acts[layer].append(batch_last)

            del cache
            if self.model.device == "cuda":
                torch.cuda.empty_cache()

        return {l: torch.cat(chunks, dim=0) for l, chunks in layer_acts.items()}

    @torch.no_grad()
    def _patch_with_precomputed(
        self,
        task_spec: TaskSpec,
        target_template: str,
        layer_0idx: int,
        test_examples: List[Tuple[str, str]],
        clean_acts: torch.Tensor,
        component: str = "resid_post",
        source_template: str = "",
    ) -> PatchingResult:
        """
        Run patched forward using pre-computed clean activations (no redundant
        clean forward pass).
        """
        tgt_template_str = task_spec.templates[target_template]
        batch_size = self.config.ops.patching_batch_size

        if component == "resid_post":
            hook_name = self.model.resid_post_hook(layer_0idx)
        elif component == "attn_out":
            hook_name = self.model.attn_out_hook(layer_0idx)
        elif component == "mlp_out":
            hook_name = self.model.mlp_out_hook(layer_0idx)
        else:
            raise ValueError(f"Unknown component: {component}")

        pad_id = self.model.tokenizer.pad_token_id
        correct = 0

        for i in range(0, len(test_examples), batch_size):
            batch = test_examples[i:i + batch_size]
            batch_clean_acts = clean_acts[i:i + len(batch)]

            corrupted_prompts = [tgt_template_str.replace("{X}", inp) for inp, _ in batch]
            corrupted_tokens = self.model.model.to_tokens(
                corrupted_prompts, prepend_bos=True,
            )

            if pad_id is not None:
                corr_mask = corrupted_tokens != pad_id
                corr_last_pos = corr_mask.sum(dim=1) - 1
            else:
                corr_last_pos = torch.full(
                    (corrupted_tokens.shape[0],), corrupted_tokens.shape[1] - 1,
                    dtype=torch.long,
                )

            _ca = batch_clean_acts
            _cp = corr_last_pos

            def patch_hook(acts, hook, _ca=_ca, _cp=_cp):
                for j in range(acts.shape[0]):
                    acts[j, _cp[j], :] = _ca[j].to(acts.dtype).to(acts.device)
                return acts

            patched_logits = self.model.model.run_with_hooks(
                corrupted_tokens.to(self.model.device),
                fwd_hooks=[(hook_name, patch_hook)],
                return_type="logits",
            )

            for j, (inp, expected) in enumerate(batch):
                next_id = patched_logits[j, corr_last_pos[j], :].argmax().item()
                predicted = self.model.tokenizer.decode([next_id]).strip()
                if task_spec.check_accuracy(inp, predicted):
                    correct += 1

            del patched_logits
            if self.model.device == "cuda":
                torch.cuda.empty_cache()

        return PatchingResult(
            task=task_spec.name,
            source_template=source_template,
            target_template=target_template,
            layer=layer_0idx + 1,
            component=component,
            recovery_accuracy=correct / len(test_examples) if test_examples else 0,
            n_correct=correct,
            n_total=len(test_examples),
        )

    def _select_cases_stratified(
        self,
        dissociation_cases: List[Dict],
        iid_summaries: List[IIDSummary],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Stratified case selection: top K per task, IID-filtered FIRST.

        Returns (selected_cases, skipped_cases).

        Instead of a global top-N that lets one task monopolize all slots,
        we: (1) filter to IID-viable cases upfront, (2) group by task,
        (3) pick top cases_per_task from each task (sorted by dissociation
        gap = cosine - transfer_accuracy), (4) cap at max_cases total.
        """
        tasks = get_tasks()
        threshold = self.config.science.iid_accuracy_threshold
        cases_per_task = self.config.science.patching_cases_per_task
        max_cases = self.config.science.patching_max_cases

        # Step 1: Filter IID-viable cases upfront (don't waste slots)
        viable_cases = []
        skipped = []
        for case in dissociation_cases:
            task_name = case.get("task", "")
            source = case.get("source", "")
            target = case.get("target", "")

            if task_name not in tasks:
                continue

            viable, iid_acc = self.is_iid_viable(task_name, source, iid_summaries)
            if not viable:
                skipped.append({
                    "case": case,
                    "reason": (
                        f"IID accuracy {iid_acc:.3f} < threshold {threshold:.2f} "
                        f"for '{task_name}' {source}->{target}"
                    ),
                    "iid_accuracy": iid_acc,
                })
                continue
            viable_cases.append(case)

        # Step 2: Group by task
        by_task: Dict[str, List[Dict]] = defaultdict(list)
        for case in viable_cases:
            by_task[case["task"]].append(case)

        # Step 3: Within each task, sort by dissociation gap and pick top K
        selected = []
        for task_name in sorted(by_task.keys()):
            task_cases = sorted(
                by_task[task_name],
                key=lambda c: c.get("cosine", 0) - c.get("transfer_accuracy", 1),
                reverse=True,
            )
            selected.extend(task_cases[:cases_per_task])

        # Step 4: If we exceed max_cases, trim globally (still stratified —
        # sort all selected by gap and keep the strongest max_cases)
        if len(selected) > max_cases:
            selected = sorted(
                selected,
                key=lambda c: c.get("cosine", 0) - c.get("transfer_accuracy", 1),
                reverse=True,
            )[:max_cases]

        logger.info(
            "Patching case selection: %d IID-viable cases across %d tasks "
            "-> %d selected (%d per task cap, %d max), %d skipped (IID-gated)",
            len(viable_cases), len(by_task), len(selected),
            cases_per_task, max_cases, len(skipped),
        )

        return selected, skipped

    def analyze_dissociation_cases(
        self,
        dissociation_cases: List[Dict],
        iid_summaries: List[IIDSummary],
        n_examples: int = 15,
        components: List[str] = ("resid_post",),
    ) -> Tuple[List[PatchingResult], Dict[str, Any]]:
        """
        Run patching on dissociation cases, WITH IID GATING.

        Case selection is stratified per-task (not global top-N) so every
        IID-viable task gets representation. IID filtering happens BEFORE
        slot allocation so no slots are wasted on non-viable cases.

        Optimized: extracts clean activations at ALL layers in ONE forward
        pass per case, then runs patched forward per layer (halves GPU work).
        """
        tasks = get_tasks()
        threshold = self.config.science.iid_accuracy_threshold

        # Stratified, IID-pre-filtered selection
        selected_cases, skipped = self._select_cases_stratified(
            dissociation_cases, iid_summaries,
        )

        results = []
        # Track per-task counts for summary
        task_counts: Dict[str, int] = {}

        for case in tqdm(selected_cases, desc="Patching cases", unit="case"):
            task_name = case.get("task", "")
            source = case.get("source", "")
            target = case.get("target", "")

            task_spec = tasks[task_name]
            test_examples = task_spec.pairs[:n_examples]
            layers_0idx = self.model.extraction_layers_0indexed()

            task_counts[task_name] = task_counts.get(task_name, 0) + 1

            # Extract clean activations at ALL layers in ONE forward pass
            for comp in components:
                clean_acts_all = self._get_clean_acts_all_layers(
                    [task_spec.templates[source].replace("{X}", inp)
                     for inp, _ in test_examples],
                    layers_0idx, component=comp,
                )

                # Patched forward per layer (reusing pre-computed clean acts)
                for layer_0idx in layers_0idx:
                    pr = self._patch_with_precomputed(
                        task_spec, target, layer_0idx,
                        test_examples,
                        clean_acts_all[layer_0idx],
                        component=comp,
                        source_template=source,
                    )
                    results.append(pr)

                del clean_acts_all

            self.model.cleanup()

        n_layers = max(len(self.model.spec.extraction_layers()), 1)
        summary = {
            "n_cases_analyzed": len(results) // n_layers,
            "n_cases_skipped_iid": len(skipped),
            "n_tasks_represented": len(task_counts),
            "cases_per_task": dict(task_counts),
            "selection_strategy": "stratified_per_task",
            "cases_per_task_cap": self.config.science.patching_cases_per_task,
            "max_cases_cap": self.config.science.patching_max_cases,
            "skipped_details": skipped,
            "iid_threshold": threshold,
        }

        return results, summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_mechanistic_analysis(
    model: ModelWrapper,
    config: ExperimentConfig,
    fvs: FVCollection,
    steering_results: Dict,
    iid_summaries: List[IIDSummary],
    geometric_results: Dict,
) -> Dict[str, Any]:
    """
    Run all mechanistic experiments with appropriate gating.

    1. Readability analysis (logit lens + FV vocab projection) — replaces
       the broken linear probing stage.  Uses cached activations from Stage 3.
    2. Activation patching (IID-gated)
    """
    results: Dict[str, Any] = {}

    # -- 1. Readability Analysis (replaces linear probing) --
    if "probe" in config.stages or "readability" in config.stages:
        from .readability import run_readability_analysis, save_readability_results
        logger.info("Running readability analysis (logit lens + FV vocab projection)...")

        readability_results = run_readability_analysis(
            model, config, fvs, iid_summaries or [],
            steering_results=steering_results,
        )
        results.update(readability_results)
        save_readability_results(readability_results, config.results_dir)

    # -- 2. Activation Patching (IID-gated) --
    if "mechanistic" in config.stages:
        logger.info("Running activation patching (IID-gated)...")
        patcher = ActivationPatcher(model, config)

        dissociation_cases = geometric_results.get("dissociation_cases", [])
        patching_results, patching_summary = patcher.analyze_dissociation_cases(
            dissociation_cases, iid_summaries,
            n_examples=15,
        )

        results["patching"] = {
            "results": [asdict(pr) for pr in patching_results],
            "summary": patching_summary,
        }

    return results


def save_mechanistic_results(results: Dict, output_dir: Path):
    path = output_dir / "mechanistic_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Mechanistic results saved to %s", path)
