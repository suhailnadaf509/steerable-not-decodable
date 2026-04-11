"""
Tuned Lens Analysis: Learned Per-Layer Translators for Enhanced Readability
===========================================================================
Complements the parameter-free logit lens (Stage 7) with per-layer affine
translators that correct for representational dialect mismatch across layers.

The logit lens projects intermediate activations through the final layer's
LayerNorm and unembedding matrix, but intermediate layers encode information
in different "dialects" than the final layer.  The tuned lens (Belrose et al.,
2023) trains a small per-layer affine corrector T_ℓ to translate layer-ℓ
representations before projection:

    tuned_lens(h_ℓ) = LayerNorm_final(T_ℓ(h_ℓ)) @ W_U

If tuned lens ALSO fails to decode where logit lens fails, the paper's
"steerability-without-decodability" claim is strengthened (H2).  If tuned
lens succeeds where logit lens fails, the claim needs qualification (H1).

The translators are task-agnostic (trained on representation geometry, not
task labels), avoiding the probe-complexity critique while providing a
better approximation of "what the network knows at layer ℓ."

Analyses:
1. **Translator Training** — Per-layer diagonal + optional low-rank affine
   translators trained on cached zero-shot activations from Stage 3,
   targeting final-layer activations with MSE loss, identity initialization,
   and early stopping.
2. **Tuned Lens Readability** — Project zero-shot activations through
   T_ℓ → ln_final → W_U to measure task decodability with dialect correction.
3. **Tuned Lens FV Projection** — Project extracted FVs through T_ℓ →
   ln_final → W_U to test whether FV vocabulary becomes coherent (H4).
4. **Expanded 2×3 Dissociation Matrix** — Compare logit lens vs tuned lens
   vs FV steering across all task×model instances (H5).
5. **Layer-Shift Analysis** — Where decodable information first appears
   with vs without dialect correction (H3).
6. **FV Coherence Comparison** — Whether tuned lens makes FV vocab
   projections more interpretable (H4).
"""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import ExperimentConfig, ModelSpec
from .extraction import FVCollection, FunctionVector
from .models import ModelWrapper
from .steering import IIDSummary
from .tasks import TaskSpec, get_tasks

logger = logging.getLogger(__name__)

# Default translator rank.  0 = diagonal-only (recommended for ~8K training
# examples).  Values 1-8 add a low-rank rotation correction.
DEFAULT_RANK = 0


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TunedLensResult:
    """Tuned-lens readability result for one (task, template, layer)."""
    task: str
    template_id: str
    layer: int          # 1-indexed (matches FV layer convention)
    top_1_accuracy: float
    top_5_accuracy: float
    top_10_accuracy: float
    mean_correct_rank: float
    mean_correct_logprob: float
    n_examples: int


@dataclass
class TunedLensFVVocabResult:
    """Tuned-lens FV vocabulary projection for one (task, template, layer)."""
    task: str
    template_id: str
    layer: int
    top_positive_tokens: List[str]
    top_positive_logits: List[float]
    top_negative_tokens: List[str]
    top_negative_logits: List[float]
    correct_output_fraction: float
    task_relevant_fraction: float
    fv_norm: float


# ---------------------------------------------------------------------------
# Per-layer translator
# ---------------------------------------------------------------------------

class TunedLensTranslator(nn.Module):
    """
    Per-layer affine translator with identity initialization.

    Default (rank=0): diagonal scaling + bias.
        T(h) = (1 + delta_scale) * h + bias

    With rank > 0: adds a low-rank rotation correction.
        T(h) = (1 + delta_scale) * h + h @ U @ V + bias

    Identity-initialized: untrained translator = logit lens (no learned
    correction).  Weight decay on delta_scale penalises deviation from 1.

    The diagonal-only variant has 2 * d_model parameters — well-suited
    to the ~8K training examples available from Stage 3 caches.
    """

    def __init__(self, d_model: int, rank: int = 0):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.delta_scale = nn.Parameter(torch.zeros(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        if rank > 0:
            self.U = nn.Parameter(torch.zeros(d_model, rank))
            self.V = nn.Parameter(torch.zeros(rank, d_model))
            nn.init.normal_(self.U, std=0.001 / (rank ** 0.5))
            nn.init.normal_(self.V, std=0.001 / (rank ** 0.5))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., d_model] -> [..., d_model]"""
        out = (1.0 + self.delta_scale) * h + self.bias
        if self.rank > 0:
            out = out + (h @ self.U) @ self.V
        return out


# ---------------------------------------------------------------------------
# Translator training
# ---------------------------------------------------------------------------

class TunedLensTrainer:
    """
    Train per-layer affine translators from cached Stage 3 activations.

    For each intermediate layer ℓ, trains T_ℓ to minimise:
        ||T_ℓ(h_ℓ) - h_L||²
    where h_L is the final-layer activation for the same example.

    Uses 80/20 train/val split, AdamW with weight decay, early stopping.
    Training data is pooled across all tasks and templates — translators
    are *task-agnostic*, correcting representational dialect only.
    """

    def __init__(self, config: ExperimentConfig, rank: int = DEFAULT_RANK):
        self.config = config
        self.rank = rank
        self.device = config.ops.device

    def _load_training_data(
        self,
        layers_0idx: List[int],
        final_layer_0idx: int,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Load cached activations and pool across all tasks/templates.

        Returns:
            layer_data: Dict[layer_0idx] -> Tensor[N, d_model]
            final_data: Tensor[N, d_model]
        """
        probe_cache_dir = self.config.cache_dir / "activations_probe"
        tasks = get_tasks(self.config.task_names)

        intermediate_layers = [l for l in layers_0idx if l != final_layer_0idx]
        per_layer: Dict[int, List[torch.Tensor]] = {
            l: [] for l in intermediate_layers
        }
        final_chunks: List[torch.Tensor] = []

        for task_name, task_spec in tasks.items():
            for tid in task_spec.template_ids:
                path = probe_cache_dir / f"{task_name}_{tid}.pt"
                if not path.exists():
                    continue
                acts = torch.load(path, map_location="cpu", weights_only=True)

                if final_layer_0idx not in acts:
                    continue

                h_final = acts[final_layer_0idx]  # [n, d_model]
                n = h_final.shape[0]

                # Only use this file if ALL intermediate layers are present
                if not all(l in acts for l in intermediate_layers):
                    continue

                final_chunks.append(h_final)
                for l in intermediate_layers:
                    per_layer[l].append(acts[l][:n])

        if not final_chunks:
            raise RuntimeError(
                f"No cached activations found in {probe_cache_dir}. "
                "Run stage 'probe_activations' first."
            )

        final_data = torch.cat(final_chunks, dim=0)
        layer_data = {
            l: torch.cat(chunks, dim=0)
            for l, chunks in per_layer.items()
            if chunks
        }

        logger.info(
            "Loaded %d training examples across %d intermediate layers",
            final_data.shape[0], len(layer_data),
        )
        return layer_data, final_data

    def train_single_translator(
        self,
        h_l: torch.Tensor,
        h_final: torch.Tensor,
        d_model: int,
        layer_idx: int,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_epochs: int = 200,
        patience: int = 15,
        val_fraction: float = 0.2,
    ) -> Tuple[TunedLensTranslator, Dict[str, float]]:
        """Train a single per-layer translator with early stopping."""
        N = h_l.shape[0]
        n_val = max(int(N * val_fraction), 1)

        rng = torch.Generator()
        rng.manual_seed(self.config.seed + layer_idx)
        perm = torch.randperm(N, generator=rng)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        device = self.device
        h_l_train = h_l[train_idx].to(device)
        h_f_train = h_final[train_idx].to(device)
        h_l_val = h_l[val_idx].to(device)
        h_f_val = h_final[val_idx].to(device)

        translator = TunedLensTranslator(d_model, self.rank).to(device)
        optimizer = torch.optim.AdamW(
            translator.parameters(), lr=lr, weight_decay=weight_decay,
        )

        # Identity baseline (val loss with no correction)
        with torch.no_grad():
            identity_val_loss = F.mse_loss(h_l_val, h_f_val).item()

        best_val_loss = identity_val_loss
        patience_counter = 0
        best_state = None
        final_epoch = 0

        for epoch in range(max_epochs):
            # Full-batch training step
            translator.train()
            pred = translator(h_l_train)
            loss = F.mse_loss(pred, h_f_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validate
            translator.eval()
            with torch.no_grad():
                val_pred = translator(h_l_val)
                val_loss = F.mse_loss(val_pred, h_f_val).item()

            if val_loss < best_val_loss - 1e-7:
                best_val_loss = val_loss
                best_state = {
                    k: v.clone() for k, v in translator.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    final_epoch = epoch + 1
                    break
            final_epoch = epoch + 1

        if best_state is not None:
            translator.load_state_dict(best_state)
        else:
            # No improvement over identity — reset to identity
            translator = TunedLensTranslator(d_model, self.rank).to(device)

        translator.eval()

        improvement = (
            (identity_val_loss - best_val_loss) / max(identity_val_loss, 1e-8)
        )
        stats = {
            "identity_val_loss": round(identity_val_loss, 6),
            "best_val_loss": round(best_val_loss, 6),
            "improvement_pct": round(improvement * 100, 2),
            "epochs_trained": final_epoch,
            "n_train": len(train_idx),
            "n_val": n_val,
        }
        return translator, stats

    def train_all_translators(
        self,
        model_spec: ModelSpec,
    ) -> Tuple[Dict[int, TunedLensTranslator], Dict[str, Any]]:
        """
        Train translators for all intermediate extraction layers.

        Returns:
            translators: Dict[layer_0idx] -> trained TunedLensTranslator
            training_log: training statistics per layer
        """
        layers_1idx = model_spec.extraction_layers()
        layers_0idx = [l - 1 for l in layers_1idx]
        final_layer_0idx = model_spec.n_layers - 1
        d_model = model_spec.d_model

        n_intermediate = sum(1 for l in layers_0idx if l != final_layer_0idx)
        logger.info(
            "Training tuned lens translators: %d layers, rank=%d, d_model=%d",
            n_intermediate, self.rank, d_model,
        )

        layer_data, final_data = self._load_training_data(
            layers_0idx, final_layer_0idx,
        )

        translators: Dict[int, TunedLensTranslator] = {}
        training_log: Dict[str, Any] = {
            "n_training_examples": final_data.shape[0],
            "rank": self.rank,
            "d_model": d_model,
            "n_layers_trained": len(layer_data),
            "per_layer": {},
        }

        for layer_0idx in tqdm(
            sorted(layer_data.keys()),
            desc="Training translators",
            unit="layer",
        ):
            layer_1idx = layer_0idx + 1
            h_l = layer_data[layer_0idx]
            n = min(h_l.shape[0], final_data.shape[0])

            translator, stats = self.train_single_translator(
                h_l[:n], final_data[:n], d_model, layer_0idx,
            )
            translators[layer_0idx] = translator
            training_log["per_layer"][str(layer_1idx)] = stats

            logger.debug(
                "Layer %d: val_loss %.6f -> %.6f (%.1f%% improvement, %d epochs)",
                layer_1idx, stats["identity_val_loss"], stats["best_val_loss"],
                stats["improvement_pct"], stats["epochs_trained"],
            )

        improvements = [
            s["improvement_pct"]
            for s in training_log["per_layer"].values()
        ]
        if improvements:
            training_log["mean_improvement_pct"] = round(
                float(np.mean(improvements)), 2
            )
            training_log["max_improvement_pct"] = round(
                float(np.max(improvements)), 2
            )

        logger.info(
            "Translator training complete: mean improvement %.1f%%, max %.1f%%",
            training_log.get("mean_improvement_pct", 0),
            training_log.get("max_improvement_pct", 0),
        )

        # Free training data from GPU
        del layer_data, final_data
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return translators, training_log


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def save_translators(
    translators: Dict[int, TunedLensTranslator],
    training_log: Dict[str, Any],
    cache_dir: Path,
    model_key: str,
    config_hash: str,
    rank: int,
):
    """Save trained translators and training log to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    state_dicts = {
        layer: translator.state_dict()
        for layer, translator in translators.items()
    }
    path = cache_dir / f"tuned_lens_translators_{model_key}_r{rank}_{config_hash}.pt"
    torch.save(state_dicts, path)

    log_path = cache_dir / f"tuned_lens_training_{model_key}_r{rank}_{config_hash}.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    logger.info("Translators saved to %s", path)


def load_translators(
    cache_dir: Path,
    model_key: str,
    config_hash: str,
    rank: int,
    d_model: int,
    device: str = "cpu",
) -> Optional[Dict[int, TunedLensTranslator]]:
    """Load cached translators, or return None if cache miss."""
    path = cache_dir / f"tuned_lens_translators_{model_key}_r{rank}_{config_hash}.pt"
    if not path.exists():
        return None

    state_dicts = torch.load(path, map_location=device, weights_only=True)
    translators = {}
    for layer, sd in state_dicts.items():
        translator = TunedLensTranslator(d_model, rank).to(device)
        translator.load_state_dict(sd)
        translator.eval()
        translators[layer] = translator

    logger.info("Loaded %d cached translators from %s", len(translators), path)
    return translators


# ---------------------------------------------------------------------------
# Tuned Lens Analyzer
# ---------------------------------------------------------------------------

class TunedLensAnalyzer:
    """
    Project zero-shot activations through trained translators and the model's
    unembedding pipeline to measure tuned-lens readability per layer.
    """

    def __init__(
        self,
        model: ModelWrapper,
        config: ExperimentConfig,
        translators: Dict[int, TunedLensTranslator],
    ):
        self.model = model
        self.config = config
        self.translators = translators
        self.top_k_values = config.science.readability_top_k
        self.device = config.ops.device

        self._W_U = model.model.W_U.detach()
        self._ln_final = model.model.ln_final
        if hasattr(model.model, 'b_U') and model.model.b_U is not None:
            self._b_U = model.model.b_U.detach()
        else:
            self._b_U = None

    @torch.no_grad()
    def _project_to_logits(
        self, activations: torch.Tensor, layer_0idx: int,
    ) -> torch.Tensor:
        """
        Project activations through translator T_ℓ, then ln_final and W_U.

        Args:
            activations: [batch, d_model]
            layer_0idx: which layer's translator to use

        Returns:
            logits: [batch, vocab_size]
        """
        h = activations.to(self._W_U.device)

        if layer_0idx in self.translators:
            h = self.translators[layer_0idx](h)

        h_norm = self._ln_final(h)
        logits = h_norm @ self._W_U
        if self._b_U is not None:
            logits = logits + self._b_U
        return logits

    @torch.no_grad()
    def analyze_task_template(
        self,
        task_spec: TaskSpec,
        template_id: str,
        layers_1idx: List[int],
        cached_activations: Dict[int, torch.Tensor],
        batch_size: int = 256,
    ) -> List[TunedLensResult]:
        """Run tuned-lens analysis for one (task, template) across all layers."""
        tokenizer = self.model.tokenizer
        pairs = task_spec.pairs

        correct_token_ids = []
        for _, expected_output in pairs:
            tokens = tokenizer.encode(expected_output, add_special_tokens=False)
            if tokens:
                correct_token_ids.append(tokens[0])
            else:
                tokens = tokenizer.encode(
                    " " + expected_output, add_special_tokens=False,
                )
                correct_token_ids.append(tokens[0] if tokens else -1)

        correct_ids = torch.tensor(correct_token_ids, dtype=torch.long)
        results = []

        for layer_1idx in layers_1idx:
            layer_0idx = layer_1idx - 1
            if layer_0idx not in cached_activations:
                continue

            acts = cached_activations[layer_0idx]
            n_use = min(acts.shape[0], len(pairs))
            acts = acts[:n_use]
            cids = correct_ids[:n_use]

            all_ranks: List[int] = []
            all_logprobs: List[float] = []
            top_k_hits = {k: 0 for k in self.top_k_values}

            for i in range(0, n_use, batch_size):
                batch_acts = acts[i:i + batch_size]
                batch_cids = cids[i:i + batch_size]

                logits = self._project_to_logits(batch_acts, layer_0idx)
                sorted_indices = logits.argsort(dim=-1, descending=True)

                for j in range(batch_acts.shape[0]):
                    cid = batch_cids[j].item()
                    if cid < 0:
                        all_ranks.append(logits.shape[-1])
                        all_logprobs.append(-float('inf'))
                        continue

                    rank_mask = (
                        sorted_indices[j] == cid
                    ).nonzero(as_tuple=True)[0]
                    rank = (
                        rank_mask[0].item()
                        if len(rank_mask) > 0
                        else logits.shape[-1]
                    )
                    all_ranks.append(rank)

                    for k in self.top_k_values:
                        if rank < k:
                            top_k_hits[k] += 1

                    log_probs = torch.log_softmax(logits[j], dim=-1)
                    all_logprobs.append(log_probs[cid].item())

            results.append(TunedLensResult(
                task=task_spec.name,
                template_id=template_id,
                layer=layer_1idx,
                top_1_accuracy=top_k_hits.get(1, 0) / n_use,
                top_5_accuracy=top_k_hits.get(5, 0) / n_use,
                top_10_accuracy=top_k_hits.get(10, 0) / n_use,
                mean_correct_rank=float(np.mean(all_ranks)),
                mean_correct_logprob=float(np.mean(all_logprobs)),
                n_examples=n_use,
            ))

        return results


# ---------------------------------------------------------------------------
# Tuned Lens FV Projector
# ---------------------------------------------------------------------------

class TunedLensFVProjector:
    """
    Project function vectors through per-layer translators and W_U.

    Tests H4: does the tuned lens correction make FV vocab projections
    more coherent (higher correct_output_fraction)?
    """

    def __init__(
        self,
        model: ModelWrapper,
        config: ExperimentConfig,
        translators: Dict[int, TunedLensTranslator],
    ):
        self.model = model
        self.config = config
        self.translators = translators
        self.top_n = config.science.fv_vocab_top_n
        self._W_U = model.model.W_U.detach()
        self._ln_final = model.model.ln_final
        if hasattr(model.model, 'b_U') and model.model.b_U is not None:
            self._b_U = model.model.b_U.detach()
        else:
            self._b_U = None

    @torch.no_grad()
    def project_fv(
        self,
        fv: FunctionVector,
        task_spec: TaskSpec,
    ) -> TunedLensFVVocabResult:
        """Project a single FV through its layer's translator and unembedding."""
        tokenizer = self.model.tokenizer
        vec = fv.vector.to(self._W_U.device)
        layer_0idx = fv.layer_0idx

        # Apply translator
        if layer_0idx in self.translators:
            vec = self.translators[layer_0idx](vec.unsqueeze(0)).squeeze(0)

        h_norm = self._ln_final(vec.unsqueeze(0))
        logits = (h_norm @ self._W_U).squeeze(0)
        if self._b_U is not None:
            logits = logits + self._b_U

        # Top positive tokens
        top_pos_idx = logits.argsort(descending=True)[:self.top_n]
        top_pos_logits = logits[top_pos_idx]
        top_pos_tokens = [
            tokenizer.decode([idx.item()]).strip() for idx in top_pos_idx
        ]

        # Top negative tokens
        top_neg_idx = logits.argsort(descending=False)[:self.top_n]
        top_neg_logits = logits[top_neg_idx]
        top_neg_tokens = [
            tokenizer.decode([idx.item()]).strip() for idx in top_neg_idx
        ]

        # Overlap with correct outputs
        correct_first_tokens: set = set()
        correct_outputs = {out.strip().lower() for _, out in task_spec.pairs}
        for _, out in task_spec.pairs:
            toks = tokenizer.encode(out, add_special_tokens=False)
            if toks:
                correct_first_tokens.add(toks[0])
            toks_sp = tokenizer.encode(" " + out, add_special_tokens=False)
            if toks_sp:
                correct_first_tokens.add(toks_sp[0])

        n_correct = sum(
            1 for idx in top_pos_idx[:self.top_n]
            if idx.item() in correct_first_tokens
        )

        n_relevant = 0
        for tok_str in top_pos_tokens:
            tok_lower = tok_str.lower().strip()
            if not tok_lower:
                continue
            for correct in correct_outputs:
                if tok_lower in correct or correct in tok_lower:
                    n_relevant += 1
                    break

        return TunedLensFVVocabResult(
            task=fv.task,
            template_id=fv.template_id,
            layer=fv.layer,
            top_positive_tokens=top_pos_tokens,
            top_positive_logits=[float(x) for x in top_pos_logits.cpu()],
            top_negative_tokens=top_neg_tokens,
            top_negative_logits=[float(x) for x in top_neg_logits.cpu()],
            correct_output_fraction=n_correct / max(self.top_n, 1),
            task_relevant_fraction=n_relevant / max(self.top_n, 1),
            fv_norm=float(fv.norm),
        )


# ---------------------------------------------------------------------------
# Expanded 2×3 Dissociation Matrix (H5)
# ---------------------------------------------------------------------------

def compute_expanded_dissociation_matrix(
    logit_lens_results: List[Dict],
    tuned_lens_results: List[TunedLensResult],
    iid_summaries: List[IIDSummary],
    readability_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Expanded 2×3 dissociation matrix.

    Rows: {logit lens succeeds, tuned lens only (logit fails), both fail}
    Cols: {FV steering succeeds, FV steering fails}

    A decoder "succeeds" if best top-10 accuracy at any layer > threshold.
    FV steering "succeeds" if best IID accuracy > threshold.

    The critical cell is "tuned_lens_and_steerable": cases where tuned lens
    reveals decodable information that the logit lens missed.  If populated,
    H1 (gap-closing) fires.  If empty, H2 (confirmation) is supported.
    """
    # Best logit lens top-10 per task
    ll_by_task: Dict[str, float] = {}
    for r in logit_lens_results:
        task = r["task"]
        ll_by_task[task] = max(ll_by_task.get(task, 0.0), r["top_10_accuracy"])

    # Best tuned lens top-10 per task
    tl_by_task: Dict[str, float] = {}
    for r in tuned_lens_results:
        tl_by_task[r.task] = max(tl_by_task.get(r.task, 0.0), r.top_10_accuracy)

    # Best steering per task
    steer_by_task: Dict[str, float] = {}
    for s in iid_summaries:
        steer_by_task[s.task] = max(
            steer_by_task.get(s.task, 0.0), s.best_accuracy,
        )

    matrix: Dict[str, List[Dict]] = {
        "logit_lens_and_steerable": [],
        "logit_lens_only": [],
        "tuned_lens_and_steerable": [],
        "tuned_lens_only": [],
        "steerable_not_decodable": [],
        "both_fail": [],
    }

    per_task: Dict[str, Dict[str, Any]] = {}
    all_tasks = (
        set(ll_by_task.keys()) | set(tl_by_task.keys()) | set(steer_by_task.keys())
    )

    for task in sorted(all_tasks):
        ll_acc = ll_by_task.get(task, 0.0)
        tl_acc = tl_by_task.get(task, 0.0)
        steer_acc = steer_by_task.get(task, 0.0)

        ll_ok = ll_acc > readability_threshold
        tl_ok = tl_acc > readability_threshold
        steerable = steer_acc > readability_threshold

        # Row assignment (priority: logit lens > tuned lens > neither)
        if ll_ok:
            row = "logit_lens"
        elif tl_ok:
            row = "tuned_lens"
        else:
            row = "neither"

        # Cell assignment
        if row == "logit_lens" and steerable:
            cell = "logit_lens_and_steerable"
        elif row == "logit_lens" and not steerable:
            cell = "logit_lens_only"
        elif row == "tuned_lens" and steerable:
            cell = "tuned_lens_and_steerable"
        elif row == "tuned_lens" and not steerable:
            cell = "tuned_lens_only"
        elif steerable:
            cell = "steerable_not_decodable"
        else:
            cell = "both_fail"

        entry = {
            "task": task,
            "logit_lens_best_top10": round(ll_acc, 4),
            "tuned_lens_best_top10": round(tl_acc, 4),
            "steering_best_accuracy": round(steer_acc, 4),
            "cell": cell,
            "tuned_lens_delta": round(tl_acc - ll_acc, 4),
        }
        matrix[cell].append(entry)
        per_task[task] = entry

    gap_closed = matrix["tuned_lens_and_steerable"]
    gap_persists = matrix["steerable_not_decodable"]

    return {
        "matrix": matrix,
        "per_task": per_task,
        "summary": {
            "n_logit_lens_and_steerable": len(matrix["logit_lens_and_steerable"]),
            "n_logit_lens_only": len(matrix["logit_lens_only"]),
            "n_tuned_lens_and_steerable": len(matrix["tuned_lens_and_steerable"]),
            "n_tuned_lens_only": len(matrix["tuned_lens_only"]),
            "n_steerable_not_decodable": len(matrix["steerable_not_decodable"]),
            "n_both_fail": len(matrix["both_fail"]),
            "gap_closed_tasks": [e["task"] for e in gap_closed],
            "gap_persists_tasks": [e["task"] for e in gap_persists],
            "h1_fires": len(gap_closed) > 0,
            "h2_supported": len(gap_persists) > 0 and len(gap_closed) == 0,
        },
    }


# ---------------------------------------------------------------------------
# Layer-Shift Analysis (H3)
# ---------------------------------------------------------------------------

def _compute_layer_shift(
    logit_lens_results: List[Dict],
    tuned_lens_results: List[TunedLensResult],
    threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    H3: at which layer does information first become decodable for each lens?

    If tuned lens consistently finds decodable information at EARLIER layers,
    the representational dialect correction matters even for tasks where both
    lenses eventually succeed — the information is present earlier but encoded
    in a form only the tuned lens can read.
    """
    # Peak layer per task for logit lens
    ll_peak: Dict[str, Tuple[int, float]] = {}
    for r in logit_lens_results:
        task = r["task"]
        acc = r["top_10_accuracy"]
        if task not in ll_peak or acc > ll_peak[task][1]:
            ll_peak[task] = (r["layer"], acc)

    # Peak layer per task for tuned lens
    tl_peak: Dict[str, Tuple[int, float]] = {}
    for r in tuned_lens_results:
        if r.task not in tl_peak or r.top_10_accuracy > tl_peak[r.task][1]:
            tl_peak[r.task] = (r.layer, r.top_10_accuracy)

    # First decodable layer (first layer above threshold)
    ll_first: Dict[str, int] = {}
    for r in sorted(logit_lens_results, key=lambda x: x["layer"]):
        task = r["task"]
        if task not in ll_first and r["top_10_accuracy"] > threshold:
            ll_first[task] = r["layer"]

    tl_first: Dict[str, int] = {}
    for r in sorted(tuned_lens_results, key=lambda x: x.layer):
        if r.task not in tl_first and r.top_10_accuracy > threshold:
            tl_first[r.task] = r.layer

    per_task: Dict[str, Dict] = {}
    first_shifts: List[int] = []

    for task in set(ll_peak.keys()) & set(tl_peak.keys()):
        ll_peak_layer, ll_peak_acc = ll_peak[task]
        tl_peak_layer, tl_peak_acc = tl_peak[task]
        peak_shift = ll_peak_layer - tl_peak_layer  # positive = tuned earlier

        ll_fl = ll_first.get(task)
        tl_fl = tl_first.get(task)
        first_shift = None
        if ll_fl is not None and tl_fl is not None:
            first_shift = ll_fl - tl_fl
            first_shifts.append(first_shift)

        per_task[task] = {
            "logit_lens_peak_layer": ll_peak_layer,
            "logit_lens_peak_acc": round(ll_peak_acc, 4),
            "tuned_lens_peak_layer": tl_peak_layer,
            "tuned_lens_peak_acc": round(tl_peak_acc, 4),
            "peak_layer_shift": peak_shift,
            "logit_lens_first_decodable": ll_fl,
            "tuned_lens_first_decodable": tl_fl,
            "first_decodable_shift": first_shift,
        }

    return {
        "per_task": per_task,
        "mean_peak_shift": round(float(np.mean([
            v["peak_layer_shift"] for v in per_task.values()
        ])), 2) if per_task else 0.0,
        "mean_first_decodable_shift": (
            round(float(np.mean(first_shifts)), 2)
            if first_shifts else None
        ),
        "h3_supported": any(
            v.get("first_decodable_shift") is not None
            and v["first_decodable_shift"] > 0
            for v in per_task.values()
        ),
    }


# ---------------------------------------------------------------------------
# FV Coherence Comparison (H4)
# ---------------------------------------------------------------------------

def _compute_fv_coherence_comparison(
    logit_fv_vocab: List[Dict],
    tuned_fv_vocab: List[Dict],
) -> Dict[str, Any]:
    """
    H4: does tuned lens FV projection produce more coherent token distributions?

    If correct_output_fraction is higher with tuned lens, FVs encode answer
    information in a rotated/shifted representation that only the tuned lens
    can decode — significantly modifying the "incoherent FV projection" finding.
    """
    ll_by_tt: Dict[Tuple[str, str], float] = {}
    for r in logit_fv_vocab:
        ll_by_tt[(r["task"], r["template_id"])] = r["correct_output_fraction"]

    tl_by_tt: Dict[Tuple[str, str], float] = {}
    for r in tuned_fv_vocab:
        tl_by_tt[(r["task"], r["template_id"])] = r["correct_output_fraction"]

    common_keys = sorted(set(ll_by_tt.keys()) & set(tl_by_tt.keys()))
    if not common_keys:
        return {"error": "no overlapping (task, template) pairs"}

    ll_fracs = [ll_by_tt[k] for k in common_keys]
    tl_fracs = [tl_by_tt[k] for k in common_keys]
    improvements = [tl - ll for ll, tl in zip(ll_fracs, tl_fracs)]

    wilcoxon_result: Dict[str, float] = {}
    diffs = np.array(tl_fracs) - np.array(ll_fracs)
    if len(diffs) >= 5 and not np.all(diffs == 0):
        try:
            from scipy.stats import wilcoxon as wilcoxon_test
            stat, p = wilcoxon_test(ll_fracs, tl_fracs)
            wilcoxon_result = {
                "statistic": float(stat),
                "p_value": float(p),
            }
        except Exception:
            pass

    return {
        "n_pairs": len(common_keys),
        "logit_lens_mean_correct_frac": round(float(np.mean(ll_fracs)), 4),
        "tuned_lens_mean_correct_frac": round(float(np.mean(tl_fracs)), 4),
        "mean_improvement": round(float(np.mean(improvements)), 4),
        "n_improved": sum(1 for d in improvements if d > 0.01),
        "n_degraded": sum(1 for d in improvements if d < -0.01),
        "wilcoxon": wilcoxon_result,
        "h4_supported": float(np.mean(tl_fracs)) > float(np.mean(ll_fracs)) + 0.02,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_expanded_matrix(matrix_data: Dict[str, Any]):
    """Print the expanded 2×3 dissociation matrix."""
    summary = matrix_data.get("summary", {})
    matrix = matrix_data.get("matrix", {})

    print("\n" + "=" * 70)
    print("EXPANDED 2x3 DISSOCIATION MATRIX")
    print("(Logit Lens vs Tuned Lens vs FV Steering)")
    print("=" * 70)

    print(f"\n{'Cell':<35} {'N':<5} Tasks")
    print("-" * 70)

    cell_labels = [
        ("logit_lens_and_steerable", "LL Readable + Steerable"),
        ("logit_lens_only", "LL Readable, Not Steerable"),
        ("tuned_lens_and_steerable", "TL Only + Steerable  [H1]"),
        ("tuned_lens_only", "TL Only, Not Steerable"),
        ("steerable_not_decodable", "Steerable, Not Decodable [H2]"),
        ("both_fail", "Both Fail"),
    ]

    for cell_key, label in cell_labels:
        entries = matrix.get(cell_key, [])
        task_names = [e["task"] for e in entries]
        print(f"  {label:<33} {len(entries):<5} "
              f"{', '.join(task_names) if task_names else '(empty)'}")

    gap_closed = summary.get("gap_closed_tasks", [])
    gap_persists = summary.get("gap_persists_tasks", [])

    print(f"\nHypothesis Assessment:")
    if gap_closed:
        print(f"  H1 (Gap-Closing): SUPPORTED -- tuned lens decodes "
              f"{len(gap_closed)} tasks: {', '.join(gap_closed)}")
    else:
        print(f"  H1 (Gap-Closing): NOT SUPPORTED -- "
              f"no new decodable tasks found")

    if gap_persists and not gap_closed:
        print(f"  H2 (Confirmation): SUPPORTED -- {len(gap_persists)} tasks "
              f"remain steerable but not decodable")
    elif gap_persists:
        print(f"  H2 (Partial): {len(gap_persists)} tasks still non-decodable, "
              f"but H1 fires for {len(gap_closed)} tasks")
    print()


def _print_tuned_lens_summary(
    readability: List[TunedLensResult],
    fv_vocab: List[TunedLensFVVocabResult],
):
    """Print tuned lens readability summary."""
    print("\n" + "=" * 70)
    print("TUNED LENS READABILITY (best layer per task x template)")
    print("=" * 70)
    print(f"{'Task':<20} {'Template':<10} {'Layer':<8} "
          f"{'Top-1':<8} {'Top-5':<8} {'Top-10':<8} {'MeanRank':<10}")
    print("-" * 75)

    by_tt: Dict[Tuple[str, str], List[TunedLensResult]] = {}
    for r in readability:
        by_tt.setdefault((r.task, r.template_id), []).append(r)

    for (task, tid), rs in sorted(by_tt.items()):
        best = max(rs, key=lambda x: x.top_10_accuracy)
        print(f"{task:<20} {tid:<10} L{best.layer:<7} "
              f"{best.top_1_accuracy:<8.3f} {best.top_5_accuracy:<8.3f} "
              f"{best.top_10_accuracy:<8.3f} {best.mean_correct_rank:<10.1f}")

    if fv_vocab:
        print(f"\nTuned Lens FV Vocab Projection:")
        print(f"  Mean correct output fraction: "
              f"{np.mean([v.correct_output_fraction for v in fv_vocab]):.3f}")
        print(f"  Mean task relevant fraction: "
              f"{np.mean([v.task_relevant_fraction for v in fv_vocab]):.3f}")
    print()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_tuned_lens_analysis(
    model: ModelWrapper,
    config: ExperimentConfig,
    fvs: FVCollection,
    iid_summaries: List[IIDSummary],
    rank: int = DEFAULT_RANK,
) -> Dict[str, Any]:
    """
    Run the full tuned lens analysis pipeline.

    1. Train (or load cached) per-layer translators
    2. Tuned lens readability evaluation
    3. Tuned lens FV vocabulary projection
    4. Expanded 2×3 dissociation matrix (vs logit lens + FV steering)
    5. Layer-shift analysis (H3)
    6. FV coherence comparison (H4)

    Requires Stages 2 (extract) and 3 (probe_activations) to have completed.
    """
    results: Dict[str, Any] = {}
    tasks = get_tasks(config.task_names)
    layers = model.spec.extraction_layers()
    batch_size = getattr(
        config.ops, "tuned_lens_batch_size",
        config.ops.extraction_batch_size,
    )

    # -------------------------------------------------------------------
    # 1. Train or load translators
    # -------------------------------------------------------------------
    cached = load_translators(
        config.cache_dir, config.model_key, config.config_hash(),
        rank, model.d_model, config.ops.device,
    )

    if cached is not None:
        translators = cached
        results["training_log"] = {"status": "loaded_from_cache"}
    else:
        logger.info("Training tuned lens translators (rank=%d)...", rank)
        trainer = TunedLensTrainer(config, rank=rank)
        translators, training_log = trainer.train_all_translators(model.spec)

        save_translators(
            translators, training_log, config.cache_dir,
            config.model_key, config.config_hash(), rank,
        )
        results["training_log"] = training_log

    # -------------------------------------------------------------------
    # 2. Load cached activations (same as readability stage)
    # -------------------------------------------------------------------
    probe_cache_dir = config.cache_dir / "activations_probe"
    cached_by_tt: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}

    for task_name, task_spec in tasks.items():
        for tid in task_spec.template_ids:
            cache_path = probe_cache_dir / f"{task_name}_{tid}.pt"
            if cache_path.exists():
                cached_by_tt[(task_name, tid)] = torch.load(
                    cache_path, map_location="cpu", weights_only=True,
                )

    if not cached_by_tt:
        logger.warning(
            "No cached activations in %s -- "
            "run stage 'probe_activations' first.",
            probe_cache_dir,
        )
        return results

    logger.info(
        "Loaded %d cached activation files for tuned lens analysis",
        len(cached_by_tt),
    )

    # -------------------------------------------------------------------
    # 3. Tuned Lens Readability
    # -------------------------------------------------------------------
    logger.info("Running tuned-lens readability analysis...")
    analyzer = TunedLensAnalyzer(model, config, translators)
    all_readability: List[TunedLensResult] = []

    for task_name, task_spec in tqdm(
        tasks.items(), desc="Tuned Lens Readability", unit="task",
    ):
        for tid in task_spec.template_ids:
            acts = cached_by_tt.get((task_name, tid))
            if acts is None:
                continue
            try:
                task_results = analyzer.analyze_task_template(
                    task_spec, tid, layers, acts, batch_size=batch_size,
                )
                all_readability.extend(task_results)
            except Exception as e:
                logger.error(
                    "Tuned lens readability failed for %s/%s: %s",
                    task_name, tid, e,
                )

    results["readability"] = [asdict(r) for r in all_readability]

    # -------------------------------------------------------------------
    # 4. Tuned Lens FV Projection
    # -------------------------------------------------------------------
    logger.info("Running tuned-lens FV vocabulary projection...")
    projector = TunedLensFVProjector(model, config, translators)

    best_layer_lookup: Dict[Tuple[str, str], int] = {}
    for s in iid_summaries:
        best_layer_lookup[(s.task, s.template_id)] = s.best_layer

    all_fv_vocab: List[TunedLensFVVocabResult] = []
    for task_name, task_spec in tqdm(
        tasks.items(), desc="Tuned Lens FV Proj", unit="task",
    ):
        if task_name not in fvs:
            continue
        for tid in task_spec.template_ids:
            if tid not in fvs[task_name]:
                continue
            best_layer = best_layer_lookup.get(
                (task_name, tid), layers[len(layers) // 2],
            )
            if best_layer in fvs[task_name][tid]:
                fv_obj = fvs[task_name][tid][best_layer]
                try:
                    vr = projector.project_fv(fv_obj, task_spec)
                    all_fv_vocab.append(vr)
                except Exception as e:
                    logger.error(
                        "Tuned FV projection failed for %s/%s/L%d: %s",
                        task_name, tid, best_layer, e,
                    )

    # Truncate token lists for JSON
    vocab_for_json = []
    for vr in all_fv_vocab:
        d = asdict(vr)
        d["top_positive_tokens"] = d["top_positive_tokens"][:10]
        d["top_positive_logits"] = d["top_positive_logits"][:10]
        d["top_negative_tokens"] = d["top_negative_tokens"][:10]
        d["top_negative_logits"] = d["top_negative_logits"][:10]
        vocab_for_json.append(d)
    results["fv_vocab_projection"] = vocab_for_json

    # -------------------------------------------------------------------
    # 5. Expanded 2×3 Dissociation Matrix
    # -------------------------------------------------------------------
    logit_lens_path = config.results_dir / "readability_results.json"
    ll_readability: List[Dict] = []
    ll_fv_vocab: List[Dict] = []
    if logit_lens_path.exists():
        with open(logit_lens_path) as f:
            ll_data = json.load(f)
        ll_readability = ll_data.get("readability", [])
        ll_fv_vocab = ll_data.get("fv_vocab_projection", [])

    if ll_readability:
        logger.info("Computing expanded 2x3 dissociation matrix...")
        expanded_matrix = compute_expanded_dissociation_matrix(
            ll_readability, all_readability, iid_summaries,
            readability_threshold=config.science.readability_threshold,
        )
        results["expanded_dissociation_matrix"] = expanded_matrix
        _print_expanded_matrix(expanded_matrix)
    else:
        logger.warning(
            "No logit lens results at %s -- "
            "cannot compute expanded matrix. Run 'readability' stage first.",
            logit_lens_path,
        )

    # -------------------------------------------------------------------
    # 6. Layer-Shift Analysis (H3)
    # -------------------------------------------------------------------
    if ll_readability:
        layer_shift = _compute_layer_shift(ll_readability, all_readability)
        results["layer_shift_analysis"] = layer_shift

    # -------------------------------------------------------------------
    # 7. FV Coherence Comparison (H4)
    # -------------------------------------------------------------------
    if ll_fv_vocab and vocab_for_json:
        fv_coherence = _compute_fv_coherence_comparison(
            ll_fv_vocab, vocab_for_json,
        )
        results["fv_coherence_comparison"] = fv_coherence

    # -------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------
    _print_tuned_lens_summary(all_readability, all_fv_vocab)

    # Cleanup
    del cached_by_tt
    gc.collect()

    return results


def save_tuned_lens_results(results: Dict, output_dir: Path):
    """Save tuned lens analysis results to JSON."""
    path = output_dir / "tuned_lens_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Tuned lens results saved to %s", path)
