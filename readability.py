"""
Readability Analysis: Logit Lens + FV Vocabulary Projection
============================================================
Replaces the broken linear probing stage (Stage 7) with two parameter-free
analyses that use the model's own unembedding matrix:

1. **Logit Lens Readability** — Projects zero-shot residual-stream activations
   through ``ln_final @ W_U`` to measure whether the model already encodes the
   correct output at each layer, *without* any ICL demonstrations or steering.
   This is the "readability" axis of the 2×2 matrix.

2. **FV Vocabulary Projection** — Projects each extracted function vector
   through the same unembedding pipeline to reveal *what the FV points toward*
   in token space.  For successful FVs we expect correct output tokens;
   for failing FVs we expect task-relevant but functionally insufficient tokens.

Both analyses are entirely GPU-batched, require no learned parameters, and
reuse cached activations from Stage 3 and cached FVs from Stage 2.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import wilcoxon
from tqdm import tqdm

from .config import ExperimentConfig
from .extraction import FVCollection, FunctionVector
from .models import ModelWrapper
from .steering import IIDSummary
from .tasks import TaskSpec, get_tasks, TaskCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LogitLensResult:
    """Logit-lens readability result for one (task, template, layer)."""
    task: str
    template_id: str
    layer: int          # 1-indexed (matches FV layer convention)
    # Fraction of examples where correct output token is in top-k
    top_1_accuracy: float
    top_5_accuracy: float
    top_10_accuracy: float
    # Mean rank of the correct token (lower = more readable)
    mean_correct_rank: float
    # Mean log-probability of the correct token
    mean_correct_logprob: float
    n_examples: int


@dataclass
class SentimentPolarityResult:
    """Special readability result for sentiment_flip using polarity scoring."""
    task: str
    template_id: str
    layer: int
    # Mean cosine similarity between zero-shot activation and sentiment
    # contrast vector (mean_positive - mean_negative output embeddings)
    mean_polarity_score: float
    # Fraction of examples where polarity sign matches expected output sentiment
    polarity_classification_accuracy: float
    n_examples: int


@dataclass
class FVVocabResult:
    """FV vocabulary projection result for one (task, template, layer)."""
    task: str
    template_id: str
    layer: int
    # Top-N tokens by logit magnitude (positive direction)
    top_positive_tokens: List[str]
    top_positive_logits: List[float]
    # Top-N tokens by logit magnitude (negative direction)
    top_negative_tokens: List[str]
    top_negative_logits: List[float]
    # What fraction of top-N positive tokens are correct outputs
    correct_output_fraction: float
    # What fraction are task-relevant (broader category)
    task_relevant_fraction: float
    fv_norm: float


@dataclass
class PostSteeringLogitLensResult:
    """Post-steering logit lens result for one (task, template)."""
    task: str
    template_id: str
    injection_layer: int        # 1-indexed layer where FV is injected
    injection_strength: float
    n_examples: int
    # Per-layer results: layer_1idx -> {zero_shot_top10, post_steering_top10, delta}
    per_layer: Dict[int, Dict[str, float]]
    # Aggregate metrics
    max_post_steering_top10: float
    max_delta: float
    answer_emerges_at_layer: Optional[int]  # 1-indexed, or None
    zero_shot_max_top10: float
    zero_shot_peak_layer: int               # 1-indexed
    post_steering_peak_layer: int           # 1-indexed
    peak_layer_delta: int                   # positive = FV makes info readable earlier
    random_control_max_top10: float
    # Classification: absent, partial, emerged, amplified
    category: str


# ---------------------------------------------------------------------------
# 1. Logit Lens Analyzer
# ---------------------------------------------------------------------------

class LogitLensAnalyzer:
    """
    Project zero-shot activations through the model's own decoding pipeline
    to measure per-layer readability without any learned parameters.

    Uses cached activations from Stage 3 (``cache/activations_probe/``).
    """

    def __init__(self, model: ModelWrapper, config: ExperimentConfig):
        self.model = model
        self.config = config
        self.top_k_values = config.science.readability_top_k
        self.device = config.ops.device

        # Extract unembedding components from the HookedTransformer.
        # W_U shape: [d_model, d_vocab]
        # ln_final: the final LayerNorm before unembedding
        self._W_U = model.model.W_U.detach()           # [d_model, vocab]
        self._ln_final = model.model.ln_final

        # If the model has an unembed bias, capture it
        if hasattr(model.model, 'b_U') and model.model.b_U is not None:
            self._b_U = model.model.b_U.detach()        # [vocab]
        else:
            self._b_U = None

    @torch.no_grad()
    def _project_to_logits(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Project activations through ln_final and W_U to get logits.

        Args:
            activations: [batch, d_model] on any device

        Returns:
            logits: [batch, vocab_size] on same device as activations
        """
        device = activations.device
        h = activations.to(self._W_U.device)
        h_norm = self._ln_final(h)
        logits = h_norm @ self._W_U
        if self._b_U is not None:
            logits = logits + self._b_U
        return logits.to(device)

    @torch.no_grad()
    def analyze_task_template(
        self,
        task_spec: TaskSpec,
        template_id: str,
        layers_1idx: List[int],
        cached_activations: Dict[int, torch.Tensor],
        batch_size: int = 256,
    ) -> List[LogitLensResult]:
        """
        Run logit-lens analysis for one (task, template) across all layers.

        Args:
            task_spec: The task specification with pairs
            template_id: Which template
            layers_1idx: 1-indexed layers to analyze
            cached_activations: Dict[layer_0idx] -> [n_examples, d_model]
            batch_size: GPU batch size for matmul (tune to GPU profile)

        Returns:
            List of LogitLensResult, one per layer
        """
        tokenizer = self.model.tokenizer
        pairs = task_spec.pairs

        # Tokenize correct output tokens (first token of expected output)
        correct_token_ids = []
        for _, expected_output in pairs:
            tokens = tokenizer.encode(expected_output, add_special_tokens=False)
            if tokens:
                correct_token_ids.append(tokens[0])
            else:
                # Fallback: try with space prefix (some tokenizers need this)
                tokens = tokenizer.encode(" " + expected_output, add_special_tokens=False)
                correct_token_ids.append(tokens[0] if tokens else -1)

        correct_ids = torch.tensor(correct_token_ids, dtype=torch.long)
        max_k = max(self.top_k_values)
        results = []

        for layer_1idx in layers_1idx:
            layer_0idx = layer_1idx - 1
            if layer_0idx not in cached_activations:
                continue

            acts = cached_activations[layer_0idx]  # [n_examples, d_model]
            n_examples = acts.shape[0]

            # Limit to number of pairs (cached acts may include extra)
            n_use = min(n_examples, len(pairs))
            acts = acts[:n_use]
            cids = correct_ids[:n_use]

            # Batch the matmul for GPU efficiency
            all_ranks = []
            all_logprobs = []
            top_k_hits = {k: 0 for k in self.top_k_values}

            for i in range(0, n_use, batch_size):
                batch_acts = acts[i:i + batch_size]
                batch_cids = cids[i:i + batch_size]

                logits = self._project_to_logits(batch_acts)  # [batch, vocab]

                # Compute ranks of correct tokens
                # Sort descending; rank = position of correct token
                sorted_indices = logits.argsort(dim=-1, descending=True)

                for j in range(batch_acts.shape[0]):
                    cid = batch_cids[j].item()
                    if cid < 0:
                        # Unknown token, skip
                        all_ranks.append(logits.shape[-1])
                        all_logprobs.append(-float('inf'))
                        continue

                    # Rank: position in sorted order (0-indexed)
                    rank_mask = (sorted_indices[j] == cid).nonzero(as_tuple=True)[0]
                    rank = rank_mask[0].item() if len(rank_mask) > 0 else logits.shape[-1]
                    all_ranks.append(rank)

                    # Top-k membership
                    for k in self.top_k_values:
                        if rank < k:
                            top_k_hits[k] += 1

                    # Log-probability
                    log_probs = torch.log_softmax(logits[j], dim=-1)
                    all_logprobs.append(log_probs[cid].item())

            valid_count = sum(1 for r in all_ranks if r < logits.shape[-1])
            if valid_count == 0:
                valid_count = 1  # avoid division by zero

            results.append(LogitLensResult(
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
# 2. Sentiment Polarity Analyzer (special handling for sentiment_flip)
# ---------------------------------------------------------------------------

class SentimentPolarityAnalyzer:
    """
    For sentiment_flip, first-token readability is meaningless (it's always "I").
    Instead, measure whether the zero-shot activation has a component along the
    sentiment polarity axis: mean(positive_output_embeddings) - mean(negative_output_embeddings).

    This tests whether the model *encodes polarity*, which is the actual
    readability question for sentiment tasks.
    """

    def __init__(self, model: ModelWrapper, config: ExperimentConfig):
        self.model = model
        self.config = config
        self._W_U = model.model.W_U.detach()
        self._ln_final = model.model.ln_final

    @torch.no_grad()
    def _build_sentiment_contrast_vector(
        self, task_spec: TaskSpec
    ) -> torch.Tensor:
        """
        Build a sentiment contrast vector from the task's own pairs.

        For sentiment_flip: inputs are positive phrases, outputs are negative.
        Contrast = mean(embed(negative_outputs)) - mean(embed(positive_inputs)).
        We use the model's embedding matrix to get token-level embeddings,
        then average over the first token of each phrase.
        """
        tokenizer = self.model.tokenizer
        W_E = self.model.model.W_E.detach()  # [vocab, d_model]

        pos_embeds = []
        neg_embeds = []
        for inp, out in task_spec.pairs:
            # Input = positive phrase, Output = negative phrase
            pos_tokens = tokenizer.encode(inp, add_special_tokens=False)
            neg_tokens = tokenizer.encode(out, add_special_tokens=False)
            if pos_tokens:
                pos_embeds.append(W_E[pos_tokens[0]])
            if neg_tokens:
                neg_embeds.append(W_E[neg_tokens[0]])

        if not pos_embeds or not neg_embeds:
            # Fallback: return zero vector
            return torch.zeros(self.model.d_model, device=W_E.device)

        pos_mean = torch.stack(pos_embeds).mean(dim=0)
        neg_mean = torch.stack(neg_embeds).mean(dim=0)
        contrast = neg_mean - pos_mean
        # Normalize to unit vector
        norm = contrast.norm()
        if norm > 0:
            contrast = contrast / norm
        return contrast

    @torch.no_grad()
    def analyze_task_template(
        self,
        task_spec: TaskSpec,
        template_id: str,
        layers_1idx: List[int],
        cached_activations: Dict[int, torch.Tensor],
    ) -> List[SentimentPolarityResult]:
        """Measure sentiment polarity readability at each layer."""
        contrast = self._build_sentiment_contrast_vector(task_spec)
        results = []

        for layer_1idx in layers_1idx:
            layer_0idx = layer_1idx - 1
            if layer_0idx not in cached_activations:
                continue

            acts = cached_activations[layer_0idx]  # [n_examples, d_model]
            n_use = min(acts.shape[0], len(task_spec.pairs))
            acts = acts[:n_use]

            # Normalize activations through ln_final before projection
            h_norm = self._ln_final(acts.to(self._W_U.device))

            # Cosine similarity with the sentiment contrast vector
            contrast_expanded = contrast.unsqueeze(0)  # [1, d_model]
            # Cosine sim = dot(h_norm, contrast) / (||h_norm|| * ||contrast||)
            dots = (h_norm * contrast_expanded).sum(dim=-1)  # [n_examples]
            norms = h_norm.norm(dim=-1)
            contrast_norm = contrast.norm()
            if contrast_norm > 0:
                cosine_sims = dots / (norms * contrast_norm + 1e-8)
            else:
                cosine_sims = torch.zeros(n_use, device=h_norm.device)

            # For sentiment_flip: inputs are positive, outputs are negative.
            # If polarity is encoded, negative-output examples should have
            # negative cosine with the (neg - pos) direction... actually
            # the sign interpretation depends on the layer's representation.
            # We use absolute cosine as the polarity score, and classify
            # based on sign: positive cosine = "negative sentiment detected"
            # (since contrast = neg - pos).
            polarity_scores = cosine_sims.float().cpu().numpy()

            # Classification: if cosine > 0, model "detects" negative sentiment
            # For sentiment_flip, the output IS negative, so positive cosine = correct
            n_correct = int((polarity_scores > 0).sum())

            results.append(SentimentPolarityResult(
                task=task_spec.name,
                template_id=template_id,
                layer=layer_1idx,
                mean_polarity_score=float(np.mean(np.abs(polarity_scores))),
                polarity_classification_accuracy=n_correct / max(n_use, 1),
                n_examples=n_use,
            ))

        return results


# ---------------------------------------------------------------------------
# 3. FV Vocabulary Projector
# ---------------------------------------------------------------------------

class FVVocabProjector:
    """
    Project function vectors through the model's unembedding matrix to reveal
    what each FV "points toward" in token space.

    For successful FVs: top tokens should be correct outputs.
    For failing FVs: top tokens may be task-relevant but not correct outputs.
    """

    def __init__(self, model: ModelWrapper, config: ExperimentConfig):
        self.model = model
        self.config = config
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
    ) -> FVVocabResult:
        """
        Project a single FV through the unembedding pipeline.

        Args:
            fv: The function vector to project
            task_spec: Task spec (for checking output token membership)

        Returns:
            FVVocabResult with top positive/negative tokens and overlap stats
        """
        tokenizer = self.model.tokenizer
        vec = fv.vector.to(self._W_U.device)

        # Apply ln_final then W_U
        # Note: ln_final expects [batch, d_model], so unsqueeze
        h_norm = self._ln_final(vec.unsqueeze(0))  # [1, d_model]
        logits = (h_norm @ self._W_U).squeeze(0)   # [vocab]
        if self._b_U is not None:
            logits = logits + self._b_U

        # Top positive tokens (FV pushes toward these)
        top_pos_idx = logits.argsort(descending=True)[:self.top_n]
        top_pos_logits = logits[top_pos_idx]
        top_pos_tokens = [
            tokenizer.decode([idx.item()]).strip()
            for idx in top_pos_idx
        ]

        # Top negative tokens (FV pushes away from these)
        top_neg_idx = logits.argsort(descending=False)[:self.top_n]
        top_neg_logits = logits[top_neg_idx]
        top_neg_tokens = [
            tokenizer.decode([idx.item()]).strip()
            for idx in top_neg_idx
        ]

        # Check overlap with correct outputs
        correct_outputs = {out.strip().lower() for _, out in task_spec.pairs}
        # Also tokenize first tokens of outputs for token-level matching
        correct_first_tokens = set()
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

        # Task-relevant: broader check (token text appears as substring
        # in any output, or output appears as substring in token text)
        n_relevant = 0
        for tok_str in top_pos_tokens:
            tok_lower = tok_str.lower().strip()
            if not tok_lower:
                continue
            for correct in correct_outputs:
                if tok_lower in correct or correct in tok_lower:
                    n_relevant += 1
                    break

        return FVVocabResult(
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

    def project_all_fvs(
        self,
        fvs: FVCollection,
        task_names: Optional[List[str]] = None,
    ) -> List[FVVocabResult]:
        """Project all FVs at their best steering layer."""
        tasks = get_tasks(task_names)
        results = []
        for task_name, task_spec in tasks.items():
            if task_name not in fvs:
                continue
            for tid in task_spec.template_ids:
                if tid not in fvs[task_name]:
                    continue
                # Project at all available layers
                for layer, fv_obj in fvs[task_name][tid].items():
                    results.append(self.project_fv(fv_obj, task_spec))
        return results


# ---------------------------------------------------------------------------
# 3b. Post-Steering Logit Lens Analyzer
# ---------------------------------------------------------------------------

class PostSteeringLogitLensAnalyzer:
    """
    Run logit lens on activations AFTER FV injection to test whether
    the FV creates new decodable information at downstream layers.

    For each (task, template), injects the FV at its best layer/strength,
    captures residual stream activations at all downstream layers, and
    projects them through ln_final @ W_U to measure top-k accuracy.

    Also runs a random-vector control of the same norm for comparison.
    """

    def __init__(self, model: ModelWrapper, config: ExperimentConfig):
        self.model = model
        self.config = config
        self.top_k_values = config.science.readability_top_k
        self.device = config.ops.device
        self.readability_threshold = config.science.readability_threshold

        # Reuse unembedding components
        self._W_U = model.model.W_U.detach()
        self._ln_final = model.model.ln_final
        if hasattr(model.model, 'b_U') and model.model.b_U is not None:
            self._b_U = model.model.b_U.detach()
        else:
            self._b_U = None

    @torch.no_grad()
    def _project_to_logits(self, activations: torch.Tensor) -> torch.Tensor:
        """Project activations through ln_final and W_U to get logits."""
        device = activations.device
        h = activations.to(self._W_U.device)
        h_norm = self._ln_final(h)
        logits = h_norm @ self._W_U
        if self._b_U is not None:
            logits = logits + self._b_U
        return logits.to(device)

    @torch.no_grad()
    def _compute_top10_accuracy(
        self,
        activations: torch.Tensor,
        correct_ids: torch.Tensor,
        batch_size: int = 128,
    ) -> float:
        """Compute top-10 accuracy for a set of activations."""
        n = activations.shape[0]
        hits = 0
        for i in range(0, n, batch_size):
            batch_acts = activations[i:i + batch_size]
            batch_cids = correct_ids[i:i + batch_size]
            logits = self._project_to_logits(batch_acts)
            top10 = logits.argsort(dim=-1, descending=True)[:, :10]
            for j in range(batch_acts.shape[0]):
                cid = batch_cids[j].item()
                if cid >= 0 and cid in top10[j]:
                    hits += 1
        return hits / max(n, 1)

    @torch.no_grad()
    def _run_with_steering_and_cache(
        self,
        prompts: List[str],
        fv_tensor: torch.Tensor,
        fv_layer_0idx: int,
        strength: float,
        capture_layers_0idx: List[int],
        batch_size: int,
    ) -> Dict[int, torch.Tensor]:
        """
        Run forward pass with FV steering hook, capturing residual stream
        activations at specified layers.

        Returns:
            Dict[layer_0idx] -> Tensor[n_prompts, d_model] (last-token activations)
        """
        hook_name = self.model.resid_post_hook(fv_layer_0idx)
        capture_hooks = [
            f"blocks.{l}.hook_resid_post" for l in capture_layers_0idx
        ]

        def steering_hook(acts, hook):
            acts[:, -1, :] += strength * fv_tensor
            return acts

        # Accumulate per-layer results across batches
        layer_acts: Dict[int, List[torch.Tensor]] = {
            l: [] for l in capture_layers_0idx
        }

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tokens = self.model.model.to_tokens(batch, prepend_bos=True)

            with self.model.model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                _, cache = self.model.model.run_with_cache(
                    tokens.to(self.device),
                    names_filter=capture_hooks,
                    return_type="logits",
                )

            for l in capture_layers_0idx:
                cache_key = f"blocks.{l}.hook_resid_post"
                if cache_key in cache:
                    layer_acts[l].append(cache[cache_key][:, -1, :].cpu())

            del cache
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Concatenate batches
        result = {}
        for l in capture_layers_0idx:
            if layer_acts[l]:
                result[l] = torch.cat(layer_acts[l], dim=0)
        return result

    @torch.no_grad()
    def analyze_task_template(
        self,
        task_spec: TaskSpec,
        template_id: str,
        fv_tensor: torch.Tensor,
        fv_layer_0idx: int,
        strength: float,
        zero_shot_top10_by_layer: Dict[int, float],
        is_both_succeed: bool = False,
        batch_size: int = 64,
    ) -> Optional[PostSteeringLogitLensResult]:
        """
        Run post-steering logit lens for one (task, template).

        Args:
            task_spec: Task specification
            template_id: Template ID
            fv_tensor: Function vector tensor on device [d_model]
            fv_layer_0idx: 0-indexed injection layer
            strength: Steering strength
            zero_shot_top10_by_layer: Dict[layer_1idx] -> zero-shot top-10 acc
            is_both_succeed: Whether this is a "both_succeed" control case
            batch_size: GPU batch size for forward passes

        Returns:
            PostSteeringLogitLensResult, or None on failure
        """
        tokenizer = self.model.tokenizer
        template_str = task_spec.templates[template_id]

        # Generate prompts matching Stage 3 format (bare zero-shot prompts)
        prompts = [
            template_str.replace("{X}", inp)
            for inp, _ in task_spec.pairs
        ]

        # Tokenize correct output (first token)
        correct_token_ids = []
        for _, expected_output in task_spec.pairs:
            tokens = tokenizer.encode(expected_output, add_special_tokens=False)
            if tokens:
                correct_token_ids.append(tokens[0])
            else:
                tokens = tokenizer.encode(
                    " " + expected_output, add_special_tokens=False
                )
                correct_token_ids.append(tokens[0] if tokens else -1)
        correct_ids = torch.tensor(correct_token_ids, dtype=torch.long)

        n_examples = len(prompts)

        # Layers to capture: injection layer through final layer
        n_layers = self.model.n_layers
        capture_layers_0idx = list(range(fv_layer_0idx, n_layers))
        fv_layer_1idx = fv_layer_0idx + 1

        # --- Run with real FV ---
        steered_acts = self._run_with_steering_and_cache(
            prompts, fv_tensor, fv_layer_0idx, strength,
            capture_layers_0idx, batch_size,
        )

        # --- Run with random control vector (same norm, fixed seed) ---
        rng = torch.Generator()
        rng.manual_seed(42)
        random_fv = torch.randn(
            fv_tensor.shape, generator=rng,
            device=fv_tensor.device, dtype=fv_tensor.dtype,
        )
        random_fv = random_fv * (fv_tensor.norm() / random_fv.norm())

        random_acts = self._run_with_steering_and_cache(
            prompts, random_fv, fv_layer_0idx, strength,
            capture_layers_0idx, batch_size,
        )

        # --- Compute per-layer top-10 accuracy ---
        per_layer: Dict[int, Dict[str, float]] = {}
        max_ps_top10 = 0.0
        max_delta = -float('inf')
        max_random_top10 = 0.0
        ps_peak_layer_1idx = fv_layer_1idx
        zs_peak_layer_1idx = fv_layer_1idx
        zs_peak_val = 0.0
        ps_peak_val = 0.0
        answer_emerges_at = None

        for l0 in capture_layers_0idx:
            l1 = l0 + 1  # 1-indexed
            zs_top10 = zero_shot_top10_by_layer.get(l1, 0.0)

            if l0 in steered_acts:
                ps_top10 = self._compute_top10_accuracy(
                    steered_acts[l0], correct_ids,
                )
            else:
                ps_top10 = 0.0

            if l0 in random_acts:
                rand_top10 = self._compute_top10_accuracy(
                    random_acts[l0], correct_ids,
                )
            else:
                rand_top10 = 0.0

            delta = ps_top10 - zs_top10
            per_layer[l1] = {
                "zero_shot_top10": round(zs_top10, 4),
                "post_steering_top10": round(ps_top10, 4),
                "delta": round(delta, 4),
            }

            if ps_top10 > max_ps_top10:
                max_ps_top10 = ps_top10
                ps_peak_layer_1idx = l1
            if ps_top10 > ps_peak_val:
                ps_peak_val = ps_top10

            if delta > max_delta:
                max_delta = delta

            if rand_top10 > max_random_top10:
                max_random_top10 = rand_top10

            if zs_top10 > zs_peak_val:
                zs_peak_val = zs_top10
                zs_peak_layer_1idx = l1

            if answer_emerges_at is None and ps_top10 > self.readability_threshold:
                answer_emerges_at = l1

        # Also check zero-shot layers before injection point for peak
        for l1, zs_val in zero_shot_top10_by_layer.items():
            if zs_val > zs_peak_val:
                zs_peak_val = zs_val
                zs_peak_layer_1idx = l1

        peak_layer_delta = zs_peak_layer_1idx - ps_peak_layer_1idx

        # --- Classify result ---
        # Use best IID steering accuracy for comparison (not passed directly,
        # but we can approximate with the threshold logic from spec)
        if is_both_succeed:
            category = "amplified" if peak_layer_delta > 0 else "unchanged"
        elif max_ps_top10 < self.readability_threshold:
            # Check if random control is comparable
            category = "absent"
        elif max_ps_top10 < 0.30:
            category = "partial"
        else:
            category = "emerged"

        # Clean up GPU
        del steered_acts, random_acts
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return PostSteeringLogitLensResult(
            task=task_spec.name,
            template_id=template_id,
            injection_layer=fv_layer_1idx,
            injection_strength=strength,
            n_examples=n_examples,
            per_layer=per_layer,
            max_post_steering_top10=round(max_ps_top10, 4),
            max_delta=round(max_delta, 4),
            answer_emerges_at_layer=answer_emerges_at,
            zero_shot_max_top10=round(zs_peak_val, 4),
            zero_shot_peak_layer=zs_peak_layer_1idx,
            post_steering_peak_layer=ps_peak_layer_1idx,
            peak_layer_delta=peak_layer_delta,
            random_control_max_top10=round(max_random_top10, 4),
            category=category,
        )


# ---------------------------------------------------------------------------
# 4. Gap & Statistical Analyses (updated for readability results)
# ---------------------------------------------------------------------------

def compute_readability_steerability_gap(
    readability_results: List[LogitLensResult],
    iid_summaries: List[IIDSummary],
    sentiment_results: Optional[List[SentimentPolarityResult]] = None,
    readability_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Compute readability-steerability gap per task for the 2×2 matrix.

    Readability = best logit-lens top-10 accuracy across templates/layers.
    Steerability = best IID steering accuracy across templates.

    For sentiment_flip: uses polarity classification accuracy if available.
    """
    # Best readability per task
    read_by_task: Dict[str, List[float]] = {}
    for r in readability_results:
        read_by_task.setdefault(r.task, []).append(r.top_10_accuracy)

    # Incorporate sentiment polarity results for sentiment_flip
    if sentiment_results:
        for sr in sentiment_results:
            read_by_task.setdefault(sr.task, []).append(
                sr.polarity_classification_accuracy
            )

    # Best steering accuracy per task
    steer_by_task: Dict[str, List[float]] = {}
    for s in iid_summaries:
        steer_by_task.setdefault(s.task, []).append(s.best_accuracy)

    gaps: Dict[str, Any] = {}
    for task in set(list(read_by_task.keys()) + list(steer_by_task.keys())):
        best_read = max(read_by_task.get(task, [0.0]))
        best_steer = max(steer_by_task.get(task, [0.0]))
        gap = best_read - best_steer

        # Classify into 2×2 matrix cells
        readable = best_read > readability_threshold
        steerable = best_steer > readability_threshold
        if readable and steerable:
            cell = "both_succeed"
        elif readable and not steerable:
            cell = "readable_not_steerable"
        elif not readable and steerable:
            cell = "steerable_not_readable"
        else:
            cell = "both_fail"

        gaps[task] = {
            "best_readability": best_read,
            "best_steering_accuracy": best_steer,
            "gap": gap,
            "cell": cell,
            "readable_not_steerable": readable and not steerable,
        }

    return gaps


def wilcoxon_readability_vs_steering(
    readability_results: List[LogitLensResult],
    iid_summaries: List[IIDSummary],
) -> Dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test: readability vs FV steering accuracy.

    Pairs are matched on (task, template_id) using the best-layer value.
    """
    # Best readability per (task, template)
    read_lookup: Dict[Tuple[str, str], float] = {}
    for r in readability_results:
        key = (r.task, r.template_id)
        read_lookup[key] = max(read_lookup.get(key, 0.0), r.top_10_accuracy)

    # Best steering per (task, template)
    steer_lookup: Dict[Tuple[str, str], float] = {}
    for s in iid_summaries:
        key = (s.task, s.template_id)
        steer_lookup[key] = max(steer_lookup.get(key, 0.0), s.best_accuracy)

    # Group by task
    results_by_task: Dict[str, Dict[str, List[float]]] = {}
    for key in set(read_lookup.keys()) | set(steer_lookup.keys()):
        task = key[0]
        results_by_task.setdefault(task, {"read": [], "steer": []})
        results_by_task[task]["read"].append(read_lookup.get(key, 0.0))
        results_by_task[task]["steer"].append(steer_lookup.get(key, 0.0))

    test_results: Dict[str, Any] = {}
    n_tasks = len(results_by_task)

    for task, data in results_by_task.items():
        read_accs = np.array(data["read"])
        steer_accs = np.array(data["steer"])

        if len(read_accs) < 5:
            test_results[task] = {"error": "too few observations"}
            continue

        # Check if all differences are zero (Wilcoxon fails on constant data)
        diffs = read_accs - steer_accs
        if np.all(diffs == 0):
            test_results[task] = {
                "statistic": 0.0,
                "p_value": 1.0,
                "p_value_bonferroni": 1.0,
                "n_observations": len(read_accs),
                "mean_readability": float(np.mean(read_accs)),
                "mean_steer": float(np.mean(steer_accs)),
                "note": "all differences zero",
            }
            continue

        try:
            stat, p = wilcoxon(read_accs, steer_accs)
            p_corrected = min(p * n_tasks, 1.0)
            test_results[task] = {
                "statistic": float(stat),
                "p_value": float(p),
                "p_value_bonferroni": float(p_corrected),
                "n_observations": len(read_accs),
                "mean_readability": float(np.mean(read_accs)),
                "mean_steer": float(np.mean(steer_accs)),
            }
        except Exception as e:
            test_results[task] = {"error": str(e)}

    return test_results


# ---------------------------------------------------------------------------
# 4b. Steering Harm Analysis (zero-shot baseline vs FV steering)
# ---------------------------------------------------------------------------

@dataclass
class SteeringHarmResult:
    """Per-(task, template) comparison of zero-shot baseline vs FV steering."""
    task: str
    template_id: str
    zero_shot_accuracy: float
    few_shot_accuracy: float
    best_steering_accuracy: float
    # Positive = steering helps; negative = steering is destructive
    steering_delta: float
    # Is steering actively worse than doing nothing?
    is_destructive: bool
    # Relative to few-shot ICL: how much of the ICL gain does FV capture?
    icl_recovery_fraction: float
    # Layer that produced the best steering accuracy for this (task, template)
    best_layer: int
    # Layer that produced the WORST (most negative) delta vs zero-shot
    worst_layer: int
    worst_layer_accuracy: float
    worst_layer_delta: float


def compute_steering_harm(
    iid_summaries: List[IIDSummary],
    baseline_results: List[Dict],
    steering_results: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Compare FV steering accuracy against zero-shot baselines to detect
    cases where steering is actively destructive.

    A destructive FV is worse than useless — it provides false confidence
    while actually degrading model performance below the zero-shot level.
    This is a direct safety concern: a deployed steering intervention that
    makes the model *worse* at the task it's supposed to improve.

    When full per-layer steering_results are provided, also identifies which
    extraction layer produces the most destructive FVs.  If destructive cases
    cluster at specific layers, that's mechanistically informative — it
    suggests those layers produce FVs that are actively misaligned with the
    model's computation, not just uninformative.

    Args:
        iid_summaries: Best IID steering accuracy per (task, template)
        baseline_results: List of dicts with task, template_id, mode, accuracy
        steering_results: Optional full steering dict:
            steering_results[task][source_template][target_template][layer_str] = acc
            For IID, source == target.

    Returns:
        Dict with per-task harm analysis, layer clustering, and aggregate stats
    """
    # Build lookup: (task, template) -> zero_shot and few_shot accuracy
    zs_lookup: Dict[Tuple[str, str], float] = {}
    fs_lookup: Dict[Tuple[str, str], float] = {}
    for br in baseline_results:
        key = (br["task"], br["template_id"])
        if br["mode"] == "zero_shot":
            zs_lookup[key] = br["accuracy"]
        elif br["mode"] == "few_shot":
            fs_lookup[key] = br["accuracy"]

    # Build lookup: (task, template) -> best steering accuracy + best layer
    steer_lookup: Dict[Tuple[str, str], float] = {}
    best_layer_lookup: Dict[Tuple[str, str], int] = {}
    for s in iid_summaries:
        steer_lookup[(s.task, s.template_id)] = s.best_accuracy
        best_layer_lookup[(s.task, s.template_id)] = s.best_layer

    # Build per-layer IID steering lookup from full results
    # steering_results[task][tid][tid][layer_str] = accuracy  (IID = source==target)
    per_layer_iid: Dict[Tuple[str, str], Dict[int, float]] = {}
    if steering_results:
        for task, src_dict in steering_results.items():
            for tid, tgt_dict in src_dict.items():
                if tid in tgt_dict:  # IID: source == target
                    layer_accs = {}
                    for layer_str, acc in tgt_dict[tid].items():
                        try:
                            layer_accs[int(layer_str)] = float(acc)
                        except (ValueError, TypeError):
                            pass
                    if layer_accs:
                        per_layer_iid[(task, tid)] = layer_accs

    # Compute per-(task, template) harm
    per_pair: List[SteeringHarmResult] = []
    for key in set(steer_lookup.keys()) & set(zs_lookup.keys()):
        task, tid = key
        zs = zs_lookup[key]
        fs = fs_lookup.get(key, zs)
        steer = steer_lookup[key]
        delta = steer - zs

        # ICL recovery: what fraction of (few_shot - zero_shot) does steering recover?
        icl_gain = fs - zs
        if icl_gain > 0.01:
            recovery = delta / icl_gain
        else:
            recovery = 0.0 if abs(delta) < 0.01 else (1.0 if delta > 0 else -1.0)

        # Find the worst layer (most negative delta vs zero-shot)
        layer_accs = per_layer_iid.get(key, {})
        if layer_accs:
            worst_layer = min(layer_accs, key=lambda l: layer_accs[l] - zs)
            worst_layer_acc = layer_accs[worst_layer]
            worst_layer_delta = worst_layer_acc - zs
        else:
            worst_layer = best_layer_lookup.get(key, -1)
            worst_layer_acc = steer
            worst_layer_delta = delta

        per_pair.append(SteeringHarmResult(
            task=task,
            template_id=tid,
            zero_shot_accuracy=zs,
            few_shot_accuracy=fs,
            best_steering_accuracy=steer,
            steering_delta=delta,
            is_destructive=delta < -0.01,
            icl_recovery_fraction=recovery,
            best_layer=best_layer_lookup.get(key, -1),
            worst_layer=worst_layer,
            worst_layer_accuracy=worst_layer_acc,
            worst_layer_delta=worst_layer_delta,
        ))

    # Aggregate by task
    by_task: Dict[str, List[SteeringHarmResult]] = {}
    for r in per_pair:
        by_task.setdefault(r.task, []).append(r)

    task_summaries: Dict[str, Any] = {}
    total_destructive = 0
    total_pairs = len(per_pair)
    # Collect all worst layers from destructive cases for clustering analysis
    destructive_worst_layers: List[int] = []

    for task, results_list in sorted(by_task.items()):
        deltas = [r.steering_delta for r in results_list]
        n_destructive = sum(1 for r in results_list if r.is_destructive)
        total_destructive += n_destructive

        # Collect worst layers for destructive cases in this task
        task_destructive_layers = [
            r.worst_layer for r in results_list if r.is_destructive
        ]
        destructive_worst_layers.extend(task_destructive_layers)

        # Per-layer harm profile: at each layer, what's the mean delta across templates?
        layer_deltas: Dict[int, List[float]] = {}
        for r in results_list:
            layer_accs = per_layer_iid.get((r.task, r.template_id), {})
            zs = r.zero_shot_accuracy
            for layer, acc in layer_accs.items():
                layer_deltas.setdefault(layer, []).append(acc - zs)

        layer_harm_profile = {
            layer: {
                "mean_delta": float(np.mean(ds)),
                "n_destructive": int(sum(1 for d in ds if d < -0.01)),
                "n_templates": len(ds),
            }
            for layer, ds in sorted(layer_deltas.items())
        }

        task_summaries[task] = {
            "mean_zero_shot": float(np.mean([r.zero_shot_accuracy for r in results_list])),
            "mean_few_shot": float(np.mean([r.few_shot_accuracy for r in results_list])),
            "mean_steering": float(np.mean([r.best_steering_accuracy for r in results_list])),
            "mean_delta": float(np.mean(deltas)),
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
            "n_destructive": n_destructive,
            "n_templates": len(results_list),
            "destructive_rate": n_destructive / max(len(results_list), 1),
            "mean_icl_recovery": float(np.mean([r.icl_recovery_fraction for r in results_list])),
            "layer_harm_profile": layer_harm_profile,
            "per_template": [
                {
                    "template_id": r.template_id,
                    "zero_shot": r.zero_shot_accuracy,
                    "few_shot": r.few_shot_accuracy,
                    "steering": r.best_steering_accuracy,
                    "delta": r.steering_delta,
                    "destructive": r.is_destructive,
                    "icl_recovery": r.icl_recovery_fraction,
                    "best_layer": r.best_layer,
                    "worst_layer": r.worst_layer,
                    "worst_layer_delta": r.worst_layer_delta,
                }
                for r in sorted(results_list, key=lambda x: x.steering_delta)
            ],
        }

    # Layer clustering analysis: do destructive cases concentrate at specific layers?
    layer_cluster: Dict[str, Any] = {}
    if destructive_worst_layers:
        from collections import Counter
        layer_counts = Counter(destructive_worst_layers)
        most_common = layer_counts.most_common(5)
        layer_cluster = {
            "total_destructive_cases": len(destructive_worst_layers),
            "worst_layer_distribution": {
                str(layer): count for layer, count in most_common
            },
            "most_destructive_layer": most_common[0][0] if most_common else None,
            "most_destructive_layer_count": most_common[0][1] if most_common else 0,
            "concentration_ratio": (
                most_common[0][1] / len(destructive_worst_layers)
                if most_common else 0.0
            ),
        }

    return {
        "per_task": task_summaries,
        "aggregate": {
            "total_pairs": total_pairs,
            "total_destructive": total_destructive,
            "destructive_rate": total_destructive / max(total_pairs, 1),
            "mean_delta_all": float(np.mean([r.steering_delta for r in per_pair])) if per_pair else 0.0,
        },
        "destructive_layer_clustering": layer_cluster,
    }


# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------

def run_readability_analysis(
    model: ModelWrapper,
    config: ExperimentConfig,
    fvs: FVCollection,
    iid_summaries: List[IIDSummary],
    steering_results: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run the full readability analysis (replaces Stage 7 linear probing).

    1. Logit lens on cached zero-shot activations (all tasks)
    2. Sentiment polarity analysis (sentiment_flip only)
    3. FV vocabulary projection (all tasks, best layers)
    4. Readability-steerability gap computation
    5. Wilcoxon signed-rank test

    Args:
        model: Loaded ModelWrapper with TransformerLens model
        config: Experiment configuration
        fvs: Extracted function vectors from Stage 2
        iid_summaries: IID steering results for steerability comparison
        steering_results: Optional full steering dict for per-layer harm analysis.
            Format: steering_results[task][src_tid][tgt_tid][layer] = accuracy

    Returns:
        Dict with all results, ready for JSON serialization
    """
    results: Dict[str, Any] = {}
    tasks = get_tasks(config.task_names)
    layers = model.spec.extraction_layers()

    # Determine batch size from GPU profile
    batch_size = config.ops.extraction_batch_size

    # -------------------------
    # Load cached activations
    # -------------------------
    probe_cache_dir = config.cache_dir / "activations_probe"
    cached_by_task_template: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}

    n_cached = 0
    for task_name, task_spec in tasks.items():
        for tid in task_spec.template_ids:
            cache_path = probe_cache_dir / f"{task_name}_{tid}.pt"
            if cache_path.exists():
                cached_by_task_template[(task_name, tid)] = torch.load(
                    cache_path, map_location="cpu", weights_only=True,
                )
                n_cached += 1

    if n_cached > 0:
        logger.info(
            "Loaded %d cached activation files from Stage 3 for readability analysis",
            n_cached,
        )
    else:
        logger.warning(
            "No cached activations found in %s — "
            "run stage 'probe_activations' first.",
            probe_cache_dir,
        )
        return results

    # -------------------------
    # 1. Logit Lens Readability
    # -------------------------
    logger.info("Running logit-lens readability analysis...")
    analyzer = LogitLensAnalyzer(model, config)
    all_readability: List[LogitLensResult] = []

    for task_name, task_spec in tqdm(tasks.items(), desc="Logit Lens", unit="task"):
        for tid in task_spec.template_ids:
            acts = cached_by_task_template.get((task_name, tid))
            if acts is None:
                continue
            try:
                task_results = analyzer.analyze_task_template(
                    task_spec, tid, layers, acts, batch_size=batch_size,
                )
                all_readability.extend(task_results)
            except Exception as e:
                logger.error(
                    "Logit lens failed for %s/%s: %s", task_name, tid, e
                )

    results["readability"] = [asdict(r) for r in all_readability]

    # -------------------------
    # 2. Sentiment Polarity (sentiment_flip only)
    # -------------------------
    all_sentiment: List[SentimentPolarityResult] = []
    if "sentiment_flip" in tasks:
        logger.info("Running sentiment polarity analysis for sentiment_flip...")
        sentiment_analyzer = SentimentPolarityAnalyzer(model, config)
        task_spec = tasks["sentiment_flip"]
        for tid in task_spec.template_ids:
            acts = cached_by_task_template.get(("sentiment_flip", tid))
            if acts is None:
                continue
            try:
                sent_results = sentiment_analyzer.analyze_task_template(
                    task_spec, tid, layers, acts,
                )
                all_sentiment.extend(sent_results)
            except Exception as e:
                logger.error("Sentiment analysis failed for %s: %s", tid, e)

    results["sentiment_polarity"] = [asdict(r) for r in all_sentiment]

    # -------------------------
    # 3. FV Vocabulary Projection
    # -------------------------
    logger.info("Running FV vocabulary projection...")
    projector = FVVocabProjector(model, config)

    # Only project at the best steering layer per (task, template)
    # to keep output manageable
    best_layer_lookup: Dict[Tuple[str, str], int] = {}
    for s in iid_summaries:
        best_layer_lookup[(s.task, s.template_id)] = s.best_layer

    all_vocab: List[FVVocabResult] = []
    for task_name, task_spec in tqdm(tasks.items(), desc="FV Vocab", unit="task"):
        if task_name not in fvs:
            continue
        for tid in task_spec.template_ids:
            if tid not in fvs[task_name]:
                continue
            # Use best steering layer, fallback to middle layer
            best_layer = best_layer_lookup.get(
                (task_name, tid),
                layers[len(layers) // 2],
            )
            if best_layer in fvs[task_name][tid]:
                fv_obj = fvs[task_name][tid][best_layer]
                try:
                    vr = projector.project_fv(fv_obj, task_spec)
                    all_vocab.append(vr)
                except Exception as e:
                    logger.error(
                        "FV projection failed for %s/%s/L%d: %s",
                        task_name, tid, best_layer, e,
                    )

    # For JSON: truncate token lists to top 10 for readability
    vocab_for_json = []
    for vr in all_vocab:
        d = asdict(vr)
        d["top_positive_tokens"] = d["top_positive_tokens"][:10]
        d["top_positive_logits"] = d["top_positive_logits"][:10]
        d["top_negative_tokens"] = d["top_negative_tokens"][:10]
        d["top_negative_logits"] = d["top_negative_logits"][:10]
        vocab_for_json.append(d)
    results["fv_vocab_projection"] = vocab_for_json

    # -------------------------
    # 4. Readability-Steerability Gap
    # -------------------------
    gap_results = compute_readability_steerability_gap(
        all_readability, iid_summaries,
        sentiment_results=all_sentiment,
        readability_threshold=config.science.readability_threshold,
    )
    results["readability_steerability_gap"] = gap_results

    # -------------------------
    # 5. Wilcoxon Test
    # -------------------------
    wilcoxon_results = wilcoxon_readability_vs_steering(
        all_readability, iid_summaries,
    )
    results["wilcoxon_test"] = wilcoxon_results

    # -------------------------
    # 6. 2×2 Matrix Summary
    # -------------------------
    matrix = {
        "both_succeed": [],
        "readable_not_steerable": [],
        "steerable_not_readable": [],
        "both_fail": [],
    }
    for task, gap in gap_results.items():
        cell = gap["cell"]
        matrix[cell].append({
            "task": task,
            "readability": gap["best_readability"],
            "steerability": gap["best_steering_accuracy"],
            "gap": gap["gap"],
        })
    results["two_by_two_matrix"] = matrix

    # -------------------------
    # 6b. Post-Steering Logit Lens
    # -------------------------
    # Identify tasks in "steerable_not_readable" and "both_succeed" cells
    ps_targets: List[Tuple[str, str, bool]] = []  # (task, template_id, is_both_succeed)

    # Build per-(task, template) IID summary lookup
    iid_lookup: Dict[Tuple[str, str], IIDSummary] = {}
    for s in iid_summaries:
        iid_lookup[(s.task, s.template_id)] = s

    # Build per-(task, template, layer) zero-shot top-10 lookup
    zs_top10_lookup: Dict[Tuple[str, str], Dict[int, float]] = {}
    for r in all_readability:
        key = (r.task, r.template_id)
        zs_top10_lookup.setdefault(key, {})[r.layer] = r.top_10_accuracy

    # Collect steerable_not_readable pairs (per task×template, not just per task)
    for task, gap in gap_results.items():
        cell = gap["cell"]
        if cell in ("steerable_not_readable", "both_succeed"):
            is_both = cell == "both_succeed"
            # Run for all templates of this task that have IID summaries
            for s in iid_summaries:
                if s.task == task and s.status == "PASS":
                    ps_targets.append((task, s.template_id, is_both))

    if ps_targets:
        logger.info(
            "Running post-steering logit lens on %d (task, template) pairs...",
            len(ps_targets),
        )
        ps_analyzer = PostSteeringLogitLensAnalyzer(model, config)
        ps_batch_size = max(config.ops.extraction_batch_size // 2, 8)
        all_ps_results: List[PostSteeringLogitLensResult] = []

        for task_name, tid, is_both in tqdm(
            ps_targets, desc="Post-Steering LL", unit="pair"
        ):
            task_spec = tasks.get(task_name)
            if task_spec is None:
                continue

            iid_info = iid_lookup.get((task_name, tid))
            if iid_info is None:
                continue

            # Get FV at best layer
            best_layer_1idx = iid_info.best_layer
            if (task_name not in fvs
                    or tid not in fvs[task_name]
                    or best_layer_1idx not in fvs[task_name][tid]):
                logger.warning(
                    "No FV for %s/%s/L%d — skipping post-steering LL",
                    task_name, tid, best_layer_1idx,
                )
                continue

            fv_obj = fvs[task_name][tid][best_layer_1idx]
            fv_tensor = fv_obj.vector.to(model.device)
            fv_layer_0idx = fv_obj.layer_0idx

            zs_by_layer = zs_top10_lookup.get((task_name, tid), {})

            try:
                ps_result = ps_analyzer.analyze_task_template(
                    task_spec=task_spec,
                    template_id=tid,
                    fv_tensor=fv_tensor,
                    fv_layer_0idx=fv_layer_0idx,
                    strength=iid_info.best_strength,
                    zero_shot_top10_by_layer=zs_by_layer,
                    is_both_succeed=is_both,
                    batch_size=ps_batch_size,
                )
                if ps_result is not None:
                    all_ps_results.append(ps_result)
            except Exception as e:
                logger.error(
                    "Post-steering LL failed for %s/%s: %s",
                    task_name, tid, e,
                )

        # Store results
        ps_results_json: Dict[str, Dict[str, Any]] = {}
        for psr in all_ps_results:
            ps_results_json.setdefault(psr.task, {})[psr.template_id] = {
                "injection_layer": psr.injection_layer,
                "injection_strength": psr.injection_strength,
                "n_examples": psr.n_examples,
                "per_layer": {
                    str(k): v for k, v in psr.per_layer.items()
                },
                "max_post_steering_top10": psr.max_post_steering_top10,
                "max_delta": psr.max_delta,
                "answer_emerges_at_layer": psr.answer_emerges_at_layer,
                "zero_shot_max_top10": psr.zero_shot_max_top10,
                "zero_shot_peak_layer": psr.zero_shot_peak_layer,
                "post_steering_peak_layer": psr.post_steering_peak_layer,
                "peak_layer_delta": psr.peak_layer_delta,
                "random_control_max_top10": psr.random_control_max_top10,
                "category": psr.category,
            }
        results["post_steering_logit_lens"] = ps_results_json

        # Print summary
        _print_post_steering_summary(all_ps_results)
    else:
        logger.info(
            "No steerable_not_readable or both_succeed pairs — "
            "skipping post-steering logit lens."
        )

    # -------------------------
    # 7. Steering Harm Analysis (zero-shot vs FV steering)
    # -------------------------
    baseline_path = config.results_dir / "baseline_results.json"
    if baseline_path.exists():
        logger.info("Running steering harm analysis (zero-shot vs FV steering)...")
        import json as _json
        with open(baseline_path) as f:
            baseline_data = _json.load(f)
        # Load per-layer steering results if not passed directly
        _steering_data = steering_results
        if _steering_data is None:
            steer_path = config.results_dir / "steering_results.json"
            if steer_path.exists():
                with open(steer_path) as sf:
                    _steering_data = _json.load(sf)
                logger.info("Loaded steering results from %s for per-layer harm analysis", steer_path)

        harm_results = compute_steering_harm(
            iid_summaries or [], baseline_data, steering_results=_steering_data,
        )
        results["steering_harm"] = harm_results
        _print_harm_summary(harm_results)
    else:
        logger.info(
            "Skipping steering harm analysis — no baseline results at %s",
            baseline_path,
        )

    # -------------------------
    # Print Summary
    # -------------------------
    _print_readability_summary(all_readability, all_sentiment, gap_results, matrix)

    # Free memory
    del cached_by_task_template
    import gc
    gc.collect()

    return results


def _print_readability_summary(
    readability: List[LogitLensResult],
    sentiment: List[SentimentPolarityResult],
    gaps: Dict[str, Any],
    matrix: Dict[str, List],
):
    """Print human-readable summary to stdout."""
    print("\n" + "=" * 70)
    print("LOGIT LENS READABILITY (best layer per task × template)")
    print("=" * 70)
    print(f"{'Task':<20} {'Template':<10} {'Layer':<8} "
          f"{'Top-1':<8} {'Top-5':<8} {'Top-10':<8} {'MeanRank':<10}")
    print("-" * 75)

    by_tt: Dict[Tuple[str, str], List[LogitLensResult]] = {}
    for r in readability:
        by_tt.setdefault((r.task, r.template_id), []).append(r)

    for (task, tid), rs in sorted(by_tt.items()):
        # Best layer by top-10 accuracy
        best = max(rs, key=lambda x: x.top_10_accuracy)
        print(f"{task:<20} {tid:<10} L{best.layer:<7} "
              f"{best.top_1_accuracy:<8.3f} {best.top_5_accuracy:<8.3f} "
              f"{best.top_10_accuracy:<8.3f} {best.mean_correct_rank:<10.1f}")

    if sentiment:
        print("\n" + "-" * 70)
        print("SENTIMENT POLARITY (sentiment_flip)")
        print("-" * 70)
        print(f"{'Template':<10} {'Layer':<8} {'Polarity Score':<16} {'Class Acc':<10}")
        print("-" * 50)
        by_t: Dict[str, List[SentimentPolarityResult]] = {}
        for sr in sentiment:
            by_t.setdefault(sr.template_id, []).append(sr)
        for tid, srs in sorted(by_t.items()):
            best = max(srs, key=lambda x: x.polarity_classification_accuracy)
            print(f"{tid:<10} L{best.layer:<7} {best.mean_polarity_score:<16.3f} "
                  f"{best.polarity_classification_accuracy:<10.3f}")

    print("\n" + "-" * 70)
    print("READABILITY-STEERABILITY GAP")
    print("-" * 70)
    print(f"{'Task':<20} {'Readability':<14} {'Steerability':<14} "
          f"{'Gap':<10} {'Cell'}")
    print("-" * 70)
    for task, gap in sorted(gaps.items()):
        print(f"{task:<20} {gap['best_readability']:<14.3f} "
              f"{gap['best_steering_accuracy']:<14.3f} "
              f"{gap['gap']:<10.3f} {gap['cell']}")

    print("\n" + "-" * 70)
    print("2×2 MATRIX")
    print("-" * 70)
    for cell, entries in matrix.items():
        task_names = [e["task"] for e in entries]
        print(f"  {cell}: {', '.join(task_names) if task_names else '(empty)'}")
    print()


def _print_harm_summary(harm: Dict[str, Any]):
    """Print steering harm analysis summary."""
    agg = harm.get("aggregate", {})
    per_task = harm.get("per_task", {})
    clustering = harm.get("destructive_layer_clustering", {})

    print("\n" + "=" * 70)
    print("STEERING HARM ANALYSIS (zero-shot baseline vs FV steering)")
    print("=" * 70)
    print(f"Total (task, template) pairs: {agg.get('total_pairs', 0)}")
    print(f"Destructive cases (steering < zero-shot): "
          f"{agg.get('total_destructive', 0)} "
          f"({agg.get('destructive_rate', 0):.1%})")
    print(f"Mean steering delta (all): {agg.get('mean_delta_all', 0):+.3f}")

    print(f"\n{'Task':<20} {'ZeroShot':<10} {'FewShot':<10} {'Steering':<10} "
          f"{'Delta':<10} {'Destruct':<10} {'ICL Recov':<10}")
    print("-" * 80)
    for task, ts in sorted(per_task.items()):
        flag = f"{ts['n_destructive']}/{ts['n_templates']}" if ts['n_destructive'] > 0 else ""
        print(f"{task:<20} {ts['mean_zero_shot']:<10.3f} {ts['mean_few_shot']:<10.3f} "
              f"{ts['mean_steering']:<10.3f} {ts['mean_delta']:<+10.3f} "
              f"{flag:<10} {ts['mean_icl_recovery']:<+10.2f}")

    # Layer clustering for destructive cases
    if clustering:
        print(f"\nDestructive Layer Clustering:")
        most_destructive = clustering.get("most_destructive_layer")
        concentration = clustering.get("concentration_ratio", 0)
        total = clustering.get("total_destructive_cases", 0)
        print(f"  Most destructive layer: L{most_destructive} "
              f"({clustering.get('most_destructive_layer_count', 0)}/{total} cases, "
              f"{concentration:.0%} concentration)")
        dist = clustering.get("worst_layer_distribution", {})
        if dist:
            print(f"  Top-5 worst layers: "
                  + ", ".join(f"L{l}={c}" for l, c in dist.items()))
    print()


def _print_post_steering_summary(
    ps_results: List[PostSteeringLogitLensResult],
):
    """Print post-steering logit lens summary to stdout."""
    if not ps_results:
        return

    print("\n" + "=" * 70)
    print("POST-STEERING LOGIT LENS ANALYSIS")
    print("=" * 70)
    print(f"{'Task':<18} {'Tmpl':<6} {'Inj.L':<7} {'ZS Max':<8} "
          f"{'PS Max':<8} {'Rand':<8} {'Delta':<8} {'Emerges@':<10} {'Category'}")
    print("-" * 95)

    for r in sorted(ps_results, key=lambda x: (x.task, x.template_id)):
        if r.answer_emerges_at_layer is not None:
            if r.category == "amplified":
                emerges = f"L{r.zero_shot_peak_layer}->L{r.post_steering_peak_layer}"
            else:
                emerges = f"L{r.answer_emerges_at_layer}"
        else:
            emerges = "--"

        print(
            f"{r.task:<18} {r.template_id:<6} L{r.injection_layer:<6} "
            f"{r.zero_shot_max_top10:<8.3f} {r.max_post_steering_top10:<8.3f} "
            f"{r.random_control_max_top10:<8.3f} {r.max_delta:<+8.3f} "
            f"{emerges:<10} {r.category}"
        )

    # Aggregate summary
    snr = [r for r in ps_results if r.category in ("absent", "partial", "emerged")]
    bs = [r for r in ps_results if r.category in ("amplified", "unchanged")]

    print(f"\nPOST-STEERING LOGIT LENS SUMMARY")
    if snr:
        n_absent = sum(1 for r in snr if r.category == "absent")
        n_partial = sum(1 for r in snr if r.category == "partial")
        n_emerged = sum(1 for r in snr if r.category == "emerged")
        print(f"Steerable-but-not-readable pairs analyzed: {len(snr)}")
        print(f"  absent  (no new decodability):  {n_absent} ({n_absent/len(snr):.0%})")
        print(f"  partial (weak emergence):       {n_partial} ({n_partial/len(snr):.0%})")
        print(f"  emerged (full decodability):    {n_emerged} ({n_emerged/len(snr):.0%})")

    if bs:
        n_amp = sum(1 for r in bs if r.category == "amplified")
        n_unch = sum(1 for r in bs if r.category == "unchanged")
        print(f"Both-succeed control pairs analyzed: {len(bs)}")
        print(f"  amplified (peak shifted earlier): {n_amp} ({n_amp/len(bs):.0%})")
        print(f"  unchanged (peak at same layer):   {n_unch} ({n_unch/len(bs):.0%})")

    # Random control comparison (paired Wilcoxon if enough data)
    ps_maxes = [r.max_post_steering_top10 for r in ps_results]
    rand_maxes = [r.random_control_max_top10 for r in ps_results]
    mean_delta = float(np.mean([p - r for p, r in zip(ps_maxes, rand_maxes)]))
    if len(ps_results) >= 5:
        diffs = np.array(ps_maxes) - np.array(rand_maxes)
        if not np.all(diffs == 0):
            try:
                _, p_val = wilcoxon(ps_maxes, rand_maxes)
                print(f"Random control: mean PS-vs-Random delta = "
                      f"{mean_delta:+.3f} (p={p_val:.3f}, paired Wilcoxon)")
            except Exception:
                print(f"Random control: mean PS-vs-Random delta = {mean_delta:+.3f}")
        else:
            print(f"Random control: mean PS-vs-Random delta = {mean_delta:+.3f} "
                  f"(all differences zero)")
    else:
        print(f"Random control: mean PS-vs-Random delta = {mean_delta:+.3f} "
              f"(too few pairs for Wilcoxon)")

    # Interpretive statement
    if snr:
        cats = [r.category for r in snr]
        from collections import Counter
        dominant = Counter(cats).most_common(1)[0][0]
        interpretations = {
            "absent": "absent -- consistent with computational instruction hypothesis",
            "partial": "partial -- weak emergence suggests partial computational instruction",
            "emerged": "emerged -- FV creates fully decodable information",
        }
        print(f"Dominant pattern: {interpretations.get(dominant, dominant)}")

    print()


def save_readability_results(results: Dict, output_dir: Path):
    """Save readability analysis results to JSON."""
    path = output_dir / "readability_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Readability results saved to %s", path)
