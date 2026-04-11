"""
Configuration system with typed, validated, documented parameters.
=================================================================
Every threshold, range, or cutoff that affects scientific conclusions is:
  - Configurable (not hardcoded in analysis code)
  - Documented with justification
  - Either derived from data or clearly flagged as a design choice

Scientific parameters are separated from operational parameters.
"""

from __future__ import annotations

import logging
import os
import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
SEED = 42


class ModelVariant(Enum):
    BASE = "base"
    INSTRUCT = "instruct"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a supported model."""
    hf_name: str
    display_name: str
    variant: ModelVariant
    family: str
    n_layers: int
    d_model: int
    n_heads: int

    # Proportional extraction: extract at every k-th layer
    # Default: every 2nd layer starting from layer 2
    extraction_step: int = 2

    def extraction_layers(self) -> List[int]:
        """Layers to extract FVs from (1-indexed, proportional to depth)."""
        return list(range(self.extraction_step, self.n_layers + 1, self.extraction_step))

    def mid_layer_range(self) -> Tuple[int, int]:
        """Middle third of layers (for attention analysis etc.)."""
        third = self.n_layers // 3
        return (third, 2 * third)

    def late_layer_range(self) -> Tuple[int, int]:
        """Last third of layers."""
        third = self.n_layers // 3
        return (2 * third, self.n_layers)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # --- Llama-3.1-8B family ---
    "llama-3.1-8b-base": ModelSpec(
        hf_name="meta-llama/Llama-3.1-8B",
        display_name="Llama-3.1-8B (Base)",
        variant=ModelVariant.BASE,
        family="llama",
        n_layers=32, d_model=4096, n_heads=32,
    ),
    "llama-3.1-8b-instruct": ModelSpec(
        hf_name="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama-3.1-8B (Instruct)",
        variant=ModelVariant.INSTRUCT,
        family="llama",
        n_layers=32, d_model=4096, n_heads=32,
    ),
    # --- Gemma-2-9B family ---
    "gemma-2-9b-base": ModelSpec(
        hf_name="google/gemma-2-9b",
        display_name="Gemma-2-9B (Base)",
        variant=ModelVariant.BASE,
        family="gemma",
        n_layers=42, d_model=3584, n_heads=16,
    ),
    "gemma-2-9b-it": ModelSpec(
        hf_name="google/gemma-2-9b-it",
        display_name="Gemma-2-9B (IT)",
        variant=ModelVariant.INSTRUCT,
        family="gemma",
        n_layers=42, d_model=3584, n_heads=16,
    ),
    # --- Mistral-7B family ---
    "mistral-7b-base": ModelSpec(
        hf_name="mistralai/Mistral-7B-v0.3",
        display_name="Mistral-7B-v0.3 (Base)",
        variant=ModelVariant.BASE,
        family="mistral",
        n_layers=32, d_model=4096, n_heads=32,
    ),
    "mistral-7b-instruct": ModelSpec(
        hf_name="mistralai/Mistral-7B-Instruct-v0.3",
        display_name="Mistral-7B-v0.3 (Instruct)",
        variant=ModelVariant.INSTRUCT,
        family="mistral",
        n_layers=32, d_model=4096, n_heads=32,
    ),
}


# ---------------------------------------------------------------------------
# Scientific parameters  (affect which data enters analysis)
# ---------------------------------------------------------------------------
@dataclass
class ScientificParams:
    """
    Every field here can change what conclusions you draw.
    Each has a JUSTIFICATION comment.
    """

    # -- IID gating threshold --
    # Justification: For a task with ~50-word output vocabulary, chance is ~2%.
    # We require IID accuracy to be *significantly* above chance.  A threshold
    # of 0.10 (10%) is ~5x chance for most single-token tasks and allows
    # borderline tasks to be investigated via probing.  This is conservative
    # enough to exclude capitalize (0.00) and sentiment_flip (0.02) from
    # causal analysis while including any task that shows minimal steering
    # effect.  The threshold is configurable so sensitivity analysis can test
    # 0.05, 0.15, 0.20.
    iid_accuracy_threshold: float = 0.10

    # -- Dissociation detection --
    # Justification: "high alignment" = top quartile of cosine distribution.
    # We use a configurable absolute cutoff as a starting point but also
    # compute data-derived percentile thresholds in analysis.
    high_alignment_cosine: float = 0.80
    # Justification: "low transfer" = accuracy where steering has essentially
    # failed.  0.40 is generous -- at this level less than half of test
    # examples are correct.
    low_transfer_accuracy: float = 0.40

    # -- Steering strength sweep --
    # Justification: Todd et al. 2024 use strengths 1-10 on GPT-2.  For
    # larger models the effective range is smaller.  We start with a broad
    # sweep, then narrow.  The adaptive refinement step adds 3 points
    # around the best strength found.
    steering_strengths: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    )
    adaptive_refine: bool = True
    adaptive_refine_points: int = 3
    adaptive_refine_delta: float = 0.25

    # -- Patching recovery threshold --
    # Justification: A layer "enables recovery" if patching it restores
    # accuracy to at least this fraction of IID accuracy.  0.50 means
    # patching recovers at least half the IID performance.
    patching_recovery_threshold: float = 0.50

    # -- Patching case selection --
    # Stratified selection: pick top K dissociation cases per task rather
    # than a global top-N, so every IID-viable task gets representation.
    patching_cases_per_task: int = 5
    # Hard cap on total patching cases (safety valve for compute).
    patching_max_cases: int = 60

    # -- Probing (spec Section 5.6.3) --
    # Data split: 70% train / 15% validation / 15% test
    probe_train_fraction: float = 0.70
    probe_val_fraction: float = 0.15
    probe_test_fraction: float = 0.15
    # Regularization C sweep on validation set (spec Section 5.6.3)
    probe_regularization_sweep: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0]
    )
    probe_max_iter: int = 2000

    # -- Readability analysis (logit lens + FV vocabulary projection) --
    # Replaces the broken linear probing stage.  The logit lens projects
    # zero-shot activations through the model's own unembedding matrix —
    # no learned parameters, so probe-complexity critique does not apply.
    # top_k: number of top tokens to check for correct output membership.
    readability_top_k: List[int] = field(
        default_factory=lambda: [1, 5, 10]
    )
    # FV vocabulary projection: how many top/bottom tokens to record.
    fv_vocab_top_n: int = 50
    # Readability threshold for the 2×2 matrix: a task is "readable" if
    # logit lens top-10 accuracy exceeds this value at the best layer.
    readability_threshold: float = 0.10

    # -- Data-derived thresholds --
    # Per spec Section 5.3: use data-derived (percentile-based) thresholds,
    # not absolute cutoffs, as the primary dissociation detection method.
    use_data_derived_thresholds: bool = True
    alignment_percentile: float = 75.0  # top quartile


# ---------------------------------------------------------------------------
# Operational parameters  (affect runtime, not conclusions)
# ---------------------------------------------------------------------------
@dataclass
class OperationalParams:
    """These affect speed/memory but not scientific results."""
    device: str = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
    dtype: str = "bfloat16"
    extraction_batch_size: int = 8
    steering_batch_size: int = 16
    patching_batch_size: int = 4
    tuned_lens_batch_size: int = 8
    max_new_tokens: int = 5
    n_icl_demos: int = 5       # demos per ICL prompt
    n_icl_positive: int = 15   # positive ICL examples for extraction
    n_icl_negative: int = 15   # negative ICL examples for extraction
    n_iid_test: int = 50
    n_ood_test: int = 50       # per target template

    # -- Performance tuning --
    # Batch N steering strengths into a single forward/generate call.
    # Set to len(steering_strengths) for maximum speed; 1 = original behaviour.
    strength_tile_factor: int = 1

    # Enable CUDA performance flags (tf32, cudnn benchmark).
    # These affect numerical precision at ~1e-7 but NOT discrete accuracy.
    cuda_optimizations: bool = True

    # NOTE: skip_ood_on_zero_iid removed — all OOD evaluations now run
    # unconditionally to ensure complete results across all tasks/layers.


# ---------------------------------------------------------------------------
# GPU Memory Profiles (affect speed/memory, NOT scientific results)
# ---------------------------------------------------------------------------
GPU_PROFILES: Dict[str, Dict] = {
    "l4": {
        "extraction_batch_size": 16,
        "steering_batch_size": 48,
        "patching_batch_size": 16,
        "tuned_lens_batch_size": 16,
        "strength_tile_factor": 8,
    },
    "a100-40": {
        "extraction_batch_size": 32,
        "steering_batch_size": 48,
        "patching_batch_size": 32,
        "tuned_lens_batch_size": 32,
        "strength_tile_factor": 8,
    },
    "a100-80": {
        "extraction_batch_size": 64,
        "steering_batch_size": 64,
        "patching_batch_size": 64,
        "tuned_lens_batch_size": 64,
        "strength_tile_factor": 8,
    },
    "rtx-4090": {
        "extraction_batch_size": 16,
        "steering_batch_size": 24,
        "patching_batch_size": 16,
        "tuned_lens_batch_size": 16,
        "strength_tile_factor": 8,
    },
    # --- H100 / H200 profiles ---
    # 7-9B models in bf16 ≈ 14-18 GB.  Remaining VRAM is for activations.
    # Steering batch = total sequences per generate call (eff = batch / tile).
    # Tile factor 8 processes all 8 base strengths in one forward pass.
    "h100": {
        # 80 GB VRAM  →  ~60 GB free after model load
        "extraction_batch_size": 128,
        "steering_batch_size": 256,   # eff 32 per strength
        "patching_batch_size": 128,
        "tuned_lens_batch_size": 128,
        "strength_tile_factor": 8,
    },
    "h200": {
        # 141 GB VRAM  →  ~120 GB free after model load
        "extraction_batch_size": 256,
        "steering_batch_size": 512,   # eff 64 per strength
        "patching_batch_size": 256,
        "tuned_lens_batch_size": 256,
        "strength_tile_factor": 8,
    },
}


def auto_detect_gpu_profile() -> Optional[str]:
    """Auto-detect GPU type and return matching profile name, or None."""
    if not (_HAS_TORCH and torch.cuda.is_available()):
        return None
    try:
        gpu_name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return None

    # Match known GPU types (careful: "l40" != "l4")
    if "h200" in gpu_name:
        return "h200"
    elif "h100" in gpu_name:
        return "h100"
    elif "l4" in gpu_name and "l40" not in gpu_name:
        return "l4"
    elif "a100" in gpu_name:
        total_mem = torch.cuda.get_device_properties(0).total_mem
        return "a100-80" if total_mem > 50 * 1024**3 else "a100-40"
    elif "4090" in gpu_name:
        return "rtx-4090"
    return None


def apply_gpu_profile(ops: OperationalParams, profile_name: str) -> OperationalParams:
    """Apply a GPU memory profile to operational parameters."""
    if profile_name not in GPU_PROFILES:
        raise ValueError(
            f"Unknown GPU profile '{profile_name}'. "
            f"Available: {list(GPU_PROFILES.keys())}"
        )
    profile = GPU_PROFILES[profile_name]
    for key, value in profile.items():
        if hasattr(ops, key):
            setattr(ops, key, value)
    logger.info("Applied GPU profile '%s': %s", profile_name, profile)
    return ops


# ---------------------------------------------------------------------------
# Experiment configuration  (top-level, composes everything)
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Which model to run
    model_key: str = "llama-3.1-8b-base"

    # Which tasks to run (None = all)
    task_names: Optional[List[str]] = None

    # Which stages to run
    # Per spec Section 4.2 + 10.1: baseline stage runs zero-shot + few-shot
    # BEFORE extraction; probe_activations caches zero-shot activations AFTER
    # extraction but BEFORE steering.
    stages: List[str] = field(
        default_factory=lambda: [
            "baseline", "extract", "probe_activations", "steer",
            "analyze", "mechanistic", "readability", "tuned_lens", "figures",
        ]
    )

    # Comparison model (for base-vs-instruct)
    comparison_model_key: Optional[str] = None

    # Scientific and operational sub-configs
    science: ScientificParams = field(default_factory=ScientificParams)
    ops: OperationalParams = field(default_factory=OperationalParams)

    # Paths -- relative to project root by default, overridable via env var
    output_root: Path = field(default_factory=lambda: Path(
        os.environ.get("FV_OUTPUT_DIR", "outputs")
    ))

    seed: int = SEED

    def __post_init__(self):
        if self.model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model key '{self.model_key}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

    # -- Derived paths --
    @property
    def model_spec(self) -> ModelSpec:
        return MODEL_REGISTRY[self.model_key]

    @property
    def run_dir(self) -> Path:
        """Per-model output directory."""
        d = self.output_root / self.model_key
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cache_dir(self) -> Path:
        d = self.run_dir / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def figures_dir(self) -> Path:
        d = self.run_dir / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def results_dir(self) -> Path:
        d = self.run_dir / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- Serialization --
    def config_hash(self) -> str:
        """Deterministic hash for cache invalidation."""
        d = {
            "model_key": self.model_key,
            "seed": self.seed,
            "science": asdict(self.science),
            "ops_n_icl_positive": self.ops.n_icl_positive,
            "ops_n_icl_negative": self.ops.n_icl_negative,
            "ops_n_icl_demos": self.ops.n_icl_demos,
            "ops_n_iid_test": self.ops.n_iid_test,
            "ops_n_ood_test": self.ops.n_ood_test,
        }
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def save(self, path: Optional[Path] = None):
        path = path or (self.run_dir / "experiment_config.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        logger.info("Config saved to %s", path)

    def validate(self) -> List[str]:
        """Return list of warnings (empty = all good)."""
        warnings = []
        spec = self.model_spec
        layers = spec.extraction_layers()
        if len(layers) == 0:
            warnings.append(f"No extraction layers for {self.model_key}")
        if self.science.iid_accuracy_threshold < 0.05:
            warnings.append(
                "IID threshold < 5% is very permissive -- tasks near chance will pass"
            )
        if self.science.iid_accuracy_threshold > 0.50:
            warnings.append(
                "IID threshold > 50% is aggressive -- many tasks may be excluded"
            )
        if self.ops.max_new_tokens < 1:
            warnings.append("max_new_tokens < 1 means no generation")
        return warnings
