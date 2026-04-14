"""
MLPDecoderConfig
================
Experiment-specific config for the MLP decoder probe study. Keeps scope and
hyperparameters in one place, separate from the main pipeline's config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Default scope
# ---------------------------------------------------------------------------
ALL_MODELS: List[str] = [
    "llama-3.1-8b-base",
    "llama-3.1-8b-instruct",
    "gemma-2-9b-base",
    "gemma-2-9b-it",
    "mistral-7b-base",
    "mistral-7b-instruct",
]

ALL_TASKS: List[str] = [
    "antonym", "synonym", "hypernym",
    "country_capital", "english_spanish", "object_color",
    "past_tense", "plural",
    "capitalize", "first_letter", "reverse_word",
    "sentiment_flip",
]


# Output directories used by the main pipeline use short names rather than
# the registry keys. Map registry-key -> short name. If the short directory
# does not exist we fall back to the registry key.
MODEL_OUTPUT_ALIAS: Dict[str, str] = {
    "llama-3.1-8b-base": "llama-base",
    "llama-3.1-8b-instruct": "llama-instruct",
    "gemma-2-9b-base": "gemma-2-base",
    "gemma-2-9b-it": "gemma-2-instruct",
    "mistral-7b-base": "mistral-base",
    "mistral-7b-instruct": "mistral-instruct",
}


def output_dir_name(model_key: str) -> str:
    """Map a registry key to the directory name used under outputs/."""
    return MODEL_OUTPUT_ALIAS.get(model_key, model_key)


@dataclass
class MLPDecoderConfig:
    """
    All parameters for the MLP decoder probe experiment.

    Defaults target an H200 (~150 GB VRAM): aggressive batching,
    bf16 forward passes, full-batch MLP training. End-to-end runtime
    target is 2-3 hours across all 6 models.
    """

    # ---- scope ----
    models: List[str] = field(default_factory=lambda: list(ALL_MODELS))
    tasks: List[str] = field(default_factory=lambda: list(ALL_TASKS))
    # If None: use the model's existing extraction layers (every 2nd layer
    # via ModelSpec.extraction_layers()). Otherwise, override.
    layers_1idx: Optional[List[int]] = None

    # ---- activation extraction ----
    # H200 with bf16: forward pass on 7-9B model with batch 256 fits easily.
    extraction_batch_size: int = 256
    dtype: str = "bfloat16"
    # Shared random seed for reproducibility.
    seed: int = 42

    # ---- MLP probe architecture ----
    # 2-layer MLP per Hewitt & Liang convention, GELU nonlinearity.
    # Hidden dim chosen to give the probe meaningful nonlinear capacity
    # without making the output-layer matmul dominate. (Output layer is
    # h x vocab -- vocab ranges 32k-256k across the 3 model families.)
    hidden_dim: int = 1024
    dropout: float = 0.1
    # If True, prepend a LayerNorm to the input -- helps when residual-stream
    # magnitudes vary by layer/model (they do, by 10x).
    input_layernorm: bool = True

    # ---- MLP training ----
    n_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Full-batch training: each task has ~600-700 examples (8 templates *
    # 75-95 unique inputs). The gradient is small so full-batch is fine,
    # and avoids DataLoader overhead. The tensors live on GPU throughout.
    full_batch: bool = True
    # If full_batch=False, use this batch size.
    batch_size: int = 128
    # AdamW with linear warmup + cosine decay.
    warmup_fraction: float = 0.1

    # ---- Train/test split ----
    # Split by *unique input* (not by example) so the probe must generalize
    # across inputs it never saw during training. This is stricter than
    # random per-example split and matches Hewitt & Liang.
    test_input_fraction: float = 0.20

    # ---- Hewitt & Liang control task ----
    # Shuffle real labels deterministically per input -- the same input gets
    # the same random label across all 8 templates, but the input-to-label
    # mapping is now arbitrary. The selectivity gap (real - control) tests
    # whether the probe is using genuine task structure or just memorizing.
    run_control_probes: bool = True
    control_seed: int = 1234

    # ---- Top-k metrics ----
    top_k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    # Threshold to populate the 2x2 / 2x3 matrix. Matches main paper's
    # readability_threshold for direct comparison.
    decodability_threshold: float = 0.10

    # ---- I/O ----
    # Project root (where this package's parent fv_cross_template lives).
    project_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "FV_PROJECT_ROOT",
                str(Path(__file__).resolve().parent.parent),
            )
        )
    )
    # Where the main pipeline writes its outputs (logit lens, tuned lens,
    # steering, etc.). The probe study reads from here for cross-comparison.
    main_outputs_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("FV_OUTPUT_DIR", "outputs")
        ).resolve()
    )
    # Where this experiment writes its outputs.
    mlp_outputs_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("MLP_OUTPUT_DIR", "outputs_mlp_decoder")
        ).resolve()
    )

    # ---- Activation cache ----
    # Per-model file: outputs_mlp_decoder/<model>/activations.pt
    # Contains: dict[task_name][template_id][layer_1idx] -> Tensor[N, d_model]
    cache_activations_to_disk: bool = True
    # If True, skip extraction when cache file already exists.
    reuse_existing_cache: bool = True

    # ---- Performance flags ----
    # (Reserved for future use. Previous flags `use_torch_compile` and
    # `pin_activations_to_gpu` were removed because they were never wired
    # up in the training loop. _stack_task_data already moves each
    # (task, layer) tensor to GPU on demand, which is fast enough.)

    # ---------------------------------------------------------------------
    # Derived paths
    # ---------------------------------------------------------------------
    def _model_dir_name(self, model_key: str) -> str:
        # Use the same short alias the main pipeline uses, so a user with
        # outputs/llama-base/ also gets outputs_mlp_decoder/llama-base/.
        return output_dir_name(model_key)

    def model_cache_path(self, model_key: str) -> Path:
        d = self.mlp_outputs_dir / self._model_dir_name(model_key)
        d.mkdir(parents=True, exist_ok=True)
        return d / "activations.pt"

    def model_results_path(self, model_key: str) -> Path:
        d = self.mlp_outputs_dir / self._model_dir_name(model_key)
        d.mkdir(parents=True, exist_ok=True)
        return d / "mlp_probe_results.json"

    def figures_dir(self) -> Path:
        d = self.mlp_outputs_dir / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def summary_path(self) -> Path:
        self.mlp_outputs_dir.mkdir(parents=True, exist_ok=True)
        return self.mlp_outputs_dir / "summary.json"

    def comparison_path(self) -> Path:
        self.mlp_outputs_dir.mkdir(parents=True, exist_ok=True)
        return self.mlp_outputs_dir / "decoder_comparison.json"

    def main_pipeline_results_for(self, model_key: str) -> Path:
        # Try the short alias first, then the registry key, then the literal
        # model_key (in case the user has a custom layout).
        candidates = [
            self.main_outputs_dir / self._model_dir_name(model_key) / "results",
            self.main_outputs_dir / model_key / "results",
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]  # default; downstream callers will warn if missing

    def all_paths_exist_for(self, model_key: str) -> bool:
        """Check whether the main-pipeline results we need are present."""
        rd = self.main_pipeline_results_for(model_key)
        needed = ["readability_results.json", "tuned_lens_results.json",
                  "iid_summary.json", "steering_results.json"]
        return all((rd / f).exists() for f in needed)


# Singleton-style helpers ---------------------------------------------------

def default_config() -> MLPDecoderConfig:
    return MLPDecoderConfig()
