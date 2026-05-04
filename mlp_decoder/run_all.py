"""
End-to-end runner for the MLP decoder probe experiment.

Pipeline:
  1. For each model: load -> extract zero-shot residual streams at every
     extraction layer -> save activations to disk -> unload model.
  2. For each model: load cached activations -> train one real-label MLP
     probe and one Hewitt & Liang control probe at every (task, layer) ->
     save full results JSON.
  3. Read main-pipeline JSONs (logit lens, tuned lens, IID steering) and
     build the cross-decoder comparison + 2x4 matrix.
  4. Generate figures.
  5. Print a one-paragraph summary.

Designed for H200 (~150 GB VRAM) with aggressive batching. Target end-to-end
runtime: 2-3 hours for all 6 models, 12 tasks, 8 templates, every-2-layer.

CLI:
    python -m fv_cross_template.mlp_decoder.run_all \
        --output-dir outputs_mlp_decoder \
        --main-output-dir outputs

    # Subset run (faster):
    python -m fv_cross_template.mlp_decoder.run_all \
        --models llama-3.1-8b-base \
        --tasks antonym country_capital first_letter

Phases can be skipped:
    --skip-extract     Re-use existing cached activations
    --skip-train       Re-use existing MLP probe results
    --skip-analyze     Re-use existing decoder_comparison.json
    --skip-figures     Skip figure generation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

from .config import (
    ALL_MODELS,
    ALL_TASKS,
    MLPDecoderConfig,
    default_config,
)


def _setup_logging(verbose: bool) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format=fmt, datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Quieten noisy libraries
    for noisy in ("transformer_lens", "transformers", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "MLP decoder probe study: tests whether the steerability-without-"
            "decodability dissociation survives a 2-layer nonlinear decoder."
        )
    )
    p.add_argument("--models", nargs="*", default=None,
                   help="Model keys to run (default: all 6).")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Task names to run (default: all 12).")
    p.add_argument("--output-dir", default=None,
                   help="Where MLP probe outputs are written.")
    p.add_argument("--main-output-dir", default=None,
                   help="Where the main pipeline wrote its outputs (we read "
                        "logit lens / tuned lens / steering JSONs from here).")
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override extraction_batch_size for the forward pass.")
    p.add_argument("--no-control", action="store_true",
                   help="Skip control-task probes (faster, but no selectivity).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-analyze", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> MLPDecoderConfig:
    cfg = default_config()
    if args.models is not None:
        for m in args.models:
            if m not in ALL_MODELS:
                raise ValueError(f"Unknown model key: {m}. Available: {ALL_MODELS}")
        cfg.models = list(args.models)
    if args.tasks is not None:
        for t in args.tasks:
            if t not in ALL_TASKS:
                raise ValueError(f"Unknown task: {t}. Available: {ALL_TASKS}")
        cfg.tasks = list(args.tasks)
    if args.output_dir:
        cfg.mlp_outputs_dir = Path(args.output_dir).resolve()
    if args.main_output_dir:
        cfg.main_outputs_dir = Path(args.main_output_dir).resolve()
    if args.hidden_dim is not None:
        cfg.hidden_dim = int(args.hidden_dim)
    if args.n_epochs is not None:
        cfg.n_epochs = int(args.n_epochs)
    if args.batch_size is not None:
        cfg.extraction_batch_size = int(args.batch_size)
    if args.no_control:
        cfg.run_control_probes = False
    if args.seed is not None:
        cfg.seed = int(args.seed)
    return cfg


def print_config_banner(cfg: MLPDecoderConfig) -> None:
    log = logging.getLogger("mlp_decoder.run_all")
    log.info("=" * 72)
    log.info("MLP DECODER PROBE STUDY")
    log.info("=" * 72)
    log.info("Models  : %s", cfg.models)
    log.info("Tasks   : %s", cfg.tasks)
    log.info("MLP cfg : hidden=%d, epochs=%d, lr=%g, dropout=%g, layernorm=%s",
             cfg.hidden_dim, cfg.n_epochs, cfg.learning_rate,
             cfg.dropout, cfg.input_layernorm)
    log.info("Split   : %.0f%% test (by unique input)", 100 * cfg.test_input_fraction)
    log.info("Control : %s (seed=%d)",
             "ON" if cfg.run_control_probes else "OFF", cfg.control_seed)
    log.info("Outputs : %s", cfg.mlp_outputs_dir)
    log.info("Main out: %s", cfg.main_outputs_dir)
    log.info("CUDA    : %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            log.info("GPU     : %s (%.0f GB)",
                     torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_memory / 1e9)
        except Exception:
            pass
    log.info("=" * 72)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------
def phase_extract(cfg: MLPDecoderConfig) -> None:
    from .activations import extract_for_model
    log = logging.getLogger("mlp_decoder.run_all")
    t0 = time.time()
    for mk in cfg.models:
        log.info("[extract] %s", mk)
        try:
            extract_for_model(mk, cfg)
        except Exception:
            log.exception("Extraction failed for %s -- aborting", mk)
            raise
        # Free GPU between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log.info("Phase 1 (extract) done in %.1fs", time.time() - t0)


def phase_train(cfg: MLPDecoderConfig) -> None:
    from .train_probes import train_all_probes_for_model
    log = logging.getLogger("mlp_decoder.run_all")
    t0 = time.time()
    for mk in cfg.models:
        cache_path = cfg.model_cache_path(mk)
        if not cache_path.exists():
            log.warning("No activation cache for %s at %s -- skipping",
                        mk, cache_path)
            continue
        log.info("[train] %s -- loading cache", mk)
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        train_all_probes_for_model(mk, cache, cfg)
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log.info("Phase 2 (train) done in %.1fs", time.time() - t0)


def phase_analyze(cfg: MLPDecoderConfig) -> None:
    from .analyze import write_comparison
    log = logging.getLogger("mlp_decoder.run_all")
    t0 = time.time()
    write_comparison(cfg)
    log.info("Phase 3 (analyze) done in %.1fs", time.time() - t0)


def phase_figures(cfg: MLPDecoderConfig) -> None:
    from .figures import make_all_figures
    log = logging.getLogger("mlp_decoder.run_all")
    t0 = time.time()
    paths = make_all_figures(cfg)
    log.info("Phase 4 (figures) wrote %d files in %.1fs",
             len(paths), time.time() - t0)


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
def print_final_summary(cfg: MLPDecoderConfig) -> None:
    import json
    log = logging.getLogger("mlp_decoder.run_all")
    cmp_path = cfg.comparison_path()
    if not cmp_path.exists():
        log.warning("No comparison file at %s -- skipping summary", cmp_path)
        return
    with open(cmp_path) as f:
        data = json.load(f)
    g = data.get("global", {})

    def _bar(): log.info("-" * 72)
    _bar()
    log.info("FINAL SUMMARY")
    _bar()
    log.info("Decodability threshold tau = %.2f", data["decodability_threshold"])
    log.info("Models analyzed: %d", len(data.get("per_model", {})))
    log.info("")
    log.info("SAND headline numbers (steerable but not decoded):")
    log.info("  Under logit lens                : %d cases",
             g.get("n_sand_under_logit_lens", 0))
    log.info("  Persists under MLP probe        : %d cases",
             g.get("n_sand_persists_under_mlp_probe", 0))
    log.info("  Closed by MLP probe (non-linear): %d cases",
             g.get("n_sand_closed_by_mlp_probe", 0))
    if g.get("n_sand_under_logit_lens", 0) > 0:
        frac = g["n_sand_persists_under_mlp_probe"] / g["n_sand_under_logit_lens"]
        log.info("  -> %.0f%% of SAND cases survive nonlinear decoding",
                 100 * frac)
    log.info("")
    log.info("MLP-only-decodable cases (information was nonlinearly encoded):")
    for mk, t in g.get("sand_closed_cases", []) or []:
        log.info("  %s : %s", mk, t)
    if not g.get("sand_closed_cases"):
        log.info("  (none -- all SAND cases survive the MLP probe too)")
    log.info("")
    log.info("Persists-everywhere cases (nonlinearly invisible too):")
    for mk, t in g.get("sand_persists_cases", []) or []:
        log.info("  %s : %s", mk, t)
    _bar()
    log.info("Output written to %s", cfg.mlp_outputs_dir)
    _bar()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    cfg = build_config(args)
    print_config_banner(cfg)

    if not args.skip_extract:
        phase_extract(cfg)
    else:
        logging.getLogger("mlp_decoder.run_all").info("Skipping extraction phase")

    if not args.skip_train:
        phase_train(cfg)
    else:
        logging.getLogger("mlp_decoder.run_all").info("Skipping training phase")

    if not args.skip_analyze:
        phase_analyze(cfg)
    else:
        logging.getLogger("mlp_decoder.run_all").info("Skipping analyze phase")

    if not args.skip_figures:
        phase_figures(cfg)
    else:
        logging.getLogger("mlp_decoder.run_all").info("Skipping figures phase")

    print_final_summary(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
