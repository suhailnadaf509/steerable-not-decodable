#!/usr/bin/env python3
"""
CLI Entry Point
===============
Run the cross-template function vector transfer pipeline.

Usage:
    python -m fv_cross_template.run_pipeline --model llama-3.1-8b-base --tasks all
    python -m fv_cross_template.run_pipeline --model gemma-2-9b-base --stages extract,steer
    python -m fv_cross_template.run_pipeline --model llama-3.1-8b-base --stages mechanistic
    python -m fv_cross_template.run_pipeline --stages figures --output-dir outputs/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Template Function Vector Transfer Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline on Llama-3.1-8B Base:
    python -m fv_cross_template.run_pipeline --model llama-3.1-8b-base

  Extract and steer only:
    python -m fv_cross_template.run_pipeline --model llama-3.1-8b-base --stages extract,steer

  Mechanistic analysis only (uses cached results):
    python -m fv_cross_template.run_pipeline --model llama-3.1-8b-base --stages mechanistic

  Probing only:
    python -m fv_cross_template.run_pipeline --model llama-3.1-8b-base --stages probe

  Generate figures from saved results:
    python -m fv_cross_template.run_pipeline --stages figures

  Specific tasks only:
    python -m fv_cross_template.run_pipeline --model llama-3.1-8b-base --tasks antonym,synonym,past_tense

  Custom IID threshold:
    python -m fv_cross_template.run_pipeline --model llama-3.1-8b-base --iid-threshold 0.15
        """,
    )

    # -- Model selection --
    from .config import MODEL_REGISTRY, GPU_PROFILES
    parser.add_argument(
        "--model", type=str, default="llama-3.1-8b-base",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to use (default: llama-3.1-8b-base)",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Run pipeline for ALL models sequentially (overrides --model)",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of model keys to run (e.g. llama-3.1-8b-base,gemma-2-9b-base)",
    )

    # -- Task selection --
    from .tasks import TASK_REGISTRY
    parser.add_argument(
        "--tasks", type=str, default="all",
        help="Comma-separated task names, or 'all' (default: all)",
    )

    # -- Stage selection --
    parser.add_argument(
        "--stages", type=str,
        default="baseline,extract,probe_activations,steer,analyze,mechanistic,readability,tuned_lens,figures",
        help="Comma-separated stages: baseline,extract,probe_activations,steer,analyze,mechanistic,readability,tuned_lens,figures (probe is alias for readability)",
    )

    # -- Scientific parameters --
    parser.add_argument(
        "--iid-threshold", type=float, default=0.10,
        help="IID accuracy threshold for causal analysis (default: 0.10)",
    )
    parser.add_argument(
        "--data-derived-thresholds", action="store_true",
        help="Use data-derived dissociation thresholds instead of fixed",
    )

    # -- Operational parameters --
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output root directory (default: outputs/)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Steering evaluation batch size (default: 16)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    # -- Performance --
    parser.add_argument(
        "--gpu-profile", type=str, default=None,
        choices=list(GPU_PROFILES.keys()),
        help="GPU memory profile for batch size tuning (e.g. l4, a100-40, a100-80)",
    )
    parser.add_argument(
        "--strength-tile-factor", type=int, default=None,
        help="Batch N strengths into single generation call (default: set by gpu-profile or 1)",
    )
    # NOTE: --skip-ood-on-zero-iid removed — all OOD evaluations now run
    # unconditionally to ensure complete results.

    args = parser.parse_args()

    # -- Setup --
    setup_logging(args.log_level)

    # -- Build config --
    from .config import ExperimentConfig, ScientificParams, OperationalParams, apply_gpu_profile

    task_names = None if args.tasks == "all" else args.tasks.split(",")
    stages = args.stages.split(",")

    science = ScientificParams(
        iid_accuracy_threshold=args.iid_threshold,
        use_data_derived_thresholds=args.data_derived_thresholds,
    )

    ops = OperationalParams(
        steering_batch_size=args.batch_size,
    )

    # Apply GPU profile (sets batch sizes and tile factor for target GPU)
    if args.gpu_profile:
        apply_gpu_profile(ops, args.gpu_profile)
    else:
        # Auto-detect GPU and apply matching profile
        from .config import auto_detect_gpu_profile
        detected_profile = auto_detect_gpu_profile()
        if detected_profile:
            logging.getLogger(__name__).info(
                "Auto-detected GPU profile: %s", detected_profile
            )
            apply_gpu_profile(ops, detected_profile)

    # Override strength tile factor if explicitly set
    if args.strength_tile_factor is not None:
        ops.strength_tile_factor = args.strength_tile_factor

    config_kwargs = {
        "model_key": args.model,
        "task_names": task_names,
        "stages": stages,
        "science": science,
        "ops": ops,
        "seed": args.seed,
    }

    if args.output_dir:
        config_kwargs["output_root"] = Path(args.output_dir)

    config = ExperimentConfig(**config_kwargs)

    # -- Run pipeline --
    from .pipeline import run_pipeline, run_all_models

    # Multi-model mode
    if args.all_models or args.models:
        if args.models:
            model_keys = args.models.split(",")
            # Validate
            for mk in model_keys:
                if mk not in MODEL_REGISTRY:
                    print(f"ERROR: Unknown model key '{mk}'. Available: {list(MODEL_REGISTRY.keys())}")
                    sys.exit(1)
        else:
            model_keys = list(MODEL_REGISTRY.keys())

        result = run_all_models(config, model_keys=model_keys)
    else:
        result = run_pipeline(config)

    return result


if __name__ == "__main__":
    main()
