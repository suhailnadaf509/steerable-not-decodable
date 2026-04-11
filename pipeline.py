"""
Pipeline Orchestration (per EXPERIMENT_REDESIGN_SPEC.md)
========================================================
Coordinates the full experimental pipeline with incremental execution,
caching, and clear stage separation.

Stages (spec ordering):
  1. baseline          -- Zero-shot + few-shot baselines (before extraction)
  2. extract           -- Generate prompts, extract function vectors
  3. probe_activations -- Cache zero-shot activations for probing (after extraction)
  4. steer             -- Evaluate steering accuracy (IID + OOD)
  5. analyze           -- Geometric analysis, dissociation detection
  6. mechanistic       -- Activation patching (IID-gated)
  7. probe/readability -- Logit lens readability + FV vocabulary projection
  8. figures           -- Generate publication figures
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .config import ExperimentConfig, MODEL_REGISTRY
from .tasks import get_tasks, validate_all_tasks, print_task_summary
from .data import generate_all_prompts, validate_prompts, save_prompts
from .extraction import FVExtractor, save_fvs, load_fvs, fv_statistics, FVCollection
from .models import ModelWrapper
from .steering import (
    SteeringEvaluator,
    BaselineEvaluator,
    print_iid_summary,
    save_steering_results,
    save_baseline_results,
    load_steering_results,
    IIDSummary,
)
from .analysis import GeometricAnalyzer, save_analysis_results
from .mechanistic import run_mechanistic_analysis, save_mechanistic_results
from .visualization import generate_all_figures

logger = logging.getLogger(__name__)


def _banner(text: str, char: str = "="):
    line = char * 70
    logger.info(line)
    logger.info(text)
    logger.info(line)
    print(f"\n{line}")
    print(text)
    print(line)


def run_pipeline(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the experiment pipeline according to config.stages.

    Each stage checks for cached results before re-running.
    Results are saved after each stage for incremental execution.
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _banner(f"FV CROSS-TEMPLATE TRANSFER PIPELINE\n"
            f"Model: {config.model_key} ({config.model_spec.display_name})\n"
            f"Start: {timestamp}\n"
            f"Stages: {config.stages}")

    # Validate config
    warnings = config.validate()
    for w in warnings:
        logger.warning("Config warning: %s", w)

    # Save config
    config.save()

    # Validate tasks
    task_warnings = validate_all_tasks()
    print_task_summary()

    stages = set(config.stages)
    timing: Dict[str, float] = {}
    model: Optional[ModelWrapper] = None

    # Track results across stages
    fvs: Optional[FVCollection] = None
    steering_results: Optional[Dict] = None
    iid_summaries: Optional[List[IIDSummary]] = None
    geometric_results: Optional[Dict] = None
    prompts = None

    # =====================================================================
    # STAGE 1: BASELINE (spec §4.2 + §7.2 Control C)
    # Zero-shot + few-shot baselines BEFORE extraction
    # =====================================================================
    if "baseline" in stages:
        _banner("STAGE 1: BASELINE EVALUATION (zero-shot + few-shot)")
        t0 = time.time()

        baseline_cache = config.results_dir / "baseline_results.json"
        if baseline_cache.exists() and "extract" not in stages:
            logger.info("Loading cached baseline results")
        else:
            prompts = generate_all_prompts(config)
            if model is None:
                model = ModelWrapper(config=config)

            evaluator = BaselineEvaluator(model, config)
            baseline_results, baseline_summary = evaluator.evaluate_all_baselines(prompts)
            save_baseline_results(baseline_results, config.results_dir)

            # Print summary
            print("\nBaseline accuracy summary:")
            for r in baseline_results:
                print(f"  {r.task:<20} {r.template_id:<6} {r.mode:<12} {r.accuracy:.3f}")

        timing["baseline"] = time.time() - t0

    # =====================================================================
    # STAGE 2: EXTRACT
    # =====================================================================
    if "extract" in stages:
        _banner("STAGE 2: EXTRACTION")
        t0 = time.time()

        # Check cache
        fv_cache_path = config.cache_dir / f"fvs_{config.config_hash()}"
        if fv_cache_path.with_suffix(".pt").exists():
            logger.info("Loading cached FVs from %s", fv_cache_path)
            fvs = load_fvs(fv_cache_path)
        else:
            # Generate prompts
            if prompts is None:
                prompts = generate_all_prompts(config)
            counts = validate_prompts(prompts)
            logger.info("Generated %d prompts", counts["total"])
            save_prompts(prompts, config.results_dir / "prompt_summary.json")

            # Load model
            if model is None:
                model = ModelWrapper(config=config)

            # Extract
            extractor = FVExtractor(model, config)
            all_fvs = extractor.extract_all(prompts, methods=["mean_diff"])
            fvs = all_fvs["mean_diff"]

            # Save
            save_fvs(fvs, fv_cache_path)

        stats = fv_statistics(fvs)
        logger.info("FV stats: %s", json.dumps(stats, indent=2))

        timing["extract"] = time.time() - t0

    # =====================================================================
    # STAGE 3: PROBE ACTIVATIONS (spec §5.6.3)
    # Cache zero-shot activations for probing BEFORE steering
    # =====================================================================
    if "probe_activations" in stages:
        _banner("STAGE 3: PROBE ACTIVATION CACHING")
        t0 = time.time()

        probe_cache_dir = config.cache_dir / "activations_probe"
        probe_cache_dir.mkdir(parents=True, exist_ok=True)

        if any(probe_cache_dir.glob("*.pt")):
            logger.info("Probe activation cache found at %s", probe_cache_dir)
        else:
            if model is None:
                model = ModelWrapper(config=config)

            tasks = get_tasks(config.task_names)
            layers_0idx = model.extraction_layers_0indexed()

            for task_name, task_spec in tasks.items():
                for tid in task_spec.template_ids:
                    template_str = task_spec.templates[tid]
                    bare_prompts = [
                        template_str.replace("{X}", inp)
                        for inp, _ in task_spec.pairs
                    ]

                    logger.info(
                        "Caching probe activations for %s/%s (%d examples)",
                        task_name, tid, len(bare_prompts),
                    )
                    acts = model.get_activations(
                        bare_prompts, layers_0idx,
                        batch_size=config.ops.extraction_batch_size,
                    )["resid_post"]

                    # Save per (task, template)
                    cache_path = probe_cache_dir / f"{task_name}_{tid}.pt"
                    torch.save(acts, cache_path)

            logger.info("Probe activations cached to %s", probe_cache_dir)

        timing["probe_activations"] = time.time() - t0

    # =====================================================================
    # STAGE 4: STEER
    # =====================================================================
    if "steer" in stages:
        _banner("STAGE 4: STEERING EVALUATION")
        t0 = time.time()

        # Check cache
        steer_cache = config.results_dir / "steering_results.json"
        iid_cache = config.results_dir / "iid_summary.json"

        if steer_cache.exists() and iid_cache.exists() and "extract" not in stages:
            logger.info("Loading cached steering results")
            steering_results, iid_summaries = load_steering_results(config.results_dir)
        else:
            # Need prompts and FVs
            if fvs is None:
                fv_cache_path = config.cache_dir / f"fvs_{config.config_hash()}"
                fvs = load_fvs(fv_cache_path)

            if prompts is None:
                prompts = generate_all_prompts(config)

            if model is None:
                model = ModelWrapper(config=config)

            evaluator = SteeringEvaluator(model, config)
            steering_results, iid_summaries, all_sr = evaluator.evaluate_all(
                fvs, prompts, save_predictions=False,
            )

            save_steering_results(steering_results, iid_summaries, config.results_dir)

        # CRITICAL: Print IID summary FIRST
        if iid_summaries:
            print_iid_summary(iid_summaries, config.science.iid_accuracy_threshold)

        timing["steer"] = time.time() - t0

    # =====================================================================
    # STAGE 5: ANALYZE
    # =====================================================================
    if "analyze" in stages:
        _banner("STAGE 5: GEOMETRIC ANALYSIS")
        t0 = time.time()

        if fvs is None:
            fv_cache_path = config.cache_dir / f"fvs_{config.config_hash()}"
            fvs = load_fvs(fv_cache_path)

        if steering_results is None:
            steering_results, iid_summaries = load_steering_results(config.results_dir)

        analyzer = GeometricAnalyzer(config)
        geometric_results = analyzer.run_all(fvs, steering_results, iid_summaries)
        save_analysis_results(geometric_results, config.results_dir)

        # Print summary
        corr = geometric_results.get("correlation", {})
        pooled = corr.get("pooled", {})
        summary = corr.get("summary", {})
        print(f"\nPooled alignment-transfer correlation: "
              f"r={pooled.get('pearson_r', 0):.3f}, "
              f"p={pooled.get('p_value', 1):.2e}")
        print(f"Mean within-task r: {summary.get('mean_within_task_r', 'N/A')}")

        per_task = corr.get("per_task", {})
        for task_name, td in per_task.items():
            print(f"  {task_name}: r={td.get('pearson_r', 0):.3f}")

        for warning in corr.get("simpson_paradox_warnings", []):
            print(f"  WARNING: {warning}")

        dsummary = geometric_results.get("dissociation_summary", {})
        print(f"\nDissociation cases: {dsummary.get('n_dissociation_total', 0)} total")
        print(f"  Threshold type: {dsummary.get('threshold_type', 'unknown')}")
        print(f"  IID-viable: {dsummary.get('n_dissociation_iid_viable', 0)}")
        print(f"  IID-non-viable: {dsummary.get('n_dissociation_iid_non_viable', 0)}")
        if dsummary.get("note"):
            print(f"  Note: {dsummary['note']}")

        # Style analysis
        style = geometric_results.get("style_analysis", {})
        if style.get("within_style_mean") is not None:
            print(f"\nStyle analysis:")
            print(f"  Within-style mean transfer: {style['within_style_mean']:.3f}")
            print(f"  Across-style mean transfer: {style['across_style_mean']:.3f}")
            if "style_effect" in style:
                print(f"  {style['style_effect']}")

        timing["analyze"] = time.time() - t0

    # =====================================================================
    # STAGE 6+7: MECHANISTIC + READABILITY
    # =====================================================================
    if "mechanistic" in stages or "probe" in stages or "readability" in stages:
        _banner("STAGE 6/7: MECHANISTIC + READABILITY")
        t0 = time.time()

        if model is None:
            model = ModelWrapper(config=config)

        if fvs is None:
            fv_cache_path = config.cache_dir / f"fvs_{config.config_hash()}"
            fvs = load_fvs(fv_cache_path)

        if steering_results is None:
            steering_results, iid_summaries = load_steering_results(config.results_dir)

        if geometric_results is None:
            geo_path = config.results_dir / "geometric_analysis.json"
            if geo_path.exists():
                with open(geo_path) as f:
                    geometric_results = json.load(f)
            else:
                geometric_results = {}

        mech_results = run_mechanistic_analysis(
            model, config, fvs, steering_results,
            iid_summaries or [], geometric_results,
        )
        save_mechanistic_results(mech_results, config.results_dir)

        timing["mechanistic"] = time.time() - t0

    # =====================================================================
    # STAGE 7.5: TUNED LENS (calibration of logit lens findings)
    # =====================================================================
    if "tuned_lens" in stages:
        _banner("STAGE 7.5: TUNED LENS ANALYSIS")
        t0 = time.time()

        if model is None:
            model = ModelWrapper(config=config)

        if fvs is None:
            fv_cache_path = config.cache_dir / f"fvs_{config.config_hash()}"
            fvs = load_fvs(fv_cache_path)

        if steering_results is None or iid_summaries is None:
            steering_results, iid_summaries = load_steering_results(config.results_dir)

        from .tuned_lens import run_tuned_lens_analysis, save_tuned_lens_results

        tuned_lens_results = run_tuned_lens_analysis(
            model, config, fvs, iid_summaries or [],
        )
        save_tuned_lens_results(tuned_lens_results, config.results_dir)

        timing["tuned_lens"] = time.time() - t0

    # =====================================================================
    # STAGE 8: FIGURES
    # =====================================================================
    if "figures" in stages:
        _banner("STAGE 8: FIGURE GENERATION")
        t0 = time.time()

        generate_all_figures(config.results_dir, config.figures_dir)

        timing["figures"] = time.time() - t0

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    total_time = time.time() - start_time

    _banner("PIPELINE COMPLETE")
    print(f"\nTiming breakdown:")
    for stage, t in timing.items():
        print(f"  {stage:<20} {t/60:.1f} min")
    print(f"  {'TOTAL':<20} {total_time/60:.1f} min")

    print(f"\nOutputs: {config.run_dir}")
    print(f"  Results:  {config.results_dir}")
    print(f"  Figures:  {config.figures_dir}")
    print(f"  Cache:    {config.cache_dir}")

    # Clean up model to free GPU memory (important for multi-model runs)
    if model is not None:
        model.unload()
        model = None

    return {
        "timing": timing,
        "total_minutes": total_time / 60,
        "model": config.model_key,
        "stages": config.stages,
    }


def run_all_models(
    base_config: ExperimentConfig,
    model_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the pipeline sequentially for multiple models.

    Properly unloads each model before loading the next to stay within
    GPU memory limits.  Reports per-model and total timing.

    Args:
        base_config: Template config (model_key will be overridden per model).
        model_keys: List of model keys to run.  None = all in MODEL_REGISTRY.

    Returns:
        Dict with per-model results and aggregate timing.
    """
    if model_keys is None:
        model_keys = list(MODEL_REGISTRY.keys())

    total_start = time.time()
    all_results: Dict[str, Any] = {}

    _banner(
        f"MULTI-MODEL PIPELINE\n"
        f"Models: {model_keys}\n"
        f"Stages: {base_config.stages}\n"
        f"Total models: {len(model_keys)}"
    )

    for i, model_key in enumerate(model_keys, 1):
        _banner(f"MODEL {i}/{len(model_keys)}: {model_key}", char="-")

        # Create per-model config (override model_key only)
        from dataclasses import replace
        model_config = replace(base_config, model_key=model_key)

        try:
            result = run_pipeline(model_config)
            all_results[model_key] = result
        except Exception as e:
            logger.error("Pipeline failed for %s: %s", model_key, e, exc_info=True)
            all_results[model_key] = {"error": str(e)}

        # Force GPU cleanup between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info(
                "GPU memory between models: %.1f GB allocated",
                torch.cuda.memory_allocated() / 1e9,
            )

    total_time = time.time() - total_start

    _banner("ALL MODELS COMPLETE")
    print(f"\nPer-model timing:")
    for mk, r in all_results.items():
        if isinstance(r, dict) and "total_minutes" in r:
            print(f"  {mk:<30} {r['total_minutes']:.1f} min")
        else:
            print(f"  {mk:<30} FAILED")
    print(f"  {'TOTAL':<30} {total_time / 60:.1f} min")

    return {
        "per_model": all_results,
        "total_minutes": total_time / 60,
        "models": model_keys,
    }
