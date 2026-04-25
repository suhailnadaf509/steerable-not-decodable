"""
Top-level: train MLP probes for every (model, task, layer)
==========================================================
Loads cached activations per model, trains one real probe and one
Hewitt & Liang control probe at each (task, layer), saves a JSON of results.

Sequential per layer/task -- but each MLP fits in the GPU and trains in ~1s
on H200. The bulk of wall-clock time is data movement, so we keep the entire
model's activations on GPU once loaded.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .config import MLPDecoderConfig
from .probe import (
    LayerComparison,
    ProbeResult,
    train_probes_for_task_layer,
)

logger = logging.getLogger(__name__)


def _summarize_per_task(comparisons: List[LayerComparison]) -> Dict:
    """
    Compute best-layer summary per task across all layers.
    """
    by_task: Dict[str, List[LayerComparison]] = {}
    for c in comparisons:
        by_task.setdefault(c.task, []).append(c)

    summary: Dict = {}
    for task, layer_comps in by_task.items():
        # Best layer = layer with highest real_test_top10
        best = max(layer_comps, key=lambda c: c.real_test_top10)
        # Best selectivity layer (might differ -- this tells us where the
        # probe is genuinely using task structure, not memorizing)
        best_sel = max(layer_comps, key=lambda c: c.selectivity_top10)
        # All-layer summary
        summary[task] = {
            "best_top10_layer_1idx": best.layer_1idx,
            "best_top10_real": best.real_test_top10,
            "best_top10_control": best.control_test_top10,
            "best_top10_selectivity": best.selectivity_top10,
            "best_selectivity_layer_1idx": best_sel.layer_1idx,
            "best_selectivity_top10": best_sel.selectivity_top10,
            "best_selectivity_real_top10": best_sel.real_test_top10,
            # Layer-wise list (for plotting)
            "layers_1idx": [c.layer_1idx for c in sorted(layer_comps, key=lambda c: c.layer_1idx)],
            "real_top10_by_layer": [
                c.real_test_top10 for c in sorted(layer_comps, key=lambda c: c.layer_1idx)
            ],
            "control_top10_by_layer": [
                c.control_test_top10 for c in sorted(layer_comps, key=lambda c: c.layer_1idx)
            ],
            "selectivity_top10_by_layer": [
                c.selectivity_top10 for c in sorted(layer_comps, key=lambda c: c.layer_1idx)
            ],
        }
    return summary


def train_all_probes_for_model(
    model_key: str,
    cache: Dict,
    cfg: MLPDecoderConfig,
) -> Dict:
    """
    Run probe training for every (task, layer) of one model. Persists the
    full per-(task, layer) results plus a per-task best-layer summary to:
        <mlp_outputs_dir>/<model_key>/mlp_probe_results.json

    Returns the result dict (also saved to disk).
    """
    meta = cache["__meta__"]
    layers_1idx = meta["layers_1idx"]
    vocab_size = int(meta["vocab_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training probes on %s, vocab=%d, layers=%s",
                device, vocab_size, layers_1idx)

    all_real: List[ProbeResult] = []
    all_ctrl: List[ProbeResult] = []
    all_cmp: List[LayerComparison] = []

    task_names = [k for k in cache.keys() if k != "__meta__"]
    n_total = len(task_names) * len(layers_1idx)
    done = 0
    t0 = time.time()
    last_log = t0

    for task_name in task_names:
        task_cache = cache[task_name]
        # Pre-flight: skip tasks whose templates lack data
        if not task_cache:
            logger.warning("No template data for %s, skipping", task_name)
            continue

        # Quick sanity: does at least one template have at least one layer?
        any_layer_present = any(
            layer_1idx in payload["acts_by_layer"]
            for payload in task_cache.values()
            for layer_1idx in layers_1idx
        )
        if not any_layer_present:
            logger.warning("No activations cached for %s -- skipping", task_name)
            continue

        for layer_1idx in layers_1idx:
            try:
                real, ctrl, cmp = train_probes_for_task_layer(
                    task_name=task_name,
                    layer_1idx=layer_1idx,
                    task_cache=task_cache,
                    cfg=cfg,
                    vocab_size=vocab_size,
                    device=device,
                )
                all_real.append(real)
                all_ctrl.append(ctrl)
                all_cmp.append(cmp)
            except Exception:
                logger.exception(
                    "Probe training failed for %s/%s/L%d",
                    model_key, task_name, layer_1idx,
                )
            done += 1
            now = time.time()
            if now - last_log > 30:
                rate = done / max(1e-3, now - t0)
                eta = (n_total - done) / max(1e-3, rate)
                logger.info(
                    "  [%s] probes %d/%d (%.1f/sec) ETA %.0fs -- last (%s,L%d) real=%.3f ctrl=%.3f sel=%.3f",
                    model_key, done, n_total, rate, eta,
                    task_name, layer_1idx,
                    cmp.real_test_top10,
                    cmp.control_test_top10,
                    cmp.selectivity_top10,
                )
                last_log = now

    elapsed = time.time() - t0
    logger.info("Probe training for %s done: %d (task,layer) probes in %.1fs",
                model_key, len(all_cmp), elapsed)

    summary = _summarize_per_task(all_cmp)

    # Aggregate stats
    real_top10s = [c.real_test_top10 for c in all_cmp]
    ctrl_top10s = [c.control_test_top10 for c in all_cmp]
    sel_top10s = [c.selectivity_top10 for c in all_cmp]

    output: Dict = {
        "meta": {
            "model_key": model_key,
            "display_name": meta.get("display_name", model_key),
            "family": meta.get("family"),
            "n_layers": meta["n_layers"],
            "d_model": meta["d_model"],
            "vocab_size": vocab_size,
            "layers_1idx": layers_1idx,
            "n_tasks": len([k for k in cache.keys() if k != "__meta__"]),
            "training_seconds": elapsed,
            "probe_config": {
                "hidden_dim": cfg.hidden_dim,
                "dropout": cfg.dropout,
                "input_layernorm": cfg.input_layernorm,
                "n_epochs": cfg.n_epochs,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "test_input_fraction": cfg.test_input_fraction,
                "seed": cfg.seed,
                "control_seed": cfg.control_seed,
                "full_batch": cfg.full_batch,
            },
        },
        "aggregate": {
            "n_probes": len(all_cmp),
            "mean_real_top10": float(sum(real_top10s) / max(1, len(real_top10s))),
            "mean_control_top10": float(sum(ctrl_top10s) / max(1, len(ctrl_top10s))),
            "mean_selectivity_top10": float(sum(sel_top10s) / max(1, len(sel_top10s))),
        },
        "per_task_summary": summary,
        "per_layer_results": [
            {**asdict(c)} for c in all_cmp
        ],
        "real_probe_results": [asdict(r) for r in all_real],
        "control_probe_results": [asdict(r) for r in all_ctrl],
    }

    out_path = cfg.model_results_path(model_key)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote MLP probe results to %s", out_path)
    return output
