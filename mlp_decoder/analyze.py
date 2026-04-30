"""
Cross-decoder comparison: MLP probe vs logit lens vs tuned lens vs steering
============================================================================
Reads JSON outputs from the main pipeline (logit lens, tuned lens, steering)
and the new MLP probe results, builds a per-(model, task) comparison table
and an updated 2x2 / 2x3 / 2x4 matrix that adds the "MLP-decodable" cell.

Key questions this populates:
  1. Does the steerable-not-decodable cell shrink under MLP probing?
     (i.e. does a nonlinear decoder recover information the logit lens missed?)
  2. Selectivity test (Hewitt & Liang): is the MLP probe's success genuine
     task structure, or memorization? real_top10 - control_top10 quantifies.
  3. For each of the 14 SAND cases the paper reports, what is the gap that
     remains under each decoder strength?

Output: <mlp_outputs_dir>/decoder_comparison.json
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import MLPDecoderConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders for main-pipeline JSONs
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        logger.warning("Missing JSON: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def _logit_lens_best_top10_per_task(
    readability_data: Dict,
) -> Dict[str, float]:
    """
    For each task, return the best-layer top-10 across templates (max over
    template, max over layer).
    """
    rows = readability_data.get("readability") or []
    by_task: Dict[str, float] = {}
    for r in rows:
        t = r["task"]
        v = r.get("top_10_accuracy", 0.0)
        if t not in by_task or v > by_task[t]:
            by_task[t] = v

    # Sentiment polarity: special-case scoring, override for sentiment_flip
    polarity = readability_data.get("sentiment_polarity") or []
    best_polarity = 0.0
    for p in polarity:
        v = p.get("polarity_classification_accuracy", 0.0)
        best_polarity = max(best_polarity, v)
    if polarity:
        by_task["sentiment_flip"] = best_polarity
    return by_task


def _tuned_lens_best_top10_per_task(
    tuned_lens_data: Dict,
) -> Dict[str, float]:
    rows = tuned_lens_data.get("readability") or []
    by_task: Dict[str, float] = {}
    for r in rows:
        t = r["task"]
        v = r.get("top_10_accuracy", 0.0)
        if t not in by_task or v > by_task[t]:
            by_task[t] = v
    return by_task


def _steering_iid_per_task(iid_data) -> Dict[str, float]:
    """
    Aggregate IID steering accuracy per task. To match the main paper's
    2x2 matrix population logic (readability.py:compute_readability_
    steerability_gap), we take the **max** across templates. This is the
    same convention applied to logit-lens and tuned-lens readability.

    iid_summary.json is a list of {task, template_id, best_accuracy, ...}
    entries; each `best_accuracy` is already the max over (layer, alpha)
    for that (task, template).
    """
    def _first_present(d: Dict, keys: Tuple[str, ...]) -> Optional[float]:
        # Use explicit `is not None` checks — `or` would skip legitimate 0.0
        # accuracy entries (Python falsy semantics).
        for k in keys:
            val = d.get(k)
            if val is not None:
                return float(val)
        return None

    acc_keys = ("best_accuracy", "iid_accuracy", "best_iid_accuracy")
    by_task: Dict[str, float] = {}
    if isinstance(iid_data, list):
        accs_by_task: Dict[str, List[float]] = defaultdict(list)
        for entry in iid_data:
            if not isinstance(entry, dict):
                continue
            t = entry.get("task")
            v = _first_present(entry, acc_keys)
            if t is None or v is None:
                continue
            accs_by_task[t].append(v)
        by_task = {t: float(max(vs)) for t, vs in accs_by_task.items()}
    elif isinstance(iid_data, dict):
        for k, v in iid_data.items():
            if isinstance(v, dict):
                accs = []
                for tid, payload in v.items():
                    if isinstance(payload, dict):
                        val = _first_present(payload, acc_keys)
                        if val is not None:
                            accs.append(val)
                if accs:
                    by_task[k] = float(max(accs))
            elif isinstance(v, (int, float)):
                by_task[k] = float(v)
    return by_task


# ---------------------------------------------------------------------------
# Per-model comparison
# ---------------------------------------------------------------------------
def build_comparison_for_model(
    model_key: str,
    cfg: MLPDecoderConfig,
) -> Dict:
    """
    Build a dict of per-task decoder comparisons for one model.

    Returns:
      {
        "meta": {...},
        "per_task": {
           task: {
             "logit_lens_top10": ...,
             "tuned_lens_top10": ...,
             "mlp_probe_real_top10": ...,
             "mlp_probe_control_top10": ...,
             "mlp_probe_selectivity_top10": ...,
             "mlp_probe_best_layer": ...,
             "fv_iid_accuracy": ...,
             "decodable_logit_lens": bool,
             "decodable_tuned_lens": bool,
             "decodable_mlp_probe": bool,
             "steerable": bool,
             "matrix_cell_2x2": str,    # "both_succeed" | "decodable_not_steerable" | ...
             "matrix_cell_2x3": str,    # adds MLP layer
           }
        },
        "matrix_summary_2x4": {...}
      }
    """
    rd = cfg.main_pipeline_results_for(model_key)
    rdb = _load_json(rd / "readability_results.json") or {}
    tld = _load_json(rd / "tuned_lens_results.json") or {}
    iid = _load_json(rd / "iid_summary.json") or {}
    mlp = _load_json(cfg.model_results_path(model_key)) or {}

    logit_top10 = _logit_lens_best_top10_per_task(rdb) if rdb else {}
    tuned_top10 = _tuned_lens_best_top10_per_task(tld) if tld else {}
    iid_acc = _steering_iid_per_task(iid) if iid else {}
    mlp_summary = (mlp or {}).get("per_task_summary", {})

    tau = cfg.decodability_threshold

    per_task: Dict[str, Dict] = {}
    all_tasks = set(logit_top10) | set(tuned_top10) | set(iid_acc) | set(mlp_summary)
    for task in sorted(all_tasks):
        ll = logit_top10.get(task, 0.0)
        tl = tuned_top10.get(task, 0.0)
        ms = mlp_summary.get(task, {})
        mlp_real = float(ms.get("best_top10_real", 0.0))
        mlp_ctrl = float(ms.get("best_top10_control", 0.0))
        mlp_sel = float(ms.get("best_top10_selectivity", 0.0))
        fv = iid_acc.get(task, 0.0)

        decodable_ll = ll >= tau
        decodable_tl = tl >= tau
        decodable_mlp = (mlp_real >= tau) and (mlp_sel >= tau / 2)
        # Selectivity gate: require mlp_sel >= tau/2 (e.g. 0.05 for tau=0.10)
        # so we don't claim "MLP decodable" when the probe just memorized.

        steerable = fv >= tau

        cell_2x2 = _classify(decodable_ll, steerable)
        cell_2x4 = _classify_2x4(decodable_ll, decodable_tl, decodable_mlp, steerable)

        per_task[task] = {
            "logit_lens_top10": ll,
            "tuned_lens_top10": tl,
            "mlp_probe_real_top10": mlp_real,
            "mlp_probe_control_top10": mlp_ctrl,
            "mlp_probe_selectivity_top10": mlp_sel,
            "mlp_probe_best_layer": int(ms.get("best_top10_layer_1idx", -1))
                if ms else -1,
            "mlp_probe_best_selectivity_layer": int(
                ms.get("best_selectivity_layer_1idx", -1)
            ) if ms else -1,
            "fv_iid_accuracy": fv,
            "decodable_logit_lens": decodable_ll,
            "decodable_tuned_lens": decodable_tl,
            "decodable_mlp_probe": decodable_mlp,
            "steerable": steerable,
            "matrix_cell_2x2_logit_lens": cell_2x2,
            "matrix_cell_2x4": cell_2x4,
            "logit_to_mlp_gain": mlp_real - ll,
            "tuned_to_mlp_gain": mlp_real - tl,
            "fv_to_mlp_gap": fv - mlp_real,
        }

    # Build per-cell counts
    cell_counts_2x2 = defaultdict(int)
    cell_counts_2x4 = defaultdict(int)
    for t, row in per_task.items():
        cell_counts_2x2[row["matrix_cell_2x2_logit_lens"]] += 1
        cell_counts_2x4[row["matrix_cell_2x4"]] += 1

    # Identify the SAND cases under logit lens; report what MLP says about each
    sand_under_ll = [
        t for t, row in per_task.items()
        if row["matrix_cell_2x2_logit_lens"] == "steerable_not_decodable"
    ]
    sand_under_mlp = [
        t for t in sand_under_ll
        if not per_task[t]["decodable_mlp_probe"]
    ]
    sand_closed_by_mlp = [
        t for t in sand_under_ll
        if per_task[t]["decodable_mlp_probe"]
    ]

    return {
        "meta": {
            "model_key": model_key,
            "display_name": (mlp.get("meta", {}) or {}).get("display_name", model_key),
            "family": (mlp.get("meta", {}) or {}).get("family"),
            "decodability_threshold": tau,
        },
        "aggregate": (mlp or {}).get("aggregate", {}),
        "per_task": per_task,
        "matrix_counts_2x2_logit_lens": dict(cell_counts_2x2),
        "matrix_counts_2x4": dict(cell_counts_2x4),
        "sand_summary": {
            "sand_under_logit_lens": sand_under_ll,
            "sand_persists_under_mlp_probe": sand_under_mlp,
            "sand_closed_by_mlp_probe": sand_closed_by_mlp,
            "n_sand_under_logit_lens": len(sand_under_ll),
            "n_sand_persists_under_mlp_probe": len(sand_under_mlp),
            "n_sand_closed_by_mlp_probe": len(sand_closed_by_mlp),
        },
    }


def _classify(decodable: bool, steerable: bool) -> str:
    if decodable and steerable:
        return "both_succeed"
    if decodable and not steerable:
        return "decodable_not_steerable"
    if not decodable and steerable:
        return "steerable_not_decodable"
    return "both_fail"


def _classify_2x4(
    decodable_ll: bool,
    decodable_tl: bool,
    decodable_mlp: bool,
    steerable: bool,
) -> str:
    """
    Stratify by which decoder first uncovers the answer. Buckets are
    exhaustive and mutually exclusive given the four boolean inputs.

    - all_fail: nobody decodes, FV doesn't steer
    - steerable_invisible_to_all_decoders: FV steers, no decoder reveals info
    - logit_lens_decodable: logit lens decodes (the easy case;
      tuned lens / MLP may also decode but logit lens already suffices)
    - tuned_lens_only_decodable: logit lens fails, tuned lens decodes
      (MLP may or may not also decode -- the tuned lens already suffices)
    - mlp_only_decodable: only the nonlinear MLP probe decodes -- the
      load-bearing case for the "FVs ride nonlinear subspaces" story
    """
    any_decoder = decodable_ll or decodable_tl or decodable_mlp
    if not any_decoder:
        return ("steerable_invisible_to_all_decoders"
                if steerable else "all_fail")
    if decodable_ll:
        return "logit_lens_decodable"
    if decodable_tl:
        return "tuned_lens_only_decodable"
    # any_decoder is True and ll/tl are both False, so mlp must be True.
    return "mlp_only_decodable"


# ---------------------------------------------------------------------------
# Cross-model summary
# ---------------------------------------------------------------------------
def build_cross_model_summary(cfg: MLPDecoderConfig) -> Dict:
    """
    Aggregate across all models. Produces:
      - Global SAND stats (the headline number for the paper)
      - Per-cell counts of the 2x4 matrix
      - List of cases where MLP closes the SAND gap (these are the
        "FVs operate through nonlinear subspaces" cases)
      - List of cases where MLP also fails (these are the
        "information genuinely absent at intermediate layers" cases --
        the paper's claim survives the strongest test)
    """
    per_model: Dict[str, Dict] = {}
    for mk in cfg.models:
        try:
            per_model[mk] = build_comparison_for_model(mk, cfg)
        except Exception:
            logger.exception("Failed to build comparison for %s", mk)

    # Aggregate counts
    total_2x2 = defaultdict(int)
    total_2x4 = defaultdict(int)
    sand_total = 0
    sand_persists_total = 0
    sand_closed_total = 0
    sand_persists_cases: List[Tuple[str, str]] = []
    sand_closed_cases: List[Tuple[str, str]] = []

    for mk, mdat in per_model.items():
        for k, v in mdat["matrix_counts_2x2_logit_lens"].items():
            total_2x2[k] += v
        for k, v in mdat["matrix_counts_2x4"].items():
            total_2x4[k] += v
        sand_total += mdat["sand_summary"]["n_sand_under_logit_lens"]
        sand_persists_total += mdat["sand_summary"]["n_sand_persists_under_mlp_probe"]
        sand_closed_total += mdat["sand_summary"]["n_sand_closed_by_mlp_probe"]
        for t in mdat["sand_summary"]["sand_persists_under_mlp_probe"]:
            sand_persists_cases.append((mk, t))
        for t in mdat["sand_summary"]["sand_closed_by_mlp_probe"]:
            sand_closed_cases.append((mk, t))

    # SAND headline numbers under each decoder
    return {
        "decodability_threshold": cfg.decodability_threshold,
        "models": cfg.models,
        "per_model": per_model,
        "global": {
            "matrix_counts_2x2_logit_lens": dict(total_2x2),
            "matrix_counts_2x4": dict(total_2x4),
            "n_sand_under_logit_lens": sand_total,
            "n_sand_persists_under_mlp_probe": sand_persists_total,
            "n_sand_closed_by_mlp_probe": sand_closed_total,
            "sand_persists_cases": [list(p) for p in sand_persists_cases],
            "sand_closed_cases": [list(p) for p in sand_closed_cases],
            "fraction_sand_persists_under_mlp": (
                sand_persists_total / max(1, sand_total)
            ),
        },
    }


def write_comparison(cfg: MLPDecoderConfig) -> Dict:
    out = build_cross_model_summary(cfg)
    path = cfg.comparison_path()
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Wrote decoder comparison to %s", path)
    return out
