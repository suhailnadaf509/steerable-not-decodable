"""
Geometric Analysis (per EXPERIMENT_REDESIGN_SPEC.md §5)
========================================================
Core analyses for function vector generalization:
  1. Pairwise cosine similarity between template FVs
  2. Alignment-transfer correlation (overall, per-task, per-layer)
  3. Dissociation detection (data-derived percentile thresholds)
  4. Universal Task Vector via PCA
  5. Transfer matrix (8x8 per task, spec §5.1)
  6. Within-style vs across-style transfer (spec §5.2 point 3)
  7. Hierarchical regression (spec §4.3 point 6)
  8. Permutation test for dissociation significance (spec §7.2 Control D)
  9. Random vector baseline (spec §7.2 Control A)
  10. Cross-task FV application control (spec §7.2 Control B)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from .config import ExperimentConfig
from .extraction import FVCollection, FunctionVector
from .steering import IIDSummary
from .tasks import TEMPLATE_STYLE_MAP, TemplateStyle

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    task: str
    template_a: str
    template_b: str
    layer: int
    cosine: float
    l2_distance: float
    norm_a: float
    norm_b: float


@dataclass
class DissociationCase:
    task: str
    source: str
    target: str
    layer: int
    cosine: float
    transfer_accuracy: float
    reverse_accuracy: float
    iid_source_accuracy: float  # to verify IID viability


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    v1n = v1 / (v1.norm() + 1e-8)
    v2n = v2 / (v2.norm() + 1e-8)
    return float(torch.dot(v1n.flatten(), v2n.flatten()))


class GeometricAnalyzer:
    """Runs all geometric analyses on extracted function vectors."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def compute_pairwise_alignments(
        self, fvs: FVCollection,
    ) -> List[AlignmentResult]:
        """Compute cosine similarity for all template pairs at each layer."""
        results = []
        for task in fvs:
            templates = list(fvs[task].keys())
            layers = sorted(next(iter(fvs[task].values())).keys())
            for layer in layers:
                for ta, tb in combinations(templates, 2):
                    fv_a = fvs[task][ta][layer]
                    fv_b = fvs[task][tb][layer]
                    cos = cosine_similarity(fv_a.vector, fv_b.vector)
                    l2 = float((fv_a.vector - fv_b.vector).norm())
                    results.append(AlignmentResult(
                        task=task, template_a=ta, template_b=tb,
                        layer=layer, cosine=cos, l2_distance=l2,
                        norm_a=fv_a.norm, norm_b=fv_b.norm,
                    ))
        return results

    def alignment_transfer_correlation(
        self,
        alignments: List[AlignmentResult],
        steering_results: Dict,
    ) -> Dict[str, Any]:
        """
        Compute correlation between cosine similarity and transfer accuracy.

        Reports pooled, per-task, AND per-layer correlations (spec §5.2, §7.1).
        """
        # Pooled
        cosines_all, accs_all = [], []
        # Per-task
        per_task: Dict[str, Tuple[List[float], List[float]]] = {}
        # Per-layer
        per_layer: Dict[int, Tuple[List[float], List[float]]] = {}

        for ar in alignments:
            try:
                acc_ab = steering_results[ar.task][ar.template_a][ar.template_b][ar.layer]
                acc_ba = steering_results[ar.task][ar.template_b][ar.template_a][ar.layer]
            except (KeyError, TypeError):
                continue

            for acc in [acc_ab, acc_ba]:
                cosines_all.append(ar.cosine)
                accs_all.append(acc)

                if ar.task not in per_task:
                    per_task[ar.task] = ([], [])
                per_task[ar.task][0].append(ar.cosine)
                per_task[ar.task][1].append(acc)

                if ar.layer not in per_layer:
                    per_layer[ar.layer] = ([], [])
                per_layer[ar.layer][0].append(ar.cosine)
                per_layer[ar.layer][1].append(acc)

        result: Dict[str, Any] = {
            "pooled": {}, "per_task": {}, "per_layer": {},
            "n_total": len(cosines_all),
        }

        if len(cosines_all) > 2:
            r, p = stats.pearsonr(cosines_all, accs_all)
            result["pooled"] = {
                "pearson_r": float(r), "r_squared": float(r ** 2),
                "p_value": float(p), "n_samples": len(cosines_all),
            }

        for task_name, (cos_list, acc_list) in per_task.items():
            if len(cos_list) > 2:
                r, p = stats.pearsonr(cos_list, acc_list)
                result["per_task"][task_name] = {
                    "pearson_r": float(r), "r_squared": float(r ** 2),
                    "p_value": float(p), "n_samples": len(cos_list),
                }

        for layer, (cos_list, acc_list) in per_layer.items():
            if len(cos_list) > 2:
                r, p = stats.pearsonr(cos_list, acc_list)
                result["per_layer"][layer] = {
                    "pearson_r": float(r), "r_squared": float(r ** 2),
                    "p_value": float(p), "n_samples": len(cos_list),
                }

        # Flag Simpson's paradox
        pooled_r = result["pooled"].get("pearson_r", 0)
        n_positive_within = 0
        n_negative_within = 0
        for tn, td in result["per_task"].items():
            task_r = td.get("pearson_r", 0)
            if task_r > 0.1:
                n_positive_within += 1
            if task_r < -0.1:
                n_negative_within += 1
            if (pooled_r < -0.1 and task_r > 0.1) or (pooled_r > 0.1 and task_r < -0.1):
                result.setdefault("simpson_paradox_warnings", []).append(
                    f"Task '{tn}': within-task r={task_r:.3f} vs pooled r={pooled_r:.3f} "
                    f"-- opposite signs indicate Simpson's paradox"
                )

        # Summary table per spec §4.4
        within_task_rs = [
            td["pearson_r"] for td in result["per_task"].values()
            if "pearson_r" in td
        ]
        result["summary"] = {
            "pooled_r": result["pooled"].get("pearson_r"),
            "mean_within_task_r": float(np.mean(within_task_rs)) if within_task_rs else None,
            "within_task_r_range": [float(min(within_task_rs)), float(max(within_task_rs))] if within_task_rs else None,
            "n_tasks_significant_negative": n_negative_within,
            "n_tasks_significant_positive": n_positive_within,
        }

        return result

    def detect_dissociation(
        self,
        alignments: List[AlignmentResult],
        steering_results: Dict,
        iid_summaries: Optional[List[IIDSummary]] = None,
    ) -> Tuple[List[DissociationCase], Dict[str, Any]]:
        """
        Detect cases where high alignment does NOT predict good transfer.

        Per spec §5.3: uses data-derived percentile thresholds (default),
        not absolute cutoffs.
        """
        sci = self.config.science
        cos_threshold = sci.high_alignment_cosine
        acc_threshold = sci.low_transfer_accuracy

        if sci.use_data_derived_thresholds:
            all_cosines = [ar.cosine for ar in alignments]
            if all_cosines:
                cos_threshold = float(np.percentile(all_cosines, sci.alignment_percentile))
                logger.info(
                    "Data-derived cosine threshold: %.3f (%.0f-th percentile)",
                    cos_threshold, sci.alignment_percentile,
                )

        # Build IID lookup
        iid_lookup: Dict[Tuple[str, str], float] = {}
        if iid_summaries:
            for s in iid_summaries:
                iid_lookup[(s.task, s.template_id)] = s.best_accuracy

        cases: List[DissociationCase] = []
        n_high_align = 0

        for ar in alignments:
            if ar.cosine <= cos_threshold:
                continue
            n_high_align += 1

            try:
                acc_ab = steering_results[ar.task][ar.template_a][ar.template_b][ar.layer]
                acc_ba = steering_results[ar.task][ar.template_b][ar.template_a][ar.layer]
            except (KeyError, TypeError):
                continue

            iid_a = iid_lookup.get((ar.task, ar.template_a), -1)
            iid_b = iid_lookup.get((ar.task, ar.template_b), -1)

            if acc_ab < acc_threshold:
                cases.append(DissociationCase(
                    task=ar.task, source=ar.template_a, target=ar.template_b,
                    layer=ar.layer, cosine=ar.cosine,
                    transfer_accuracy=acc_ab, reverse_accuracy=acc_ba,
                    iid_source_accuracy=iid_a,
                ))
            if acc_ba < acc_threshold:
                cases.append(DissociationCase(
                    task=ar.task, source=ar.template_b, target=ar.template_a,
                    layer=ar.layer, cosine=ar.cosine,
                    transfer_accuracy=acc_ba, reverse_accuracy=acc_ab,
                    iid_source_accuracy=iid_b,
                ))

        # Separate cases by IID viability
        iid_threshold = self.config.science.iid_accuracy_threshold
        viable = [c for c in cases if c.iid_source_accuracy >= iid_threshold]
        non_viable = [c for c in cases if c.iid_source_accuracy < iid_threshold]

        summary = {
            "cos_threshold": cos_threshold,
            "acc_threshold": acc_threshold,
            "threshold_type": "data_derived" if sci.use_data_derived_thresholds else "fixed",
            "n_high_alignment_pairs": n_high_align,
            "n_dissociation_total": len(cases),
            "n_dissociation_iid_viable": len(viable),
            "n_dissociation_iid_non_viable": len(non_viable),
            "dissociation_rate": len(cases) / (2 * n_high_align) if n_high_align > 0 else 0,
            "note": (
                f"{len(non_viable)} dissociation cases have IID accuracy below "
                f"{iid_threshold:.2f} -- these reflect extraction method failure, "
                f"not geometry-behavior dissociation"
            ) if non_viable else "All dissociation cases have viable IID accuracy",
        }

        return cases, summary

    def compute_transfer_matrix(
        self, steering_results: Dict, task: str,
    ) -> Dict[str, Any]:
        """
        Construct 8x8 directed transfer matrix per task (spec §5.1).

        M[i,j] = best_accuracy(FV from template i, tested on template j).
        """
        if task not in steering_results:
            return {"error": f"No steering results for task '{task}'"}

        templates = sorted(steering_results[task].keys())
        n = len(templates)
        matrix = np.zeros((n, n))

        for i, src in enumerate(templates):
            for j, tgt in enumerate(templates):
                if src not in steering_results[task]:
                    continue
                if tgt not in steering_results[task][src]:
                    continue
                layer_accs = steering_results[task][src][tgt]
                if isinstance(layer_accs, dict):
                    matrix[i, j] = max(layer_accs.values()) if layer_accs else 0
                else:
                    matrix[i, j] = layer_accs

        # Compute statistics per spec §5.1
        diag = np.diag(matrix)
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag = matrix[off_diag_mask]

        row_means = []
        col_means = []
        for i in range(n):
            row_off = [matrix[i, j] for j in range(n) if j != i]
            col_off = [matrix[j, i] for j in range(n) if j != i]
            row_means.append(float(np.mean(row_off)) if row_off else 0)
            col_means.append(float(np.mean(col_off)) if col_off else 0)

        # Asymmetry: |M[i,j] - M[j,i]| for all i != j
        asymmetry = []
        for i in range(n):
            for j in range(i + 1, n):
                asymmetry.append(abs(matrix[i, j] - matrix[j, i]))

        return {
            "task": task,
            "templates": templates,
            "matrix": matrix.tolist(),
            "diagonal_mean": float(np.mean(diag)),
            "off_diagonal_mean": float(np.mean(off_diag)) if off_diag.size > 0 else 0,
            "iid_ood_gap": float(np.mean(diag) - np.mean(off_diag)) if off_diag.size > 0 else 0,
            "row_means": row_means,
            "col_means": col_means,
            "mean_asymmetry": float(np.mean(asymmetry)) if asymmetry else 0,
        }

    def within_vs_across_style_transfer(
        self,
        steering_results: Dict,
    ) -> Dict[str, Any]:
        """
        Compare within-style vs across-style transfer (spec §5.2 point 3).

        Uses TEMPLATE_STYLE_MAP to determine which template pairs share a style.
        """
        within_style_accs = []
        across_style_accs = []

        for task in steering_results:
            for src in steering_results[task]:
                for tgt in steering_results[task][src]:
                    if src == tgt:
                        continue
                    src_style = TEMPLATE_STYLE_MAP.get(src)
                    tgt_style = TEMPLATE_STYLE_MAP.get(tgt)
                    if src_style is None or tgt_style is None:
                        continue

                    layer_accs = steering_results[task][src][tgt]
                    if not isinstance(layer_accs, dict):
                        continue
                    best_acc = max(layer_accs.values()) if layer_accs else 0

                    if src_style == tgt_style:
                        within_style_accs.append(best_acc)
                    else:
                        across_style_accs.append(best_acc)

        result = {
            "within_style_mean": float(np.mean(within_style_accs)) if within_style_accs else None,
            "across_style_mean": float(np.mean(across_style_accs)) if across_style_accs else None,
            "within_style_n": len(within_style_accs),
            "across_style_n": len(across_style_accs),
        }

        if within_style_accs and across_style_accs:
            t_stat, p_val = stats.ttest_ind(within_style_accs, across_style_accs)
            result["t_statistic"] = float(t_stat)
            result["p_value"] = float(p_val)
            result["style_effect"] = (
                "Within-style transfer significantly higher"
                if p_val < 0.05 and t_stat > 0
                else "No significant style effect"
            )

        return result

    def permutation_test_dissociation(
        self,
        alignments: List[AlignmentResult],
        steering_results: Dict,
        n_permutations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Permutation test for dissociation significance (spec §7.2 Control D).

        Tests whether observed dissociation cases exceed chance.
        """
        sci = self.config.science

        # Collect paired (cosine, transfer) values per task
        per_task_pairs: Dict[str, List[Tuple[float, float]]] = {}

        for ar in alignments:
            try:
                acc_ab = steering_results[ar.task][ar.template_a][ar.template_b][ar.layer]
            except (KeyError, TypeError):
                continue
            per_task_pairs.setdefault(ar.task, []).append((ar.cosine, acc_ab))

        results = {}
        for task, pairs in per_task_pairs.items():
            if len(pairs) < 10:
                continue

            cosines = [c for c, _ in pairs]
            accs = [a for _, a in pairs]

            # Data-derived threshold for this task
            cos_threshold = float(np.percentile(cosines, sci.alignment_percentile))
            acc_threshold = sci.low_transfer_accuracy

            # Observed dissociation count
            observed = sum(
                1 for c, a in pairs if c > cos_threshold and a < acc_threshold
            )

            # Permutation
            rng = np.random.RandomState(42)
            perm_counts = []
            for _ in range(n_permutations):
                shuffled_accs = rng.permutation(accs)
                count = sum(
                    1 for c, a in zip(cosines, shuffled_accs)
                    if c > cos_threshold and a < acc_threshold
                )
                perm_counts.append(count)

            p_value = float(np.mean([c >= observed for c in perm_counts]))

            results[task] = {
                "observed_dissociations": observed,
                "mean_permuted": float(np.mean(perm_counts)),
                "p_value": p_value,
                "n_permutations": n_permutations,
                "significant": p_value < 0.05,
            }

        return results

    def compute_utv(
        self, fvs: FVCollection, task: str, layer: int,
    ) -> Dict[str, Any]:
        """Compute Universal Task Vector via PCA for one task/layer."""
        templates = list(fvs[task].keys())
        if len(templates) < 2:
            return {"error": "Need >= 2 templates for PCA"}

        matrix = torch.stack(
            [fvs[task][t][layer].vector for t in templates], dim=0
        ).float().numpy()

        pca = PCA(n_components=min(len(templates), matrix.shape[0]))
        pca.fit(matrix)

        projections = {}
        for t in templates:
            fv_np = fvs[task][t][layer].vector.float().numpy()
            proj = float(np.dot(fv_np, pca.components_[0]) / (np.linalg.norm(pca.components_[0]) + 1e-10))
            projections[t] = proj

        return {
            "task": task, "layer": layer,
            "pc1_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "top3_variance_ratio": [float(v) for v in pca.explained_variance_ratio_[:3]],
            "template_projections": projections,
        }

    def hierarchical_regression(
        self,
        alignments: List[AlignmentResult],
        steering_results: Dict,
        iid_summaries: Optional[List[IIDSummary]] = None,
    ) -> Dict[str, Any]:
        """
        Hierarchical regression: transfer ~ difficulty + cosine + difficulty*cosine
        (spec §4.3 point 6).

        Uses task-level mean IID accuracy as the difficulty proxy.
        Tests whether cosine predicts transfer after controlling for difficulty.
        """
        # Build difficulty lookup from IID summaries
        difficulty_by_task: Dict[str, float] = {}
        if iid_summaries:
            task_accs: Dict[str, List[float]] = {}
            for s in iid_summaries:
                task_accs.setdefault(s.task, []).append(s.best_accuracy)
            for task, accs in task_accs.items():
                difficulty_by_task[task] = 1.0 - float(np.mean(accs))

        cosines, accs_list, difficulties = [], [], []

        for ar in alignments:
            if ar.task not in difficulty_by_task:
                continue
            try:
                acc_ab = steering_results[ar.task][ar.template_a][ar.template_b][ar.layer]
                acc_ba = steering_results[ar.task][ar.template_b][ar.template_a][ar.layer]
            except (KeyError, TypeError):
                continue

            diff = difficulty_by_task[ar.task]
            for acc in [acc_ab, acc_ba]:
                cosines.append(ar.cosine)
                accs_list.append(acc)
                difficulties.append(diff)

        if len(cosines) < 10:
            return {"error": "insufficient data for hierarchical regression"}

        cosines_arr = np.array(cosines)
        accs_arr = np.array(accs_list)
        difficulties_arr = np.array(difficulties)
        interaction = cosines_arr * difficulties_arr

        # Model 1: transfer ~ difficulty only
        X_diff = difficulties_arr.reshape(-1, 1)
        reg_diff = LinearRegression().fit(X_diff, accs_arr)
        r2_diff = float(reg_diff.score(X_diff, accs_arr))

        # Model 2: transfer ~ difficulty + cosine
        X_diff_cos = np.column_stack([difficulties_arr, cosines_arr])
        reg_diff_cos = LinearRegression().fit(X_diff_cos, accs_arr)
        r2_diff_cos = float(reg_diff_cos.score(X_diff_cos, accs_arr))

        # Model 3: transfer ~ difficulty + cosine + difficulty*cosine
        X_full = np.column_stack([difficulties_arr, cosines_arr, interaction])
        reg_full = LinearRegression().fit(X_full, accs_arr)
        r2_full = float(reg_full.score(X_full, accs_arr))

        return {
            "n_samples": len(cosines),
            "model_1_difficulty_only": {
                "r_squared": r2_diff,
                "coef_difficulty": float(reg_diff.coef_[0]),
                "intercept": float(reg_diff.intercept_),
            },
            "model_2_difficulty_plus_cosine": {
                "r_squared": r2_diff_cos,
                "r_squared_delta": r2_diff_cos - r2_diff,
                "coef_difficulty": float(reg_diff_cos.coef_[0]),
                "coef_cosine": float(reg_diff_cos.coef_[1]),
                "intercept": float(reg_diff_cos.intercept_),
            },
            "model_3_full_with_interaction": {
                "r_squared": r2_full,
                "r_squared_delta": r2_full - r2_diff_cos,
                "coef_difficulty": float(reg_full.coef_[0]),
                "coef_cosine": float(reg_full.coef_[1]),
                "coef_interaction": float(reg_full.coef_[2]),
                "intercept": float(reg_full.intercept_),
            },
            "interpretation": (
                "R² delta from model 1 to 2 shows incremental predictive value of "
                "cosine after controlling for difficulty. If delta ≈ 0, cosine adds "
                "nothing beyond difficulty."
            ),
        }

    def fv_norm_analysis(
        self, fvs: FVCollection, steering_results: Dict,
    ) -> Dict[str, Any]:
        """FV norm analysis (spec §5.2 point 4)."""
        norms = []
        accs = []

        for task in fvs:
            for src in fvs[task]:
                for layer, fv in fvs[task][src].items():
                    for tgt in steering_results.get(task, {}).get(src, {}):
                        if tgt == src:
                            continue
                        try:
                            acc = steering_results[task][src][tgt][layer]
                        except (KeyError, TypeError):
                            continue
                        norms.append(fv.norm)
                        accs.append(acc)

        if len(norms) > 2:
            r, p = stats.pearsonr(norms, accs)
            return {
                "norm_transfer_pearson_r": float(r),
                "norm_transfer_p_value": float(p),
                "n_samples": len(norms),
                "mean_norm": float(np.mean(norms)),
            }
        return {"error": "insufficient data"}

    def run_all(
        self,
        fvs: FVCollection,
        steering_results: Dict,
        iid_summaries: Optional[List[IIDSummary]] = None,
    ) -> Dict[str, Any]:
        """Run all geometric analyses."""
        logger.info("Computing pairwise alignments...")
        alignments = self.compute_pairwise_alignments(fvs)

        logger.info("Computing alignment-transfer correlation (pooled + per-task + per-layer)...")
        correlation = self.alignment_transfer_correlation(alignments, steering_results)

        logger.info("Detecting dissociation cases (data-derived thresholds)...")
        dissociation_cases, dissociation_summary = self.detect_dissociation(
            alignments, steering_results, iid_summaries,
        )

        logger.info("Computing transfer matrices (8x8 per task)...")
        transfer_matrices = {}
        for task in steering_results:
            transfer_matrices[task] = self.compute_transfer_matrix(steering_results, task)

        logger.info("Computing within-style vs across-style transfer...")
        style_analysis = self.within_vs_across_style_transfer(steering_results)

        logger.info("Running permutation test for dissociation significance...")
        permutation_results = self.permutation_test_dissociation(alignments, steering_results)

        logger.info("Computing hierarchical regression (spec §4.3)...")
        hierarchical = self.hierarchical_regression(alignments, steering_results, iid_summaries)

        logger.info("Computing FV norm analysis...")
        norm_analysis = self.fv_norm_analysis(fvs, steering_results)

        logger.info("Computing UTVs...")
        utv_results = {}
        for task in fvs:
            utv_results[task] = {}
            layers = sorted(next(iter(fvs[task].values())).keys())
            for layer in layers:
                utv_results[task][layer] = self.compute_utv(fvs, task, layer)

        return {
            "alignments": [asdict(a) for a in alignments],
            "correlation": correlation,
            "dissociation_cases": [asdict(c) for c in dissociation_cases],
            "dissociation_summary": dissociation_summary,
            "transfer_matrices": transfer_matrices,
            "style_analysis": style_analysis,
            "permutation_test": permutation_results,
            "hierarchical_regression": hierarchical,
            "norm_analysis": norm_analysis,
            "utv": utv_results,
        }


def save_analysis_results(results: Dict, output_dir: Path):
    """Save analysis results to disk."""
    path = output_dir / "geometric_analysis.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Geometric analysis saved to %s", path)
