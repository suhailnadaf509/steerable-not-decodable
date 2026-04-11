"""
Additive Steering Evaluation
=============================
Evaluates function vectors by adding them to the model's residual stream
and measuring task accuracy on IID and OOD test sets.

Key features:
  - Adaptive steering strength sweep (broad then refine)
  - Per-example prediction logging
  - IID accuracy summary as FIRST output
  - Task-specific accuracy evaluation
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
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from .config import ExperimentConfig
from .data import PromptSet
from .extraction import FunctionVector, FVCollection
from .models import ModelWrapper
from .tasks import TaskSpec, get_tasks

logger = logging.getLogger(__name__)


@dataclass
class SteeringResult:
    """Result of one steering evaluation."""
    task: str
    source_template: str
    target_template: str
    layer: int
    strength: float
    accuracy: float
    n_correct: int
    n_total: int
    is_iid: bool
    predictions: Optional[List[Dict[str, str]]] = None  # per-example


@dataclass
class IIDSummary:
    """IID accuracy summary for one task x template, best across layers/strengths."""
    task: str
    template_id: str
    best_accuracy: float
    best_layer: int
    best_strength: float
    status: str  # "PASS" or "FAIL"


class SteeringEvaluator:
    """
    Evaluates function vectors via additive steering.
    """

    def __init__(self, model: ModelWrapper, config: ExperimentConfig):
        self.model = model
        self.config = config
        self._fv_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}

    def cache_fv_tensors(self, fvs: FVCollection):
        """Move FV tensors to device for fast access."""
        count = 0
        for task in fvs:
            for tid in fvs[task]:
                for layer, fv in fvs[task][tid].items():
                    key = (task, tid, layer)
                    self._fv_cache[key] = fv.vector.to(self.model.device)
                    count += 1
        logger.info("Cached %d FV tensors on %s", count, self.model.device)

    @torch.no_grad()
    def steer_and_evaluate(
        self,
        fv_tensor: torch.Tensor,
        layer_0idx: int,
        test_prompts: List[Dict[str, str]],
        task_spec: TaskSpec,
        strength: float,
        save_predictions: bool = False,
    ) -> SteeringResult:
        """
        Apply additive steering and evaluate accuracy.

        Args:
            fv_tensor: function vector (on device)
            layer_0idx: 0-indexed layer for hook
            test_prompts: list of {"prompt", "expected", "input"} dicts
            task_spec: for accuracy checking
            strength: steering multiplier
            save_predictions: whether to store per-example results
        """
        hook_name = self.model.resid_post_hook(layer_0idx)
        batch_size = self.config.ops.steering_batch_size
        max_tokens = task_spec.max_new_tokens

        all_outputs: List[str] = []

        for i in range(0, len(test_prompts), batch_size):
            batch = test_prompts[i:i + batch_size]
            batch_texts = [p["prompt"] for p in batch]
            tokens = self.model.model.to_tokens(batch_texts, prepend_bos=True)

            def steering_hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:
                acts[:, -1, :] += strength * fv_tensor
                return acts

            outputs = self.model.generate_with_hook(
                tokens, hook_name, steering_hook, max_new_tokens=max_tokens,
            )
            all_outputs.extend(outputs)

        # Evaluate accuracy
        n_correct = 0
        predictions = [] if save_predictions else None

        for prompt_dict, output in zip(test_prompts, all_outputs):
            correct = task_spec.check_accuracy(prompt_dict["input"], output)
            if correct:
                n_correct += 1
            if save_predictions:
                predictions.append({
                    "input": prompt_dict["input"],
                    "expected": prompt_dict["expected"],
                    "output": output,
                    "correct": correct,
                })

        n_total = len(test_prompts)
        accuracy = n_correct / n_total if n_total > 0 else 0.0

        # Placeholder fields -- caller fills in task/template/layer/is_iid
        return SteeringResult(
            task="", source_template="", target_template="",
            layer=0, strength=strength,
            accuracy=accuracy, n_correct=n_correct, n_total=n_total,
            is_iid=False, predictions=predictions,
        )

    @torch.no_grad()
    def steer_and_evaluate_multi_strength(
        self,
        fv_tensor: torch.Tensor,
        layer_0idx: int,
        test_prompts: List[Dict[str, str]],
        task_spec: TaskSpec,
        strengths: List[float],
        save_predictions: bool = False,
    ) -> Dict[float, SteeringResult]:
        """
        Evaluate multiple steering strengths in batched generation calls.

        Tiles each sub-batch across strengths so that one model.generate() call
        produces results for ALL strengths simultaneously.  This gives up to
        len(strengths)x speedup on the steering evaluation bottleneck while
        producing IDENTICAL results to calling steer_and_evaluate per strength.

        The number of strengths processed per call is controlled by
        config.ops.strength_tile_factor.
        """
        if not strengths:
            return {}

        hook_name = self.model.resid_post_hook(layer_0idx)
        max_tokens = task_spec.max_new_tokens
        tile_factor = self.config.ops.strength_tile_factor

        # Fall back to sequential if tile_factor == 1
        if tile_factor <= 1:
            results = {}
            for s in strengths:
                results[s] = self.steer_and_evaluate(
                    fv_tensor, layer_0idx, test_prompts,
                    task_spec, s, save_predictions=save_predictions,
                )
            return results

        # Effective per-strength batch size (total batch = tile * effective)
        effective_batch = max(1, self.config.ops.steering_batch_size // tile_factor)

        all_outputs: Dict[float, List[str]] = {s: [] for s in strengths}

        # Process strengths in tiles of tile_factor
        for s_start in range(0, len(strengths), tile_factor):
            batch_strengths = strengths[s_start:s_start + tile_factor]

            for i in range(0, len(test_prompts), effective_batch):
                batch = test_prompts[i:i + effective_batch]
                batch_texts = [p["prompt"] for p in batch]
                tokens = self.model.model.to_tokens(batch_texts, prepend_bos=True)

                outputs_by_strength = self.model.generate_multi_strength(
                    tokens, hook_name, fv_tensor,
                    batch_strengths, max_tokens,
                )
                for s in batch_strengths:
                    all_outputs[s].extend(outputs_by_strength[s])

        # Evaluate accuracy per strength
        results: Dict[float, SteeringResult] = {}
        for s in strengths:
            n_correct = 0
            predictions = [] if save_predictions else None

            for prompt_dict, output in zip(test_prompts, all_outputs[s]):
                correct = task_spec.check_accuracy(prompt_dict["input"], output)
                if correct:
                    n_correct += 1
                if save_predictions:
                    predictions.append({
                        "input": prompt_dict["input"],
                        "expected": prompt_dict["expected"],
                        "output": output,
                        "correct": correct,
                    })

            n_total = len(test_prompts)
            results[s] = SteeringResult(
                task="", source_template="", target_template="",
                layer=0, strength=s,
                accuracy=n_correct / n_total if n_total > 0 else 0.0,
                n_correct=n_correct, n_total=n_total,
                is_iid=False, predictions=predictions,
            )

        return results

    def _generate_steered_raw(
        self,
        fv_tensor: torch.Tensor,
        layer_0idx: int,
        test_prompts: List[Dict[str, str]],
        strengths: List[float],
        max_new_tokens: int,
    ) -> Dict[float, List[str]]:
        """
        Generate steered outputs for all strengths.

        Returns raw output text per strength without computing accuracy.
        Uses the same batching/tiling as steer_and_evaluate_multi_strength.
        """
        hook_name = self.model.resid_post_hook(layer_0idx)
        tile_factor = self.config.ops.strength_tile_factor

        if tile_factor <= 1:
            # Sequential fallback
            all_outputs: Dict[float, List[str]] = {}
            batch_size = self.config.ops.steering_batch_size
            for s in strengths:
                outputs: List[str] = []
                for i in range(0, len(test_prompts), batch_size):
                    batch = test_prompts[i:i + batch_size]
                    batch_texts = [p["prompt"] for p in batch]
                    tokens = self.model.model.to_tokens(batch_texts, prepend_bos=True)

                    def steering_hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:
                        acts[:, -1, :] += s * fv_tensor
                        return acts

                    outs = self.model.generate_with_hook(
                        tokens, hook_name, steering_hook, max_new_tokens=max_new_tokens,
                    )
                    outputs.extend(outs)
                all_outputs[s] = outputs
            return all_outputs

        effective_batch = max(1, self.config.ops.steering_batch_size // tile_factor)
        all_outputs = {s: [] for s in strengths}

        for s_start in range(0, len(strengths), tile_factor):
            batch_strengths = strengths[s_start:s_start + tile_factor]
            for i in range(0, len(test_prompts), effective_batch):
                batch = test_prompts[i:i + effective_batch]
                batch_texts = [p["prompt"] for p in batch]
                tokens = self.model.model.to_tokens(batch_texts, prepend_bos=True)
                outputs_by_strength = self.model.generate_multi_strength(
                    tokens, hook_name, fv_tensor,
                    batch_strengths, max_new_tokens,
                )
                for s in batch_strengths:
                    all_outputs[s].extend(outputs_by_strength[s])

        return all_outputs

    def evaluate_all(
        self,
        fvs: FVCollection,
        prompts: Dict[str, Dict[str, PromptSet]],
        save_predictions: bool = False,
    ) -> Tuple[Dict[str, Dict], List[IIDSummary], List[SteeringResult]]:
        """
        Full steering evaluation across all tasks, templates, layers, strengths.

        Returns:
            results[task][src_template][tgt_template][layer] = best_accuracy
            iid_summary: list of IIDSummary (one per task x template)
            all_results: flat list of all SteeringResult objects
        """
        tasks = get_tasks(self.config.task_names)
        strengths = list(self.config.science.steering_strengths)

        self.cache_fv_tensors(fvs)

        results: Dict[str, Dict] = {}
        iid_summaries: List[IIDSummary] = []
        all_results: List[SteeringResult] = []

        # Count total evaluations for progress bar
        total = 0
        for task_name in fvs:
            if task_name not in prompts:
                continue
            n_templates = len(fvs[task_name])
            n_layers = len(next(iter(fvs[task_name].values())))
            # IID + OOD targets
            n_targets = n_templates  # IID + (n_templates - 1) OOD
            total += n_templates * n_layers * n_targets

        pbar = tqdm(total=total, desc="Steering evaluation")

        for task_name in fvs:
            if task_name not in prompts or task_name not in tasks:
                continue

            task_spec = tasks[task_name]
            results[task_name] = {}

            for src_tid in fvs[task_name]:
                results[task_name][src_tid] = {}
                best_iid_acc = 0.0
                best_iid_layer = 0
                best_iid_strength = 0.0

                for tgt_tid in prompts[task_name]:
                    results[task_name][src_tid][tgt_tid] = {}

                for layer_1idx, fv_obj in fvs[task_name][src_tid].items():
                    fv_tensor = self._fv_cache[(task_name, src_tid, layer_1idx)]
                    layer_0idx = fv_obj.layer_0idx

                    # --- IID: all strengths batched ---
                    iid_prompts = prompts[task_name][src_tid].iid_test
                    iid_by_strength = self.steer_and_evaluate_multi_strength(
                        fv_tensor, layer_0idx, iid_prompts,
                        task_spec, strengths,
                        save_predictions=save_predictions,
                    )

                    best_iid_at_layer = 0.0
                    best_iid_s = strengths[0]

                    for s, sr in iid_by_strength.items():
                        sr.task = task_name
                        sr.source_template = src_tid
                        sr.target_template = src_tid
                        sr.layer = layer_1idx
                        sr.is_iid = True
                        all_results.append(sr)

                        if sr.accuracy > best_iid_at_layer:
                            best_iid_at_layer = sr.accuracy
                            best_iid_s = s

                    # Adaptive refinement (batched; skip if already at 100%)
                    if self.config.science.adaptive_refine and 0 < best_iid_at_layer < 1.0:
                        delta = self.config.science.adaptive_refine_delta
                        n_pts = self.config.science.adaptive_refine_points
                        refine_strengths = [
                            best_iid_s + delta * (i - n_pts // 2)
                            for i in range(n_pts)
                        ]
                        refine_strengths = [
                            s for s in refine_strengths
                            if s > 0 and s not in strengths
                        ]
                        if refine_strengths:
                            refine_by_strength = self.steer_and_evaluate_multi_strength(
                                fv_tensor, layer_0idx, iid_prompts,
                                task_spec, refine_strengths,
                                save_predictions=False,
                            )
                            for s, sr in refine_by_strength.items():
                                sr.task = task_name
                                sr.source_template = src_tid
                                sr.target_template = src_tid
                                sr.layer = layer_1idx
                                sr.is_iid = True
                                all_results.append(sr)
                                if sr.accuracy > best_iid_at_layer:
                                    best_iid_at_layer = sr.accuracy

                    results[task_name][src_tid][src_tid][layer_1idx] = best_iid_at_layer
                    pbar.update(1)

                    if best_iid_at_layer > best_iid_acc:
                        best_iid_acc = best_iid_at_layer
                        best_iid_layer = layer_1idx
                        best_iid_strength = best_iid_s

                    # --- OOD: batch ALL target templates together ---
                    # Instead of 7 separate generate calls, concatenate all
                    # OOD prompts and run one batched call, then split results
                    # back per target.  Same FV, same layer — safe to merge.
                    all_ood_prompts: List[Dict[str, str]] = []
                    ood_segments: List[Tuple[str, int, int]] = []  # (tgt_tid, start, end)

                    for tgt_tid in prompts[task_name]:
                        if tgt_tid == src_tid:
                            continue
                        ood = prompts[task_name][src_tid].ood_test.get(tgt_tid, [])
                        if not ood:
                            results[task_name][src_tid][tgt_tid][layer_1idx] = 0.0
                            pbar.update(1)
                            continue
                        seg_start = len(all_ood_prompts)
                        all_ood_prompts.extend(ood)
                        ood_segments.append((tgt_tid, seg_start, len(all_ood_prompts)))

                    if all_ood_prompts:
                        # Single batched generation for ALL OOD targets
                        ood_outputs_by_strength = self._generate_steered_raw(
                            fv_tensor, layer_0idx, all_ood_prompts,
                            strengths, task_spec.max_new_tokens,
                        )

                        for tgt_tid, seg_start, seg_end in ood_segments:
                            seg_prompts = all_ood_prompts[seg_start:seg_end]
                            best_ood = 0.0

                            for s in strengths:
                                seg_outputs = ood_outputs_by_strength[s][seg_start:seg_end]
                                n_correct = 0
                                predictions = [] if save_predictions else None

                                for p, o in zip(seg_prompts, seg_outputs):
                                    correct = task_spec.check_accuracy(p["input"], o)
                                    if correct:
                                        n_correct += 1
                                    if save_predictions:
                                        predictions.append({
                                            "input": p["input"],
                                            "expected": p["expected"],
                                            "output": o,
                                            "correct": correct,
                                        })

                                n_total = len(seg_prompts)
                                accuracy = n_correct / n_total if n_total > 0 else 0.0
                                sr = SteeringResult(
                                    task=task_name,
                                    source_template=src_tid,
                                    target_template=tgt_tid,
                                    layer=layer_1idx,
                                    strength=s,
                                    accuracy=accuracy,
                                    n_correct=n_correct,
                                    n_total=n_total,
                                    is_iid=False,
                                    predictions=predictions,
                                )
                                all_results.append(sr)
                                best_ood = max(best_ood, accuracy)

                            results[task_name][src_tid][tgt_tid][layer_1idx] = best_ood
                            pbar.update(1)

                    # Cleanup once per source template per layer (not per target)
                    self.model.cleanup()

                # IID summary for this task x template
                threshold = self.config.science.iid_accuracy_threshold
                iid_summaries.append(IIDSummary(
                    task=task_name,
                    template_id=src_tid,
                    best_accuracy=best_iid_acc,
                    best_layer=best_iid_layer,
                    best_strength=best_iid_strength,
                    status="PASS" if best_iid_acc >= threshold else "FAIL",
                ))

        pbar.close()
        return results, iid_summaries, all_results


@dataclass
class BaselineResult:
    """Result of zero-shot or few-shot baseline evaluation (spec §7.2 Control C)."""
    task: str
    template_id: str
    mode: str  # "zero_shot" or "few_shot"
    accuracy: float
    n_correct: int
    n_total: int


class BaselineEvaluator:
    """
    Evaluate zero-shot and few-shot baselines (spec §4.2 + §7.2 Control C).

    Zero-shot: template only, no demos, no steering.
    Few-shot: 5 ICL demonstrations, no steering.
    """

    def __init__(self, model: ModelWrapper, config: ExperimentConfig):
        self.model = model
        self.config = config

    @torch.no_grad()
    def evaluate_zero_shot(
        self,
        task_spec: TaskSpec,
        template_id: str,
        test_prompts: List[Dict[str, str]],
    ) -> BaselineResult:
        """Run zero-shot evaluation (template only, no demos, no steering)."""
        batch_size = self.config.ops.steering_batch_size
        max_tokens = task_spec.max_new_tokens
        n_correct = 0

        for i in range(0, len(test_prompts), batch_size):
            batch = test_prompts[i:i + batch_size]
            batch_texts = [p["prompt"] for p in batch]
            tokens = self.model.model.to_tokens(batch_texts, prepend_bos=True)
            outputs = self.model.model.generate(
                tokens.to(self.model.device),
                max_new_tokens=max_tokens,
                do_sample=False,
            )
            for j, p in enumerate(batch):
                gen_text = self.model.tokenizer.decode(
                    outputs[j, tokens.shape[1]:], skip_special_tokens=True,
                )
                if task_spec.check_accuracy(p["input"], gen_text):
                    n_correct += 1

        return BaselineResult(
            task=task_spec.name, template_id=template_id,
            mode="zero_shot", accuracy=n_correct / len(test_prompts) if test_prompts else 0,
            n_correct=n_correct, n_total=len(test_prompts),
        )

    @torch.no_grad()
    def evaluate_few_shot(
        self,
        task_spec: TaskSpec,
        template_id: str,
        test_prompts_with_demos: List[Dict[str, str]],
    ) -> BaselineResult:
        """Run few-shot evaluation (5 ICL demos, no steering)."""
        batch_size = self.config.ops.steering_batch_size
        max_tokens = task_spec.max_new_tokens
        n_correct = 0

        for i in range(0, len(test_prompts_with_demos), batch_size):
            batch = test_prompts_with_demos[i:i + batch_size]
            batch_texts = [p["prompt"] for p in batch]
            tokens = self.model.model.to_tokens(batch_texts, prepend_bos=True)
            outputs = self.model.model.generate(
                tokens.to(self.model.device),
                max_new_tokens=max_tokens,
                do_sample=False,
            )
            for j, p in enumerate(batch):
                gen_text = self.model.tokenizer.decode(
                    outputs[j, tokens.shape[1]:], skip_special_tokens=True,
                )
                if task_spec.check_accuracy(p["input"], gen_text):
                    n_correct += 1

        return BaselineResult(
            task=task_spec.name, template_id=template_id,
            mode="few_shot",
            accuracy=n_correct / len(test_prompts_with_demos) if test_prompts_with_demos else 0,
            n_correct=n_correct, n_total=len(test_prompts_with_demos),
        )

    def evaluate_all_baselines(
        self,
        prompts: Dict[str, Dict[str, 'PromptSet']],
    ) -> Tuple[List[BaselineResult], Dict[str, Any]]:
        """Run zero-shot + few-shot baselines for all tasks x templates."""
        tasks = get_tasks(self.config.task_names)
        results: List[BaselineResult] = []

        for task_name, task_spec in tasks.items():
            if task_name not in prompts:
                continue
            for tid in prompts[task_name]:
                pset = prompts[task_name][tid]

                # Zero-shot
                zs = self.evaluate_zero_shot(task_spec, tid, pset.iid_test)
                results.append(zs)

                # Few-shot (reuse ICL positive prompts format for test inputs)
                fs = self.evaluate_few_shot(task_spec, tid, pset.icl_positive)
                results.append(fs)

        # Summary
        summary: Dict[str, Any] = {}
        for r in results:
            summary.setdefault(r.task, {})[f"{r.template_id}_{r.mode}"] = r.accuracy

        return results, summary


def save_baseline_results(results: List[BaselineResult], output_dir: Path):
    """Save baseline results."""
    path = output_dir / "baseline_results.json"
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("Baseline results saved to %s", path)


def print_iid_summary(summaries: List[IIDSummary], threshold: float):
    """Print the IID accuracy summary table -- the FIRST output."""
    print()
    print("=" * 80)
    print("IID STEERING ACCURACY SUMMARY")
    print(f"Threshold for causal analysis: {threshold:.2f}")
    print("=" * 80)
    print(f"{'Task':<20} {'Template':<10} {'Best IID Acc':<14} {'Layer':<8} {'Strength':<10} {'Status'}")
    print("-" * 80)

    # Group by task
    tasks_seen = {}
    for s in summaries:
        if s.task not in tasks_seen:
            tasks_seen[s.task] = []
        tasks_seen[s.task].append(s)

    for task_name, task_summaries in tasks_seen.items():
        for s in sorted(task_summaries, key=lambda x: x.template_id):
            status_str = "PASS -> run mech interp" if s.status == "PASS" else "FAIL -> skip patching, run probing"
            print(
                f"{s.task:<20} {s.template_id:<10} {s.best_accuracy:<14.3f} "
                f"L{s.best_layer:<7} {s.best_strength:<10.1f} {status_str}"
            )

    # Task-level summary
    print()
    print("-" * 80)
    print(f"{'Task':<20} {'Category':<22} {'Mean IID':<10} {'Max IID':<10} {'Status'}")
    print("-" * 80)
    task_tasks = get_tasks()
    for task_name, task_summaries in tasks_seen.items():
        accs = [s.best_accuracy for s in task_summaries]
        mean_acc = np.mean(accs)
        max_acc = max(accs)
        status = "PASS" if max_acc >= threshold else "FAIL"
        cat = task_tasks[task_name].category.value if task_name in task_tasks else "?"
        print(f"{task_name:<20} {cat:<22} {mean_acc:<10.3f} {max_acc:<10.3f} {status}")

    n_pass = sum(1 for t, ts in tasks_seen.items() if max(s.best_accuracy for s in ts) >= threshold)
    n_total = len(tasks_seen)
    print(f"\nTasks passing IID threshold: {n_pass}/{n_total}")
    print("=" * 80)
    print()


def save_steering_results(
    results: Dict,
    summaries: List[IIDSummary],
    output_dir: Path,
):
    """Save steering results, metrics, and IID summaries."""

    def _convert(d):
        if isinstance(d, dict):
            return {str(k): _convert(v) for k, v in d.items()}
        return d

    with open(output_dir / "steering_results.json", "w") as f:
        json.dump(_convert(results), f, indent=2)

    summary_dicts = [asdict(s) for s in summaries]
    with open(output_dir / "iid_summary.json", "w") as f:
        json.dump(summary_dicts, f, indent=2)

    logger.info("Steering results saved to %s", output_dir)


def load_steering_results(output_dir: Path) -> Tuple[Dict, List[IIDSummary]]:
    """Load steering results from disk."""
    with open(output_dir / "steering_results.json") as f:
        raw = json.load(f)

    # Convert layer keys back to int
    def _convert(d, depth=0):
        if isinstance(d, dict):
            new = {}
            for k, v in d.items():
                try:
                    k = int(k)
                except (ValueError, TypeError):
                    pass
                new[k] = _convert(v, depth + 1)
            return new
        return d

    results = _convert(raw)

    with open(output_dir / "iid_summary.json") as f:
        summary_dicts = json.load(f)

    summaries = [IIDSummary(**d) for d in summary_dicts]
    return results, summaries
