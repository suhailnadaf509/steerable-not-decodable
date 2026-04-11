"""
Dataset Construction
====================
Generates ICL prompts and test sets from task specifications.
Handles train/test splitting, prompt formatting, and serialization.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import ExperimentConfig, SEED
from .tasks import TaskSpec, get_tasks

logger = logging.getLogger(__name__)


@dataclass
class PromptSet:
    """All prompts for a single (task, template) combination."""
    task: str
    template_id: str
    template_str: str

    # ICL prompts for FV extraction
    icl_positive: List[Dict[str, str]]   # [{"prompt": ..., "expected": ..., "input": ...}]
    icl_negative: List[Dict[str, str]]

    # IID test prompts (same template as extraction)
    iid_test: List[Dict[str, str]]

    # OOD test prompts, keyed by target template
    ood_test: Dict[str, List[Dict[str, str]]]


def _format_icl_prompt(
    template_str: str,
    test_input: str,
    demos: List[Tuple[str, str]],
    n_demos: int = 5,
) -> str:
    """
    Format a positive ICL prompt with demonstrations.

    Format:
        template(demo1_in) demo1_out
        template(demo2_in) demo2_out
        ...
        template(test_input)
    """
    # Exclude test_input from demos
    available = [(inp, out) for inp, out in demos if inp != test_input]
    selected = available[:n_demos]

    lines = []
    for inp, out in selected:
        formatted_input = template_str.replace("{X}", inp)
        lines.append(f"{formatted_input} {out}")

    test_formatted = template_str.replace("{X}", test_input)
    lines.append(test_formatted)

    return "\n".join(lines)


def generate_prompts_for_task(
    task: TaskSpec,
    config: ExperimentConfig,
) -> Dict[str, PromptSet]:
    """
    Generate all prompt sets for one task.

    Returns:
        Dict[template_id] -> PromptSet
    """
    rng = random.Random(config.seed)
    np_rng = np.random.RandomState(config.seed)

    # Shuffle pairs
    pairs = list(task.pairs)
    rng.shuffle(pairs)

    n_icl = config.ops.n_icl_positive
    n_iid = config.ops.n_iid_test
    n_ood = config.ops.n_ood_test
    n_demos = config.ops.n_icl_demos

    # Need at least n_icl + n_iid unique inputs.
    # If not enough pairs, cycle (with warning).
    needed = n_icl + n_iid
    if len(pairs) < needed:
        logger.warning(
            "Task '%s' has %d pairs but needs %d -- cycling data",
            task.name, len(pairs), needed,
        )
        while len(pairs) < needed:
            pairs = pairs + list(task.pairs)

    icl_pairs = pairs[:n_icl]
    test_pairs = pairs[n_icl:n_icl + n_iid]
    ood_pairs = pairs[n_icl:n_icl + n_ood]  # overlap with test_pairs is fine

    # Demo pool: all pairs except those used for test
    demo_pool = [(inp, out) for inp, out in task.pairs]
    rng.shuffle(demo_pool)

    result: Dict[str, PromptSet] = {}

    for tid, template_str in task.templates.items():
        # --- ICL positive (with demonstrations) ---
        icl_pos = []
        for inp, out in icl_pairs:
            prompt = _format_icl_prompt(
                template_str, inp, demo_pool, n_demos=n_demos,
            )
            icl_pos.append({"prompt": prompt, "expected": out, "input": inp})

        # --- ICL negative (no demonstrations) ---
        icl_neg = []
        for inp, out in icl_pairs:
            bare_prompt = template_str.replace("{X}", inp)
            icl_neg.append({"prompt": bare_prompt, "expected": out, "input": inp})

        # --- IID test (bare prompts, same template) ---
        iid_test = []
        for inp, out in test_pairs:
            bare_prompt = template_str.replace("{X}", inp)
            iid_test.append({"prompt": bare_prompt, "expected": out, "input": inp})

        # --- OOD test (bare prompts, OTHER templates) ---
        ood_test: Dict[str, List[Dict[str, str]]] = {}
        for other_tid, other_template_str in task.templates.items():
            if other_tid == tid:
                continue
            ood_prompts = []
            for inp, out in ood_pairs:
                bare_prompt = other_template_str.replace("{X}", inp)
                ood_prompts.append({"prompt": bare_prompt, "expected": out, "input": inp})
            ood_test[other_tid] = ood_prompts

        result[tid] = PromptSet(
            task=task.name,
            template_id=tid,
            template_str=template_str,
            icl_positive=icl_pos,
            icl_negative=icl_neg,
            iid_test=iid_test,
            ood_test=ood_test,
        )

    return result


def generate_all_prompts(
    config: ExperimentConfig,
    task_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, PromptSet]]:
    """
    Generate complete prompt dataset for all requested tasks.

    Returns:
        prompts[task_name][template_id] = PromptSet
    """
    tasks = get_tasks(task_names or config.task_names)

    all_prompts: Dict[str, Dict[str, PromptSet]] = {}
    total_count = 0

    for task_name, task_spec in tasks.items():
        prompt_sets = generate_prompts_for_task(task_spec, config)
        all_prompts[task_name] = prompt_sets

        for tid, pset in prompt_sets.items():
            n = (
                len(pset.icl_positive)
                + len(pset.icl_negative)
                + len(pset.iid_test)
                + sum(len(v) for v in pset.ood_test.values())
            )
            total_count += n

    logger.info(
        "Generated %d total prompts across %d tasks x %d templates",
        total_count,
        len(all_prompts),
        len(next(iter(all_prompts.values()))) if all_prompts else 0,
    )
    return all_prompts


def validate_prompts(prompts: Dict[str, Dict[str, PromptSet]]) -> Dict[str, int]:
    """Validate and count all generated prompts."""
    counts = {"icl_pos": 0, "icl_neg": 0, "iid": 0, "ood": 0, "total": 0}

    for task_name, templates in prompts.items():
        for tid, pset in templates.items():
            counts["icl_pos"] += len(pset.icl_positive)
            counts["icl_neg"] += len(pset.icl_negative)
            counts["iid"] += len(pset.iid_test)
            for ood_list in pset.ood_test.values():
                counts["ood"] += len(ood_list)

    counts["total"] = sum(counts.values())
    return counts


def save_prompts(prompts: Dict[str, Dict[str, PromptSet]], path: Path):
    """Save prompts to JSON (for reproducibility logging)."""
    serializable = {}
    for task_name, templates in prompts.items():
        serializable[task_name] = {}
        for tid, pset in templates.items():
            serializable[task_name][tid] = {
                "task": pset.task,
                "template_id": pset.template_id,
                "template_str": pset.template_str,
                "icl_positive_count": len(pset.icl_positive),
                "icl_negative_count": len(pset.icl_negative),
                "iid_test_count": len(pset.iid_test),
                "ood_test_counts": {
                    k: len(v) for k, v in pset.ood_test.items()
                },
            }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Prompt summary saved to %s", path)
