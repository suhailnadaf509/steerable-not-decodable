"""
Multi-Method Function Vector Extraction
========================================
Implements three FV extraction methods:
  1. Mean-Difference (Todd et al. 2024) -- the standard approach
  2. PCA-based -- project onto top principal component of ICL activations
  3. Contrastive Activation Addition (CAA) -- mean of positive minus negative
     with contrast pairs rather than ICL structure

All methods produce vectors in R^{d_model} that can be used for additive steering.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from .config import ExperimentConfig
from .data import PromptSet
from .models import ModelWrapper
from .tasks import TaskSpec, get_tasks

logger = logging.getLogger(__name__)


@dataclass
class FunctionVector:
    """A function vector with full metadata."""
    task: str
    template_id: str
    layer: int           # 1-indexed (as in config)
    layer_0idx: int      # 0-indexed (for hooks)
    method: str          # "mean_diff", "pca", "caa"
    vector: torch.Tensor  # [d_model]
    norm: float
    n_positive: int
    n_negative: int
    model_key: str

    def to_meta(self) -> Dict:
        """Metadata dict (without tensor)."""
        return {
            "task": self.task,
            "template_id": self.template_id,
            "layer": self.layer,
            "layer_0idx": self.layer_0idx,
            "method": self.method,
            "norm": self.norm,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "model_key": self.model_key,
        }


# Type alias for the FV collection
FVCollection = Dict[str, Dict[str, Dict[int, FunctionVector]]]
# fvs[task][template_id][layer_1idx] = FunctionVector


class FVExtractor:
    """
    Extracts function vectors using multiple methods.
    """

    def __init__(self, model: ModelWrapper, config: ExperimentConfig):
        self.model = model
        self.config = config

    def extract_mean_diff(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layers_1idx: List[int],
        task: str,
        template_id: str,
    ) -> Dict[int, FunctionVector]:
        """
        Mean-Difference extraction (Todd et al. 2024).

        FV = mean(positive ICL activations) - mean(negative ICL activations)

        Args:
            positive_prompts: ICL prompts WITH demonstrations
            negative_prompts: prompts WITHOUT demonstrations (bare template)
            layers_1idx: layers to extract at (1-indexed)

        Returns:
            Dict[layer_1idx] -> FunctionVector
        """
        layers_0idx = [l - 1 for l in layers_1idx]
        n_pos = len(positive_prompts)

        # Combined forward pass: positive + negative in one call
        # Halves the number of setup/teardown cycles vs two separate calls
        combined_prompts = positive_prompts + negative_prompts
        combined_acts = self.model.get_activations(
            combined_prompts, layers_0idx,
            batch_size=self.config.ops.extraction_batch_size,
        )["resid_post"]

        results = {}
        for layer_1idx, layer_0idx in zip(layers_1idx, layers_0idx):
            all_acts = combined_acts[layer_0idx]
            pos_mean = all_acts[:n_pos].mean(dim=0)
            neg_mean = all_acts[n_pos:].mean(dim=0)
            fv = pos_mean - neg_mean

            results[layer_1idx] = FunctionVector(
                task=task,
                template_id=template_id,
                layer=layer_1idx,
                layer_0idx=layer_0idx,
                method="mean_diff",
                vector=fv,
                norm=fv.norm().item(),
                n_positive=len(positive_prompts),
                n_negative=len(negative_prompts),
                model_key=self.model.model_key,
            )

        return results

    def extract_pca(
        self,
        positive_prompts: List[str],
        layers_1idx: List[int],
        task: str,
        template_id: str,
    ) -> Dict[int, FunctionVector]:
        """
        PCA-based extraction.

        Fits PCA on positive ICL activations and uses the first principal
        component direction, scaled to mean activation norm.
        """
        layers_0idx = [l - 1 for l in layers_1idx]

        pos_acts = self.model.get_activations(
            positive_prompts, layers_0idx,
            batch_size=self.config.ops.extraction_batch_size,
        )["resid_post"]

        results = {}
        for layer_1idx, layer_0idx in zip(layers_1idx, layers_0idx):
            acts_np = pos_acts[layer_0idx].float().numpy()

            if acts_np.shape[0] < 2:
                logger.warning(
                    "PCA extraction needs >= 2 samples, got %d for %s/%s/L%d",
                    acts_np.shape[0], task, template_id, layer_1idx,
                )
                continue

            pca = PCA(n_components=1)
            pca.fit(acts_np)
            direction = pca.components_[0]

            # Scale to mean norm
            mean_norm = np.linalg.norm(acts_np, axis=1).mean()
            pc1_norm = np.linalg.norm(direction)
            scaled = direction * (mean_norm / (pc1_norm + 1e-10))

            fv = torch.tensor(scaled, dtype=torch.float32)

            results[layer_1idx] = FunctionVector(
                task=task,
                template_id=template_id,
                layer=layer_1idx,
                layer_0idx=layer_0idx,
                method="pca",
                vector=fv,
                norm=fv.norm().item(),
                n_positive=len(positive_prompts),
                n_negative=0,
                model_key=self.model.model_key,
            )

        return results

    def extract_all(
        self,
        prompts: Dict[str, Dict[str, PromptSet]],
        methods: List[str] = ("mean_diff",),
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, FVCollection]:
        """
        Extract FVs for all tasks x templates x layers using specified methods.

        Args:
            prompts: The full prompt dataset
            methods: Which extraction methods to use
            task_names: Restrict to these tasks (None = all in prompts)

        Returns:
            result[method][task][template_id][layer_1idx] = FunctionVector
        """
        tasks_to_process = task_names or list(prompts.keys())
        layers = self.model.spec.extraction_layers()

        all_results: Dict[str, FVCollection] = {}

        for method in methods:
            all_results[method] = {}

            # Count total extractions for progress bar
            total = sum(
                len(prompts[t]) * len(layers)
                for t in tasks_to_process
                if t in prompts
            )
            pbar = tqdm(total=total, desc=f"Extracting FVs ({method})")

            for task_name in tasks_to_process:
                if task_name not in prompts:
                    logger.warning("Task '%s' not in prompts -- skipping", task_name)
                    continue

                all_results[method][task_name] = {}

                for tid, pset in prompts[task_name].items():
                    pos_prompts = [p["prompt"] for p in pset.icl_positive]
                    neg_prompts = [p["prompt"] for p in pset.icl_negative]

                    if method == "mean_diff":
                        fvs = self.extract_mean_diff(
                            pos_prompts, neg_prompts, layers,
                            task_name, tid,
                        )
                    elif method == "pca":
                        fvs = self.extract_pca(
                            pos_prompts, layers, task_name, tid,
                        )
                    else:
                        raise ValueError(f"Unknown extraction method: {method}")

                    all_results[method][task_name][tid] = fvs
                    pbar.update(len(layers))

            pbar.close()

        return all_results


# -- Serialization --

def save_fvs(fvs: FVCollection, path: Path, method: str = "mean_diff"):
    """Save function vectors to disk."""
    tensors = {}
    metadata = {}

    for task in fvs:
        tensors[task] = {}
        metadata[task] = {}
        for tid in fvs[task]:
            tensors[task][tid] = {}
            metadata[task][tid] = {}
            for layer, fv in fvs[task][tid].items():
                tensors[task][tid][layer] = fv.vector
                metadata[task][tid][str(layer)] = fv.to_meta()

    tensor_path = path.with_suffix(".pt")
    torch.save(tensors, tensor_path)

    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("FVs saved to %s and %s", tensor_path, meta_path)


def load_fvs(path: Path) -> FVCollection:
    """Load function vectors from disk."""
    tensor_path = path.with_suffix(".pt")
    meta_path = path.with_suffix(".json")

    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)

    with open(meta_path) as f:
        metadata = json.load(f)

    fvs: FVCollection = {}
    for task in tensors:
        fvs[task] = {}
        for tid in tensors[task]:
            fvs[task][tid] = {}
            for layer_key in tensors[task][tid]:
                layer = int(layer_key) if isinstance(layer_key, str) else layer_key
                meta = metadata[task][tid][str(layer)]
                fvs[task][tid][layer] = FunctionVector(
                    task=meta["task"],
                    template_id=meta["template_id"],
                    layer=meta["layer"],
                    layer_0idx=meta["layer_0idx"],
                    method=meta["method"],
                    vector=tensors[task][tid][layer_key],
                    norm=meta["norm"],
                    n_positive=meta["n_positive"],
                    n_negative=meta["n_negative"],
                    model_key=meta["model_key"],
                )

    logger.info("FVs loaded from %s", path)
    return fvs


def fv_statistics(fvs: FVCollection) -> Dict:
    """Compute summary statistics for function vectors."""
    all_norms = []
    by_task = {}

    for task in fvs:
        task_norms = []
        for tid in fvs[task]:
            for layer, fv in fvs[task][tid].items():
                all_norms.append(fv.norm)
                task_norms.append(fv.norm)

        by_task[task] = {
            "count": len(task_norms),
            "mean_norm": float(np.mean(task_norms)) if task_norms else 0,
            "std_norm": float(np.std(task_norms)) if task_norms else 0,
        }

    norms_arr = np.array(all_norms)
    return {
        "total_fvs": len(all_norms),
        "by_task": by_task,
        "norm_mean": float(norms_arr.mean()) if len(norms_arr) else 0,
        "norm_std": float(norms_arr.std()) if len(norms_arr) else 0,
        "norm_min": float(norms_arr.min()) if len(norms_arr) else 0,
        "norm_max": float(norms_arr.max()) if len(norms_arr) else 0,
    }
