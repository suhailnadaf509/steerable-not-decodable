"""
Zero-shot residual-stream activation extraction
================================================
For each (task, template) we run the model on bare zero-shot prompts (no ICL
demonstrations, no steering) and cache the residual stream at the final token
position at every extraction layer.

This matches what the main pipeline's logit lens analyses (and the tuned lens)
operate on -- so the MLP probe results are directly comparable to those.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Imports from main package
from ..config import ExperimentConfig, MODEL_REGISTRY
from ..models import ModelWrapper
from ..tasks import TaskSpec, get_tasks

from .config import MLPDecoderConfig

logger = logging.getLogger(__name__)


# Cache layout:
#   activations[task_name][template_id]["acts_by_layer"][layer_1idx] -> Tensor[N, d_model] (cpu, bf16)
#   activations[task_name][template_id]["inputs"] -> List[str]  (the input strings)
#   activations[task_name][template_id]["correct_first_tokens"] -> Tensor[N] (long, cpu)
#   activations["__meta__"] -> dict with d_model, vocab_size, n_layers, layers_1idx, model_key, dtype


@torch.no_grad()
def extract_for_model(
    model_key: str,
    cfg: MLPDecoderConfig,
    log_every_n_tasks: int = 1,
) -> Dict:
    """
    Load a single model, extract zero-shot residual streams for every (task,
    template) at every extraction layer, save to disk, then unload.

    Returns the in-memory cache dict (also saved to disk).
    """
    cache_path = cfg.model_cache_path(model_key)
    if cfg.reuse_existing_cache and cache_path.exists():
        logger.info("Reusing existing activation cache for %s at %s",
                    model_key, cache_path)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    spec = MODEL_REGISTRY[model_key]
    logger.info(
        "Extracting activations for %s (%d layers, d_model=%d)",
        spec.display_name, spec.n_layers, spec.d_model,
    )

    # Build a minimal ExperimentConfig so ModelWrapper loads correctly.
    exp_cfg = ExperimentConfig(model_key=model_key)
    exp_cfg.ops.dtype = cfg.dtype
    exp_cfg.ops.extraction_batch_size = cfg.extraction_batch_size

    t_load = time.time()
    model = ModelWrapper(model_key=model_key, config=exp_cfg)
    logger.info("Model load took %.1fs", time.time() - t_load)

    # Determine layers to probe
    layers_1idx = cfg.layers_1idx or spec.extraction_layers()
    layers_0idx = [l - 1 for l in layers_1idx]
    logger.info("Probing layers (1-indexed): %s", layers_1idx)

    # Tasks
    tasks: Dict[str, TaskSpec] = get_tasks(cfg.tasks)
    tokenizer = model.tokenizer

    cache: Dict = {
        "__meta__": {
            "model_key": model_key,
            "display_name": spec.display_name,
            "family": spec.family,
            "n_layers": spec.n_layers,
            "d_model": spec.d_model,
            "vocab_size": int(model.model.W_U.shape[1]),
            "layers_1idx": layers_1idx,
            "dtype": cfg.dtype,
            "n_tasks": len(tasks),
        }
    }

    t_start = time.time()
    for ti, (task_name, task_spec) in enumerate(tasks.items()):
        cache[task_name] = {}

        for tid in task_spec.template_ids:
            template_str = task_spec.templates[tid]
            inputs = [inp for inp, _ in task_spec.pairs]
            outputs = [out for _, out in task_spec.pairs]
            prompts = [template_str.replace("{X}", inp) for inp in inputs]

            # Get activations at all layers in one batched forward pass.
            # ModelWrapper.get_activations returns last-token activations
            # already moved to cpu.
            acts = model.get_activations(
                prompts,
                layers_0idx,
                components=["resid_post"],
                batch_size=cfg.extraction_batch_size,
            )["resid_post"]

            # Convert keys back to 1-indexed and save in bf16 to save disk
            acts_by_layer: Dict[int, torch.Tensor] = {}
            for layer_0 in layers_0idx:
                if layer_0 not in acts:
                    continue
                t = acts[layer_0].to(torch.bfloat16).contiguous()
                acts_by_layer[layer_0 + 1] = t

            # First-token labels
            first_token_ids: List[int] = []
            for out in outputs:
                tok_ids = tokenizer.encode(out, add_special_tokens=False)
                if not tok_ids:
                    tok_ids = tokenizer.encode(" " + out, add_special_tokens=False)
                first_token_ids.append(tok_ids[0] if tok_ids else -1)

            cache[task_name][tid] = {
                "acts_by_layer": acts_by_layer,
                "inputs": inputs,
                "outputs": outputs,
                "correct_first_tokens": torch.tensor(first_token_ids, dtype=torch.long),
            }

        if (ti + 1) % log_every_n_tasks == 0 or ti + 1 == len(tasks):
            logger.info(
                "  [%s] task %d/%d done (%s) -- %.1fs elapsed",
                model_key, ti + 1, len(tasks), task_name,
                time.time() - t_start,
            )

    # Save to disk
    if cfg.cache_activations_to_disk:
        logger.info("Saving activation cache to %s", cache_path)
        torch.save(cache, cache_path)
        # Print disk size
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info("Cache size: %.1f MB", size_mb)

    # Unload model
    model.unload()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "Activation extraction for %s done in %.1fs",
        model_key, time.time() - t_start,
    )
    return cache


def extract_all_models(cfg: MLPDecoderConfig) -> Dict[str, Path]:
    """
    Extract activations for every model in cfg.models. Returns a dict mapping
    model_key -> path to the saved cache file.

    Models are processed sequentially so only one is on the GPU at a time.
    """
    paths: Dict[str, Path] = {}
    for mk in cfg.models:
        try:
            extract_for_model(mk, cfg)
            paths[mk] = cfg.model_cache_path(mk)
        except Exception:
            logger.exception("Failed to extract activations for %s", mk)
            raise
    return paths
