"""
Model-Agnostic TransformerLens Wrapper
======================================
Handles loading models, discovering architecture properties, and providing
a uniform interface for hooks across different model families.
"""

from __future__ import annotations

import gc
import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformer_lens import HookedTransformer

from .config import ExperimentConfig, ModelSpec, MODEL_REGISTRY

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Wraps a TransformerLens HookedTransformer with model-agnostic utilities.

    Discovers model properties from the loaded model rather than hardcoding.
    Provides uniform hook name generation across architectures.
    """

    def __init__(
        self,
        model: Optional[HookedTransformer] = None,
        model_key: Optional[str] = None,
        config: Optional[ExperimentConfig] = None,
    ):
        if config is None:
            from .config import ExperimentConfig
            config = ExperimentConfig()

        self.config = config
        self.model_key = model_key or config.model_key
        self.spec = MODEL_REGISTRY[self.model_key]

        # -- CUDA performance flags (affect runtime, NOT scientific results) --
        if config.ops.device == "cuda" and config.ops.cuda_optimizations:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA optimizations enabled: tf32 matmul, cudnn benchmark")

        if model is not None:
            self.model = model
        else:
            logger.info("Loading model: %s (%s)", self.spec.display_name, self.spec.hf_name)
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            self.model = HookedTransformer.from_pretrained(
                self.spec.hf_name,
                device=config.ops.device,
                dtype=dtype_map.get(config.ops.dtype, torch.bfloat16),
            )
            self.model.eval()

        # Discover properties from the loaded model
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model
        self.n_heads = self.model.cfg.n_heads

        # Verify against spec
        if self.n_layers != self.spec.n_layers:
            logger.warning(
                "Model reports %d layers but spec says %d -- using model value",
                self.n_layers, self.spec.n_layers,
            )
        if self.d_model != self.spec.d_model:
            logger.warning(
                "Model reports d_model=%d but spec says %d -- using model value",
                self.d_model, self.spec.d_model,
            )

        logger.info(
            "Model loaded: n_layers=%d, d_model=%d, n_heads=%d, device=%s",
            self.n_layers, self.d_model, self.n_heads, config.ops.device,
        )

    # -- Hook name generation (model-agnostic) --

    def resid_post_hook(self, layer_idx: int) -> str:
        """Hook name for residual stream post at layer (0-indexed)."""
        return f"blocks.{layer_idx}.hook_resid_post"

    def attn_out_hook(self, layer_idx: int) -> str:
        """Hook name for attention output at layer (0-indexed)."""
        return f"blocks.{layer_idx}.hook_attn_out"

    def mlp_out_hook(self, layer_idx: int) -> str:
        """Hook name for MLP output at layer (0-indexed)."""
        return f"blocks.{layer_idx}.hook_mlp_out"

    def attn_pattern_hook(self, layer_idx: int) -> str:
        """Hook name for attention patterns at layer (0-indexed)."""
        return f"blocks.{layer_idx}.attn.hook_pattern"

    def layer_1indexed_to_0indexed(self, layer_1idx: int) -> int:
        """Convert 1-indexed layer (from config) to 0-indexed (for hooks)."""
        return layer_1idx - 1

    def extraction_layers_0indexed(self) -> List[int]:
        """Extraction layers as 0-indexed for hooks."""
        return [l - 1 for l in self.spec.extraction_layers()]

    @property
    def device(self) -> str:
        return self.config.ops.device

    @property
    def tokenizer(self):
        return self.model.tokenizer

    # -- Forward pass utilities --

    @torch.no_grad()
    def get_activations(
        self,
        prompts: List[str],
        layers_0idx: List[int],
        components: List[str] = ("resid_post",),
        batch_size: int = 8,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Get activations at final token position for given prompts and layers.

        Args:
            prompts: list of prompt strings
            layers_0idx: 0-indexed layer numbers
            components: which components ("resid_post", "attn_out", "mlp_out")
            batch_size: processing batch size

        Returns:
            results[component][layer_0idx] = tensor of shape [n_prompts, d_model]
        """
        # Build hook names
        hook_names = []
        hook_to_component_layer = {}

        for layer in layers_0idx:
            for comp in components:
                if comp == "resid_post":
                    name = self.resid_post_hook(layer)
                elif comp == "attn_out":
                    name = self.attn_out_hook(layer)
                elif comp == "mlp_out":
                    name = self.mlp_out_hook(layer)
                else:
                    raise ValueError(f"Unknown component: {comp}")
                hook_names.append(name)
                hook_to_component_layer[name] = (comp, layer)

        # Accumulate results
        results: Dict[str, Dict[int, List[torch.Tensor]]] = {
            comp: {layer: [] for layer in layers_0idx}
            for comp in components
        }

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tokens = self.model.to_tokens(batch, prepend_bos=True)

            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=hook_names,
                return_type="logits",
            )

            for name, (comp, layer) in hook_to_component_layer.items():
                if name in cache:
                    acts = cache[name][:, -1, :].cpu()  # [batch, d_model]
                    results[comp][layer].append(acts)

            del cache
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Concatenate batches (more efficient than unbind + stack)
        final: Dict[str, Dict[int, torch.Tensor]] = {}
        for comp in components:
            final[comp] = {}
            for layer in layers_0idx:
                if results[comp][layer]:
                    final[comp][layer] = torch.cat(results[comp][layer], dim=0)

        return final

    @torch.no_grad()
    def generate_with_hook(
        self,
        tokens: torch.Tensor,
        hook_name: str,
        hook_fn,
        max_new_tokens: int = 5,
    ) -> List[str]:
        """Generate text with a single hook applied."""
        prompt_length = tokens.shape[1]

        with self.model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            generated = self.model.generate(
                tokens.to(self.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                stop_at_eos=True,
                prepend_bos=False,
            )

        outputs = []
        for j in range(generated.shape[0]):
            new_tokens = generated[j, prompt_length:]
            text = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(text.strip())

        return outputs

    @torch.no_grad()
    def generate_multi_strength(
        self,
        tokens: torch.Tensor,
        hook_name: str,
        fv_tensor: torch.Tensor,
        strengths: List[float],
        max_new_tokens: int = 5,
    ) -> Dict[float, List[str]]:
        """
        Generate text with multiple steering strengths in a single batched call.

        Tiles the input batch across strengths: [n_strengths * batch, seq_len].
        A per-strength hook applies different multipliers to different batch slices.
        Results are IDENTICAL to calling generate_with_hook once per strength.

        Args:
            tokens: [batch, seq_len] tokenized prompts (already prepend_bos'd)
            hook_name: TransformerLens hook point name
            fv_tensor: function vector on device [d_model]
            strengths: list of steering multipliers
            max_new_tokens: how many tokens to generate

        Returns:
            Dict mapping each strength -> list of decoded output strings
        """
        n_strengths = len(strengths)
        if n_strengths == 0:
            return {}

        actual_batch = tokens.shape[0]
        prompt_length = tokens.shape[1]

        # Tile tokens: each strength gets its own copy of the batch
        tiled_tokens = tokens.repeat(n_strengths, 1)  # [n_strengths * batch, seq_len]

        # Vectorized hook: pre-compute per-sequence strength multipliers
        # [n_strengths * actual_batch, 1] — avoids Python loop in hot path
        strength_multipliers = torch.repeat_interleave(
            torch.tensor(strengths, device=self.device, dtype=fv_tensor.dtype),
            actual_batch,
        ).unsqueeze(-1)

        def multi_strength_hook(acts: torch.Tensor, hook) -> torch.Tensor:
            acts[:, -1, :] += strength_multipliers * fv_tensor
            return acts

        with self.model.hooks(fwd_hooks=[(hook_name, multi_strength_hook)]):
            generated = self.model.generate(
                tiled_tokens.to(self.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                stop_at_eos=True,
                prepend_bos=False,
            )

        # Split and batch-decode outputs (faster than per-sequence decode)
        new_token_seqs = generated[:, prompt_length:]
        all_decoded = self.model.tokenizer.batch_decode(
            new_token_seqs, skip_special_tokens=True,
        )
        results: Dict[float, List[str]] = {}
        for si, s in enumerate(strengths):
            start = si * actual_batch
            end = start + actual_batch
            results[s] = [t.strip() for t in all_decoded[start:end]]

        return results

    def cleanup(self):
        """Free GPU memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def unload(self):
        """Fully unload model from GPU memory (for multi-model pipelines)."""
        logger.info("Unloading model from %s", self.device)
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        if self.device == "cuda":
            logger.info(
                "GPU memory after unload: %.1f GB allocated",
                torch.cuda.memory_allocated() / 1e9,
            )
