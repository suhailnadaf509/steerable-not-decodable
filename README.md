# Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens

This repository contains the official codebase for the paper **[Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens]** by Mohammed Suhail B Nadaf.

## Abstract

Function vectors (FVs)—mean-difference directions extracted from in-context learning demonstrations—can steer large language model behavior when added to the residual stream. We hypothesized that steering failures reflect information absence: the logit lens would fail alongside steering. In a cross-template FV transfer study spanning 4,032 pairs across 12 tasks, 6 models from 3 families (Llama-3.1-8B, Gemma-2-9B, Mistral-7B; base and instruction-tuned), and 8 templates per task, we find the opposite: **FV steering succeeds even when the logit lens cannot decode the correct answer at any layer.**

This *steerability-without-decodability* pattern is universal. FV vocabulary projection reveals that high-accuracy FVs project to incoherent tokens, indicating FVs encode *computational instructions* rather than answer directions. The dissociation is robust to tuned-lens dialect correction (per-layer diagonal affine translators close only 1 of 14 gaps, 93% persist) and to a 2-layer nonlinear MLP probe with a Hewitt & Liang control (closes 5 of 10 SAND cells via nonlinearly encoded information; 5 remain invisible to every decoder tried). Activation patching causally confirms the difficulty hierarchy. Furthermore, post-steering analysis reveals a model-family divergence: Mistral FVs rewrite intermediate representations (delta up to +0.91) while Llama/Gemma FVs produce near-zero changes despite successful steering — evidence for two distinct mechanisms.

## Repository Structure

### Main pipeline
* `run_pipeline.py`: The main CLI entry point for running the extraction, steering, probing, and analysis pipeline.
* `tasks.py`: Definitions for the 12 tasks spanning 5 categories (Lexical, Factual, Morphological, Character, Compositional).
* `data.py`: Dataset generation, prompt formatting, and task-specific templates.
* `extraction.py`: Mean-of-differences FV extraction using HookedTransformer.
* `models.py`: Model loading, GPU profile configuration, and adaptive strength steering interventions.
* `readability.py`: Logit Lens projection and FV vocabulary projection logic.
* `tuned_lens.py`: Per-layer diagonal affine tuned-lens translators for dialect correction.
* `mechanistic.py`: Activation patching for causal localization.
* `analysis.py`: Geometric and statistical analysis (Simpson's paradox, hierarchical regression, transfer gaps).
* `visualization.py`: Generation of publication-quality figures from pipeline results.

### MLP decoder follow-up (`mlp_decoder/`)
A standalone sub-package implementing the nonlinear-decoder robustness check from §5.4 of the paper. Reads the main pipeline's JSON outputs (logit lens, tuned lens, IID steering) and decomposes the 14 steerable-not-decodable (SAND) cases into nonlinearly-encoded (recoverable) vs. truly-invisible.

* `mlp_decoder/config.py`: Configuration (models, tasks, layer spacing, probe hyperparameters, output paths).
* `mlp_decoder/activations.py`: Zero-shot residual-stream extraction across the 8 templates at every extraction layer.
* `mlp_decoder/probe.py`: 2-layer MLP probe (`LN → Linear(d_model→1024) → GELU → Dropout(0.1) → Linear(1024→|V|)`) with cosine LR schedule.
* `mlp_decoder/train_probes.py`: Per-(model, task, layer) training of one real-label probe and one Hewitt & Liang shuffled-label control. Splits by **unique input** to force input-generalization.
* `mlp_decoder/analyze.py`: Cross-decoder comparison (MLP vs logit lens vs tuned lens vs steering); populates the updated 4-bucket decoder ladder.
* `mlp_decoder/run_all.py`: End-to-end runner (`python -m fv_cross_template.mlp_decoder.run_all`); supports `--skip-extract`, `--skip-train`, `--skip-analyze`, `--skip-figures`.

The selectivity gate `real_top10 − control_top10 ≥ τ/2 = 0.05` is what prevents the probe from falsely declaring `object_color` decodable on every model by exploiting the small label alphabet.

## Setup & Installation

The framework requires PyTorch and Hugging Face Transformers. `TransformerLens` is used for mechanistic interpretability hooks.

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd fv_cross_template
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Authenticate with Hugging Face (Required for gated models like Llama-3.1 and Gemma-2):
   ```bash
   huggingface-cli login
   ```
   Ensure you have requested and been granted access to the respective models on the Hugging Face Hub.

## Usage

You can run the full cross-template function vector transfer pipeline using the CLI entry point `run_pipeline.py`.

### Examples

**Run the full pipeline on Llama-3.1-8B Base:**
```bash
python run_pipeline.py --model llama-3.1-8b-base
```

**Extract and steer only:**
```bash
python run_pipeline.py --model llama-3.1-8b-base --stages extract,steer
```

**Mechanistic analysis only (uses cached results):**
```bash
python run_pipeline.py --model llama-3.1-8b-base --stages mechanistic
```

**Run for specific tasks:**
```bash
python run_pipeline.py --model llama-3.1-8b-base --tasks antonym,synonym,past_tense
```

**Generate figures from saved results:**
```bash
python run_pipeline.py --stages figures
```

### MLP decoder probe

The MLP decoder follow-up is a separate runner that consumes the main pipeline's outputs. Run the main pipeline first, then:

```bash
# Full run: 6 models × 12 tasks × every-other-layer × {real, control} probes.
# Target wall-clock: 2–3 hours on a single H200.
python -m fv_cross_template.mlp_decoder.run_all \
    --output-dir outputs_mlp_decoder \
    --main-output-dir outputs

# Subset:
python -m fv_cross_template.mlp_decoder.run_all \
    --models llama-3.1-8b-base \
    --tasks antonym country_capital first_letter

# Reuse cached activations / probe results:
python -m fv_cross_template.mlp_decoder.run_all --skip-extract --skip-train
```

The runner writes `outputs_mlp_decoder/<model>/mlp_probe_results.json` per model and `outputs_mlp_decoder/decoder_comparison.json` for the cross-decoder ladder.

### Supported Models

- `llama-3.1-8b-base`
- `llama-3.1-8b-it`
- `gemma-2-9b-base`
- `gemma-2-9b-it`
- `mistral-7b-v0.3-base`
- `mistral-7b-v0.3-it`

## License

This project is released under the MIT License. See the `LICENSE` file for details.
