# Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens

This repository contains the official codebase for the paper **[Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens]** by Mohammed Suhail B Nadaf.

## Abstract

Function vectors (FVs)—mean-difference directions extracted from in-context learning demonstrations—can steer large language model behavior when added to the residual stream. We hypothesized that steering failures reflect information absence: the logit lens would fail alongside steering. In a cross-template FV transfer study spanning 4,032 pairs across 12 tasks, 6 models from 3 families (Llama-3.1-8B, Gemma-2-9B, Mistral-7B; base and instruction-tuned), and 8 templates per task, we find the opposite: **FV steering succeeds even when the logit lens cannot decode the correct answer at any layer.**

This *steerability-without-decodability* pattern is universal. FV vocabulary projection reveals that high-accuracy FVs project to incoherent tokens, indicating FVs encode *computational instructions* rather than answer directions. The dissociation is robust to tuned-lens dialect correction and confirmed by activation patching. Furthermore, post-steering analysis reveals a model-family divergence: Mistral FVs rewrite intermediate representations while Llama/Gemma FVs produce near-zero changes despite successful steering.

## Repository Structure

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

### Supported Models

- `llama-3.1-8b-base`
- `llama-3.1-8b-it`
- `gemma-2-9b-base`
- `gemma-2-9b-it`
- `mistral-7b-v0.3-base`
- `mistral-7b-v0.3-it`

## License

This project is released under the MIT License. See the `LICENSE` file for details.
