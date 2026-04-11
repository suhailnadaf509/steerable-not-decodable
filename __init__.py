"""
fv_cross_template: Cross-Template Function Vector Transfer Analysis
===================================================================
A rigorous experimental framework for studying when the linear representation
hypothesis holds for steering via function vectors, and when it breaks.
Designed per EXPERIMENT_REDESIGN_SPEC.md targeting ICLR 2026 Workshop.

12 tasks x 8 templates x 16 layers = 1,536 FVs per model.

Modules:
    config      - Typed configuration with validation and justification
    tasks       - 12-task battery with 5-category taxonomy, 8 templates each
    data        - Prompt generation and dataset construction
    models      - Model-agnostic TransformerLens wrapper
    extraction  - Multi-method FV extraction (mean-diff, CAA, PCA)
    steering    - Additive steering + zero-shot/few-shot baselines
    analysis    - Geometric analysis, transfer matrices, style effects, permutation tests
    mechanistic - Activation patching (IID-gated)
    readability - Logit-lens readability analysis + FV vocabulary projection
    tuned_lens  - Tuned-lens calibration (learned per-layer translators)
    visualization - Publication-quality figure generation
    pipeline    - 8-stage orchestration with incremental caching
"""

__version__ = "0.3.0"
