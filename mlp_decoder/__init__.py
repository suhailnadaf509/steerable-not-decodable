"""
MLP Decoder Probe Experiment
============================
Standalone follow-up experiment for the cross-template FV transfer paper.

Tests whether the *steerability-without-decodability* dissociation reported in
the main paper survives a stronger decoder. The main paper shows that the
parameter-free logit lens and per-layer diagonal tuned lens cannot decode the
correct task answer from the residual stream at the layers where FV steering
succeeds. The paper's own Limitations section flags the obvious next test:
nonlinear decoders. This sub-package implements that test.

For each (model, task, layer) we train a 2-layer MLP probe on zero-shot
residual-stream activations to predict the first token of the correct answer,
and a matched control probe on Hewitt & Liang shuffled labels. The
selectivity = real_top10 - control_top10 controls for raw decoder capacity.

Two outcomes both strengthen the paper:
  - MLP also fails -> the dissociation is not a linear-decoder artifact;
    information is invisible to bounded-capacity decoders.
  - MLP succeeds where logit/tuned lens fail -> FVs operate through
    nonlinearly encoded subspaces orthogonal to the unembedding.

This sub-package is *separate* from the main pipeline (does not modify
pipeline.py or any main-experiment artifacts). It reads JSONs produced by the
main pipeline (logit lens, tuned lens, steering) for downstream comparison.
"""

__version__ = "0.1.0"
