# Stage 6D Replication Plan

## Purpose

Stage 6D is a focused ten-seed replication of the calibrated Stage 6 latent-compositional regime (`latent_d4_margin010`). It follows Stage 6C, which identified a clean, non-saturated benchmark and modest QFC OOD-composition plus stronger predictive-kernel signals.

## Pre-registered model families

### Classical comparators

Majority, logistic regression, sequence-aware MLP, random forest, extra trees, gradient boosting, corrected RBF-SVM, and degree-3 polynomial SVM.

### Quantum topology family and controls

Random sparse topology 01, ring, all-to-all, linear chain, grammar topology, no-entanglement control, and random-label-trained random topology 01 control.

## Primary evaluation

Balanced accuracy on ID test, held-out-composition OOD, and depth OOD. Paired deltas are reported against the strongest pre-registered classical comparator for each split, chosen by ten-seed mean after all registered models have been run. Bootstrap intervals describe uncertainty and do not by themselves establish significance.

## Output policy

To reduce package clutter, all diagnostics, run artifacts, generated configs, manifests, kernel outputs and summary tables are stored under `results/stage6d/`. All plots are stored directly under `figures/`.

## Claim boundary

The experiment may support a replication-level quantum-kernel or topology observation only if it remains stable under ten seeds and appropriate classical comparisons. It cannot establish quantum advantage, hardware utility or general QNLP validity.
