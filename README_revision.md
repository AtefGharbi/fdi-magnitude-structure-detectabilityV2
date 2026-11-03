# Revision code (Frontiers in Neuroscience, round 1)

Adds the analyses requested by the reviewers and corrects the controlled experiment.

| File | Purpose |
|---|---|
| `common.py` | Data generation (same RNG order as the original scripts; network built once per process for speed). Includes the original H1 constructions, the previous factorial generator, and the clean operating-point bank used by the corrected design. |
| `detectors.py` | Random Forest, published physics-informed detector (`train_pi`, with diagnostic variants), diagonal-precision and full-precision (Ledoit–Wolf Gaussian likelihood-ratio) readouts. |
| `structcheck.py` | Shows the previous factorial "independent" and "correlated" conditions are identical in distribution (Supplementary Note S1). |
| `run_v2.py` | Corrected controlled experiment (`main`, 20 seeds), channel-count sweep (`ksweep`, 8 seeds), and dispatch of detector diagnostics. Run `python3 run_v2.py` (2 worker processes). |
| `run_all.py` | Physics-informed detector diagnostics on the motivating-example data (`diag`). |
| `theory.py` | Exact ideal-observer AUC versus ρ, k and SNR (isotropic noise); writes `theory.json`. |
| `kdump.py` | Learned vs. least-squares susceptances for seed 0 (Figure 7a). |
| `analyze.py` | Statistics: Wilcoxon, 95% CI, d_z, JZS Bayes factor, TOST (±0.02 AUC); writes `analysis.json`. |
| `figs_new.py` | Figures 4–7 of the revised manuscript; `fig3.py` regenerates Figure 3. |
| `results_v2/`, `results/` | Per-seed JSON results. |

Corrected design: for each system/seed/k, a fixed channel set S and sign pattern u are shared by all conditions; deviation on S is
`t * (sqrt(1-rho) * z + sqrt(rho) * c * u)`, so every affected channel is N(0, t^2) in every condition and only cross-channel correlation differs.
Each condition is a separate binary task against the same 1,200 normal samples.

Requirements: pandapower, numba, numpy, scipy, scikit-learn, torch, matplotlib.
