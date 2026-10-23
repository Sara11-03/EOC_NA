# Emergency Hub Placement with Neutral-Atom Quantum Computing

This repository contains the code and data supporting our research on **Emergency Operation Center (EOC) placement using neutral-atom quantum computing**.

We formulate the placement problem as a **Minimum Dominating Set (mDS)** problem and investigate a hybrid quantum–classical approach based on independent-set sampling and classical post-processing.

The repository includes numerical benchmarks on synthetic unit-disk graphs and two geographical case studies: **Palermo (41 nodes)** and **Modena (100 nodes)**, investigated using Pasqal's Fresnel quantum processor and classical emulation.

## Repository contents

- `eoc_na/`: graph algorithms, independent-set correction and mDS solvers.
- `eoc_geo/`: geographical preprocessing and atomic-register construction.
- `data/`: graph instances, archived quantum samples and experimental data.
- `scripts/`: benchmark, analysis and figure-reproduction scripts.
- `configs/`: geographical case-study configurations.
- `docs/`: reproduction instructions, methodology and experimental provenance.
- `tests/`: automated tests.

## Installation

Python 3.12 or later is recommended. From the repository root:

```bash
conda create -n eoc_repro python=3.12
conda activate eoc_repro
python -m pip install -r requirements.txt
python -m pip install -r requirements-cplex.txt
```

IBM CPLEX is required to recompute exact mDS solutions.

## Reproducing the results

All commands below use archived quantum measurements and do not require access to a QPU.

**Figure 3 — Synthetic benchmark**

```bash
python scripts/07_generate_synthetic_udgs.py --out-dir outputs/benchmark --lower 10 --upper 100 --step 5 --instances 30 --rescale historical
python scripts/10_make_fig3_cache.py --bench-dir outputs/benchmark --emu-dir data/synthetic_samples --out-dir outputs/recomputed_cplex --protocol paper20260505 --exact-solver cplex
python scripts/06_synthetic_approx_ratio_boxplot.py --run-dir outputs/recomputed_cplex --output-dir outputs/figure3 --strict
```

**Figures 4–5 — Independent-set sampling statistics**

```bash
python scripts/14_reproduce_is_statistics.py --bench-dir outputs/benchmark --samples-dir data/synthetic_samples --out-dir outputs/figures45
```

**Figures 6–7 and geographical case-study results**

```bash
python scripts/13_reproduce_case_studies.py --classical-samples 1000 --classical-seed 0 --out-dir outputs/case_studies
```

Generated figures and results are saved in `outputs/`, which is excluded from version control.

## Reproducibility notes

The repository provides archived QPU measurements, the corresponding graph instances, sampling configurations and classical post-processing algorithms. New hardware measurements are not required to reproduce the archived analyses.

The archived Modena QPU data yield **32 and 29 hubs** for the two post-processing branches, respectively. This differs from the value of 31 reported for the first branch in an earlier manuscript version. Experimental provenance and other historical clarifications are documented in [`docs/METHODS_AND_PROVENANCE.md`](docs/METHODS_AND_PROVENANCE.md).

For full instructions, see [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Authors

Sara Tarquini, Matteo Vandelli, Francesco Ferrari, Daniele Dragoni and Francesco Tudisco.
