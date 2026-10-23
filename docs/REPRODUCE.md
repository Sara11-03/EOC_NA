# Reproduce the archived analyses

Run commands from the repository root. Use Python 3.12 or later and a working
IBM CPLEX runtime. No offline command submits a quantum job or requires Pasqal
credentials. Read `METHODS_AND_PROVENANCE.md` before interpreting the results.

## Environment

```text
conda create -n eoc_repro python=3.12
conda activate eoc_repro
python -m pip install -r requirements.txt
python -m pip install -r requirements-cplex.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

CPLEX/DOcplex versions depend on the runtime installed by the user. They are
recorded in each exact-solver audit. Install the appropriate licensed runtime
when the pip distribution is not the runtime used by your institution. CPLEX
was unavailable in the preparation environment; its integration still requires
this local test. A skipped CPLEX test is not evidence of a successful solver run.

## Figure 3

Generate 30 instances per size, without reading an expected-result file:

```text
python scripts/07_generate_synthetic_udgs.py --out-dir outputs/benchmark --lower 10 --upper 100 --step 5 --instances 30 --rescale historical
python scripts/08_audit_synthetic_udgs.py --bench-dir outputs/benchmark
```

Recompute the exact MDS with CPLEX and both heuristic branches. Quantum inputs
are the archived raw samples; exact and heuristic answers are recomputed.

```text
python scripts/10_make_fig3_cache.py --bench-dir outputs/benchmark --emu-dir data/synthetic_samples --out-dir outputs/recomputed_cplex --protocol paper20260505 --exact-solver cplex
python scripts/06_synthetic_approx_ratio_boxplot.py --run-dir outputs/recomputed_cplex --output-dir outputs/figure3 --strict
```

Only the final comparison reads the archived solution sizes:

```text
python scripts/12_compare_fig3_runs.py --historical data/synthetic_case_studies/result_caches/results_not_corr/run_20260505_113509 --new outputs/recomputed_cplex --out outputs/comparison.csv --require-identical --require-full-coverage
```

Expected coverage is 570 instances and 2850 cardinalities (one exact and four
heuristic values per instance). Preparation tests independently verified the
2280 heuristic values; CPLEX exact recomputation remains a local test.
The plot displays every second size, as in the manuscript. Numerical results
for all 19 sizes remain available. Use `--resume` on script 10 after an
interruption; do not mix protocols or solvers in one output directory.

## Figures 4 and 5

```text
python scripts/14_reproduce_is_statistics.py --bench-dir outputs/benchmark --samples-dir data/synthetic_samples --out-dir outputs/figures45
```

This recomputes raw quantum statistics, the historical classical IS sample pool,
and a certified maximum-IS reference using the original branch-and-bound code.
It does not use CPLEX for the maximum-IS reference. A timeout without an optimum
certificate stops the calculation. Use `--resume` to continue saved instances.
The corrected reference label is `maximum IS (exact)`.

## Figures 6 and 7, case-study table, QPU parameters

```text
python scripts/13_reproduce_case_studies.py --out-dir outputs/case_studies
```

This rebuilds both graphs, applies the archived bit-to-node mapping, reads raw
EMU/QPU counts, and recalculates the heuristics and exact MDS with CPLEX.
Outputs include figures, `results.csv`, `table_case_studies.tex`,
`qpu_parameters.csv`, per-case solver records and `verification.json`.

Expected rows (direct branch, complement branch):

| Case | Emulation | QPU | Exact MDS |
| --- | --- | --- | --- |
| Palermo | 13, 12 | 13, 12 | 12 |
| Modena | 31, 29 | 32, 29 | 28 |

The QPU Modena direct branch does not reproduce the old manuscript's 31 or 30.
Do not change samples or tie-breaking to force that value. Both archived QPU
pulse durations are 3000 ns. Historical source discrepancies are documented
in `METHODS_AND_PROVENANCE.md`.

`--plots-only` avoids all MDS computations. `--skip-exact` runs the heuristics
without CPLEX and deliberately leaves exact values empty; it is not a complete
reproduction of the table. No result is silently filled from the reference.

## Geographical graphs

```text
python scripts/run_geographical_demo.py --case modena --out-dir outputs/geographical/modena
python scripts/run_geographical_demo.py --case palermo --out-dir outputs/geographical/palermo
```

These commands use archived geographical inputs without network access. They
save intermediate graphs, coordinate transforms, removed vertices, maps,
atomic-register views and validation records. They reconstruct Figure 1's
graphs, not its original satellite imagery or page composition. Source download
is separately available through `geo_00_download_coordinates.py --help`.

## Fresh quantum executions

The scripts 03, 09 and 11 are optional cloud interfaces. They are not called
by the commands above. Install `requirements-emulation.txt` separately only
when needed. A live cloud run was not tested during release preparation.
Complete historical configurations are not available for every job; new
emulations must not be described as an exact replay of archived quantum runs.

## Before a commit

```text
python tools/check_release.py
git add .
python tools/check_release.py --staged
git diff --cached --stat
```

The scanner checks common secret/path patterns; it does not guarantee absence
of all sensitive content. Do not commit credentials, local environments,
original personal backups or generated `outputs/`. The root README is empty
by request. Choose an appropriate license with the coauthors before a public
release; preserve the geographical-data attribution.
