# Methods and provenance

## Scope

This release replays saved raw samples; it does not regenerate a physical QPU
measurement. A new hardware execution will not return identical shots. Stored
reference results are read for comparisons only, not to choose computed answers.
The root README is intentionally empty pending the authors' final text.

The original manuscript needs two corrections for the archived Modena QPU data:
its direct small-maximal-IS branch gives 32 hubs, not 31 (or the 30 in the prose),
and the reported QPU pulse lasts 3000 ns, not 5400 ns. The complement branch is 29.
No data or algorithm has been changed to force agreement with the old values.

## Synthetic instances (Referee B1)

There are 30 instances at each N=10,15,...,100, or 570 graphs. The generator is a
sequential connected growth procedure, not uniform independent placement over a
disk. Its initial UDG radius is 7.186760677748386 micrometres. Minimum separation
is 5 micrometres; chamber radius is 46 micrometres, with a 0.5 micrometre placement
margin. Target average degree 2.5 controls the placement scale; it is not an exact
constraint on the final degree. Placement-scale trials are 1, 1.15, 1.35, 1.6,
1.9, 2.3. The implementation and attempt limits are in
`eoc_na/synthetic_udg_legacy.py` and `scripts/07_generate_synthetic_udgs.py`.

For instance i at size N and attempt t, the seed is
`numpy.random.SeedSequence([N, i, t]).generate_state(1, dtype=numpy.uint32)[0]`.
Each accepted seed and the subsequent historical uniform enlargement are saved
in the generated metadata. No historical reference CSV is needed to generate
the instances. Original archived geometries are included as JSON, preserving
node insertion order. Script 08 writes observed degree, clustering, diameter,
leaf fraction and geometric statistics. This is a controlled family of sparse,
connected, device-compatible UDGs, not a representative sample of all UDGs.

## Geographical instances (Referees A3/A5)

Fixed selected municipality inputs are in `data/inputs`. The upstream algorithm
that selected the initial 112 Modena and 50 Palermo points is not available;
the selected points themselves are archived. Do not replace them with a new
selection when reproducing the case studies.

Geographical edges use Haversine thresholds of 11.26 km (Modena) and 13.5 km
(Palermo). Coordinates are projected to EPSG:32632, expressed in km, centered
at their mean and initially scaled by 8.5/11.26 and 10/13.5 micrometres/km.
The code checks close points, clips outlying vertices using a 46 micrometre
radius, recenters, and uniformly scales the survivors to a minimum separation
of 5.0001 micrometres (with the archived floating-point guard).
No point merging occurs for these two inputs. Clipping removes seven Modena
and nine Palermo vertices. Modena then removes two high-degree and three
peripheral vertices while preserving connectivity to reach 100 atoms.
There is no additional hardware maximum-degree constraint. These last removals
are not another correction of the chamber radius. The final Modena graph has
100 vertices/145 edges and Palermo has 41 vertices/56 edges. All intermediate
stages and the removed vertices are written by the geographic CLI.

## Classical correction and refinement (Referees A3/A5)

`paper20260505` preserves the historical single-pass algorithm. The alternative
`clean-v1` is not the protocol used to reproduce the paper. Samples are never
repaired before computing the raw validity histograms.

At each repair step, conflict degree is recomputed in the currently selected
set. A conflicted vertex with maximum conflict degree is removed. For the small
maximal-IS branch, ties favor removal of a lower full-graph degree; for the
large branch, a higher degree. Remaining ties use the string representation of
the vertex label. Extension chooses feasible vertices with maximum degree for
the small branch and minimum degree for the large branch, again with label
tie-breaking as implemented in `repair_to_targeted_maximal_is`.

The degree modes do not draw random numbers. Reproducibility still requires the
same labels and node order. Case-study graphs are relabelled to integers in the
archived bit order before post-processing. Refinement first tries removals in
ascending degree order (stable graph insertion order for ties), then performs
one two-neighbourhood sweep in insertion order, not iterations to convergence.

Figure 3 and case-study classical sampling use NetworkX maximal independent
sets with seed `seed_offset + sample_index` (offset 0). Figure 4 used a separate
seed convention: `12345 + 1000000*N + 10000*instance + sample_index`.
Figure 3 uses 500 samples per method/instance. Case-study replay uses 500
classical samples and all 1000 archived quantum samples per case/backend,
as in the recovered scripts. The old manuscript's reference to 1000 classical
samples must be reconciled separately, not silently imposed on this replay.

## CPLEX (Referee A5)

CPLEX is the default exact MDS solver. There is no automatic fallback. The
model has one binary variable per vertex, minimizes their sum, and requires
at least one selected vertex in each closed neighbourhood. Unless `--time-limit`
is supplied, no time limit is set. Tolerances, threads, parallel mode and solver
seed remain at the installed CPLEX defaults, as in the recovered solver source.

Scripts 10 and 13 write the installed CPLEX/DOcplex versions, effective parameter
values and defaults, LP model, solver status, bound, gap and solution values.
A feasible incumbent alone is not accepted as an exact optimum. CPLEX execution
was not tested in the preparation environment because its runtime was unavailable.

These are records of a NEW replay run. The historical installed CPLEX version
and complete old logs were not recovered. Installing CPLEX now does not recover
those records. Alternative optimal vertex sets may exist; equality of optimum
cardinality does not require identical selected municipalities.

The Figure 4 exact reference is different: it is an exact maximum independent
set computed by the historical maximum-clique branch-and-bound routine on the
complement. Its optimum flag is checked. Means for N<=65 are rounded to two
decimals for the historical plotting convention. The legend is corrected to
"maximum IS (exact)".

## Quantum data and pulse information (Referees A3/A5)

The case-study raw counts, per-job IDs and saved sample lists are in
`data/realistic_case_studies`. `data/qpu` contains explicit bit-to-node mappings
and diagnostic references. Histograms preserve multiplicities. Their ordering
is not asserted to be acquisition chronology.

The confirmed archived QPU jobs are:
- Modena: 70fc3039-da71-4480-9238-fd3c438c0a0e (1000 shots).
- Palermo: 78a7b8e9-1915-4efc-860a-7a61844b1943 (1000 shots).

Their recorded pulse parameters are in `data/parameters`. Both have 500/1500/1000
ns rise/sweep/fall durations, totaling 3000 ns. Initial/final detunings differ
between the cases. The synthetic raw samples are in `data/synthetic_samples`;
metadata identify the original diagnostic CSV, per-instance job IDs, graph
fingerprint and sample index ordering. They are not newly emulated data.

A Palermo sequence reconstituted from the supplied portal export is in
`data/pulser`, with its provenance. A complete original Modena serialized
sequence and per-job synthetic/emulator configurations were not supplied in
this ZIP. The new-submission helpers therefore cannot certify a fresh replay
of the historical quantum dynamics. They require explicit submission and are
not called by any offline reproduction command.

The complete executed pulse-search grids and all their results are not archived
here. Do not describe the final-parameter table as the optimization range. This
release does not fill that gap with a guessed grid.

## Calibration and measurement treatment (Referee A3)

The Palermo device snapshot supplies a nominal default noise model: readout
false-positive probability 0.025, false-negative probability 0.10, dephasing
rate 0.2222222222222222 per microsecond and relaxation rate 0.01 per microsecond.
These are not time-stamped calibrations measured for each job. No run-specific
calibration report is included. Nor does the snapshot prove that these noise
parameters were used in the archived emulation.

The replay applies no readout-mitigation model. It analyses the archived complete
binary counts before combinatorial correction. Partial-register outcomes (X)
are not supplied for all jobs in this release; these 1000-shot statistics must
not be interpreted as including every physical loading attempt. Any provider
filtering and loss treatment not recorded in the source exports remain unknown.
A software seed in submission metadata cannot make QPU measurements repeatable.

## Figure scope

Scripts 06, 13 and 14 reproduce numerical data and plots for Figures 3-7.
Saved PDF bytes, rendering details and fonts are not expected to be bit-identical
across platforms. The geometry scripts reconstruct the data behind Figure 1,
not its original satellite imagery or assembled inset layout. Figure 2 is a
schematic, and its editable source is not present. Table 1 contains definitions,
not computed data. Table 2 is regenerated using the supported value 32 for the
Modena QPU direct branch. Correct the manuscript before claiming agreement.
