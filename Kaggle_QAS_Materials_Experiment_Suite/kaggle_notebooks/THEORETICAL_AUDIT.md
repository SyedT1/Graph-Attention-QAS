# Theoretical consistency audit

Audit date: 2026-09-06

## Shared physical model

- The two-site spinful Hubbard ordering is `(0 up, 1 up, 0 down, 1 down)`.
  Adjacent same-spin hopping maps under Jordan-Wigner to
  `-t(XX+YY)/2`; onsite terms use `U(I-Z_up-Z_down+Z_up Z_down)/4`.
- Exact diagonalization is restricted to the two-particle sector. Every generated
  Hubbard reference is checked against `(U-sqrt(U^2+16t^2))/2`.
- Single- and double-excitation gates preserve Hamming weight. They are qubit
  excitation gates, not a claim of exact fermionic-UCC generators.
- LiH uses a neutral STO-3G Hamiltonian with a `(2 electrons, 3 spatial orbitals)`
  active space and Jordan-Wigner mapping. Exact energies and reported gaps refer only
  to that active-space Hamiltonian.

## Shared statistical and machine-learning design

- Architecture pools use five independently seeded random streams and are explicitly
  de-duplicated across pools.
- A fixed architecture receives paired VQE parameter initializations across all
  Hamiltonian parameters or bond lengths.
- Surrogate targets are within-Hamiltonian standardized *VQE energies*, constructed
  from training labels only. Exact ground energies are evaluation-only.
- Rank-loss pairs are formed only between circuits evaluated on the same Hamiltonian.
- Kendall and Spearman results are computed within Hamiltonian and macro-averaged.
- Every non-identity Pauli word is represented by its own factor node. Its signed
  coefficient is stored on the node and on incident edges, and each term–qubit edge is
  labelled X, Y, or Z. A global node stores the identity coefficient and scale.
- Edge-conditioned attention uses relation type, Pauli axis, signed coefficient, and
  absolute coefficient in both attention logits and messages.
- Exact circuit fingerprints are checked across every architecture-held-out split.

## Notebook-specific checks

### 00 — dataset

Checks the analytic Hubbard result, variational bound, finite labels, complete dense
parameter grid, consistent circuit identity across parameters, and absence of exact
duplicates across architecture pools. It also reconstructs every Hubbard Hamiltonian
from its factor graph with coefficient error below `1e-10`. Interrupted calculations
checkpoint after every interaction value.

### 01 — multiple seeds

Each replicate holds out both circuit IDs and U/t values. The reported replicate score
is the mean of within-U/t Kendall correlations. Bootstrap resampling is performed over
the five end-to-end replicate scores, not over correlated circuit rows.

### 02 — factorial transfer

Separately measures seen-circuit/unseen-Hamiltonian, unseen-circuit/seen-Hamiltonian,
and unseen-both quadrants. Metrics are macro-averaged within U/t; different Hamiltonian
energy scales are never pooled into one ranking.

### 03 — convergence

The top-five circuits per seed at U/t in `{0,4,8}` are rerun with 160 steps and five
restarts. Kendall stability is explicitly a *leading-set* diagnostic, not an estimate
for the entire architecture population.

### 04 — GAT/KAN ablation

All models use the identical data split and label budget. The legacy lossy pairwise graph
is compared directly with the Pauli-factor graph. Hamiltonian-only is a negative control
whose within-Hamiltonian ranking is mathematically undefined because every circuit
receives the same input. GAT+KAN is not parameter matched to the MLP head; trainable
parameter counts are reported.

### 05 — dense sweep

All rank-stability entries use paired architectures across U/t. The heatmap is
descriptive and makes no extrapolation claim.

### 06 — LiH transfer

Bond and architecture identities are both held out. Every molecular factor graph must
pass Pauli-word reconstruction below `1e-10`. Molecular finalists receive an additional
160-step/five-restart audit. The notebook states that independently computed canonical
orbitals are not overlap-tracked across geometries.

## Remaining limitations, not inconsistencies

- Five architecture seeds yield a coarse percentile-bootstrap interval; individual
  seed results and sample standard deviation must accompany it.
- Base VQE labels can remain optimizer dependent; notebook 03 and the molecular
  finalist audit quantify this for selected circuits.
- Factor-graph size grows with the number of Pauli words, which can become expensive for
  larger active spaces even though no Hamiltonian information is discarded here.
- LiH uses a minimal active space and canonical orbitals may reorder along the scan.
- State-vector, zero-shot simulations do not establish shot-noise robustness, hardware
  performance, scalability, or quantum advantage.
- U/t test points 5 and 7 are held-out interpolation points inside the training range,
  not out-of-range extrapolation.
