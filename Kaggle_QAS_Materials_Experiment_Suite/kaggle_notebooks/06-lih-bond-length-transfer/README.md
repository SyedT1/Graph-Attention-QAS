# LiH bond-length transfer

This folder contains the completed LiH transfer experiment and its exported Kaggle
artifacts. The experiment asks whether graph-based quantum architecture search (QAS)
surrogates can rank previously unseen particle-conserving VQE circuits at previously
unseen Li–H bond lengths.

The calculation uses a six-qubit Jordan–Wigner Hamiltonian for LiH in STO-3G with a
two-electron, three-active-orbital space. All reported errors are relative to exact
diagonalization **within this same active space**.

## Contents

| Path | Description |
|---|---|
| [`06-lih-bond-length-transfer.ipynb`](06-lih-bond-length-transfer.ipynb) | Self-contained Kaggle notebook with data generation, graph construction, model training, evaluation, plotting, and convergence auditing. |
| [`lih_bond_multiseed.csv`](<results (1)/qas_materials/lih_bond_multiseed.csv>) | Complete publication-mode label set: 810 VQE evaluations. |
| [`lih_transfer_model_summary.csv`](<results (1)/qas_materials/lih_transfer_model_summary.csv>) | Validation and macro-averaged held-out ranking metrics for four graph models. |
| [`lih_transfer_by_bond.csv`](<results (1)/qas_materials/lih_transfer_by_bond.csv>) | Held-out Kendall rank correlation at each test bond length. |
| [`lih_finalist_convergence_audit.csv`](<results (1)/qas_materials/lih_finalist_convergence_audit.csv>) | High-precision reruns of the three best held-out circuits at each test geometry. |
| [`__results___9_0.png`](<results (1)/__results___files/__results___9_0.png>) | Exported bond-difficulty and unseen-geometry ranking figure. |

## Experimental design

### Molecular labels

- Bond lengths: `1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2` Å.
- Architecture seeds: `11, 23, 37, 51, 79`.
- Architectures per seed: 18, giving 90 circuits at every geometry and 810 labels
  overall.
- Circuit depth: 3–8 parameterized single- and double-excitation gates.
- Base VQE optimization: 70 Adam steps, two restarts, noiseless exact state-vector
  simulation.
- The same restart initialization is reused along the bond scan for a fixed circuit,
  reducing optimizer-noise confounding across geometries.
- Exact energies are obtained by diagonalizing the two-particle sector. The supervised
  target uses only observable VQE energies; exact energies are used only to calculate
  benchmark gaps.

The saved dataset identifies itself as
`2026-09-v3-pauli-factor-graph`. Each row records the circuit gate list, VQE and exact
energies, gap, optimization budget, depth, and excitation counts.

### Transfer split

The evaluated split simultaneously holds out complete architecture pools and bond
lengths:

| Split | Architecture seeds | Bond lengths (Å) | Rows |
|---|---|---:|---:|
| Train | 11, 23, 37 | 1.0, 1.4, 1.8, 2.4, 3.2 | 270 |
| Validation | 51 | 1.2 | 18 |
| Test | 79 | 1.6, 2.0, 2.8 | 54 |

An assertion verifies that no exact circuit gate sequence crosses from the
training/validation side into the test architecture pool. Dataset rows outside these
three Cartesian subsets are retained in the label CSV but are not used by this model
comparison.

### Graph models

The notebook trains all models with seed `2026` and early stopping on validation
Kendall \(\tau\):

- **Circuit GAT:** circuit graph only.
- **Joint pairwise GAT:** circuit graph plus the legacy, lossy qubit-pair projection of
  the Hamiltonian.
- **Joint factor GAT:** circuit graph plus a lossless Pauli factor graph.
- **Joint factor GAT+KAN:** the same factor-graph encoder with a KAN prediction head.

The factor graph represents every aggregated non-identity Pauli word as a node, uses
axis-labelled term–qubit edges, and retains signed and absolute coefficient features.
The notebook reconstructs each molecular Hamiltonian from this representation and
requires a maximum coefficient error below `1e-10` before training. GAT attention also
uses interaction magnitude as an edge prior.

Training energies are standardized independently within each training Hamiltonian.
Rank correlations are computed within a bond length and then macro-averaged, avoiding
invalid comparisons of absolute energies across different Hamiltonians.

## Results

### Overall held-out ranking

| Model | Validation \(\tau\) | Epochs | Test \(\tau\) | Test Spearman \(\rho\) |
|---|---:|---:|---:|---:|
| Circuit GAT | 0.5033 | 43 | **0.2593** | **0.3664** |
| Joint pairwise GAT | **0.5686** | 134 | 0.2026 | 0.3058 |
| Joint factor GAT | 0.5033 | 68 | 0.1373 | 0.2274 |
| Joint factor GAT+KAN | 0.5556 | 54 | 0.2331 | 0.3622 |

The test values are macro averages over the three held-out Hamiltonians, with all
three bond-level Kendall correlations defined. In this run, circuit GAT ranks best on
both test metrics. Adding the KAN head substantially improves the factor-graph model
and approaches the circuit-only result, but the saved run does **not** demonstrate a
Hamiltonian-encoding advantage.

### Kendall \(\tau\) by held-out bond length

Each entry ranks the same 18 unseen seed-79 circuits.

| Model | 1.6 Å | 2.0 Å | 2.8 Å |
|---|---:|---:|---:|
| Circuit GAT | **0.1895** | **0.2288** | **0.3595** |
| Joint pairwise GAT | 0.1503 | 0.1895 | 0.2680 |
| Joint factor GAT | 0.0850 | 0.0980 | 0.2288 |
| Joint factor GAT+KAN | 0.1634 | 0.2026 | 0.3333 |

### Bond-stretch difficulty

| Li–H distance (Å) | Best gap (Ha) | Median gap (Ha) |
|---:|---:|---:|
| 1.0 | 8.276e-06 | 1.130e-03 |
| 1.2 | 1.003e-05 | 9.756e-04 |
| 1.4 | 1.085e-05 | 8.770e-04 |
| 1.6 | 1.682e-05 | 8.060e-04 |
| 1.8 | 2.719e-05 | 7.616e-04 |
| 2.0 | 4.932e-05 | 7.559e-04 |
| 2.4 | 2.095e-04 | 1.866e-03 |
| 2.8 | 1.126e-03 | 7.552e-03 |
| 3.2 | 1.236e-03 | 1.156e-02 |

The median circuit crosses the plotted 1.6 mHa active-space threshold at 2.4 Å and
worsens sharply as the bond stretches. The best circuit remains below that threshold
at every sampled geometry, showing that architecture choice becomes increasingly
important in the stretched regime.

![Best and median VQE gaps across the bond scan, followed by held-out Kendall tau for each graph model](<results (1)/__results___files/__results___9_0.png>)

### High-precision finalist audit

The three lowest-gap test circuits at each held-out geometry were rerun for 160 steps
and five restarts:

| Bond (Å) | Seed / architecture | Base gap (Ha) | Audited gap (Ha) |
|---:|---:|---:|---:|
| 1.6 | 79 / 12 | 3.329e-05 | 3.302e-05 |
| 1.6 | 79 / 5 | 2.279e-04 | 2.274e-04 |
| 1.6 | 79 / 4 | 2.281e-04 | 2.274e-04 |
| 2.0 | 79 / 12 | 1.014e-04 | 1.010e-04 |
| 2.0 | 79 / 5 | 5.792e-04 | 5.787e-04 |
| 2.0 | 79 / 4 | 5.794e-04 | 5.787e-04 |
| 2.8 | 79 / 9 | 1.747e-03 | 1.745e-03 |
| 2.8 | 79 / 12 | 2.764e-03 | 2.761e-03 |
| 2.8 | 79 / 6 | 3.256e-03 | 3.256e-03 |

Every audited gap decreases slightly. The largest absolute change is
`3.13e-06 Ha`, supporting the convergence stability of the selected finalists under
the larger optimization budget.

## Running the notebook on Kaggle

1. Upload the notebook as a Python notebook. It is self-contained and does not require
   cloning this repository.
2. Enable Internet for the first run if PennyLane is not already installed. The setup
   cell installs `pennylane>=0.40,<0.44` only when needed.
3. Leave `FAST_MODE = False` for the configuration documented above. `FAST_MODE = True`
   is only a pipeline check and writes a separately named `_fast.csv` dataset.
4. Run all cells. Outputs are written to `/kaggle/working/qas_materials/`. The label
   generator checkpoints after every bond length and can resume a compatible partial
   CSV; an attached Kaggle Dataset containing the CSV is detected automatically.
5. Download the four CSV files and the generated figure, or publish the output folder
   as a Kaggle Dataset for reuse.

The recorded publication run used Python 3.12.13, PennyLane 0.43.3, PyTorch
2.10.0+cu128, and a CUDA device for neural training. Label generation took about
4 hours 29 minutes in that run; exact state-vector and quantum-chemistry operations do
not all benefit from the GPU.

## Interpretation limits

- These are minimal-active-space errors, not complete-basis or experimental LiH
  errors.
- Canonical molecular orbitals are recomputed independently at each geometry. Orbital
  continuity should be checked before assigning a sharp transfer change solely to
  electronic correlation.
- Results come from noiseless state-vector simulation and do not establish robustness
  to hardware noise or quantum advantage.
- The neural comparison is one recorded training seed and one held-out architecture
  pool. Model-seed replication is needed before making a strong comparative claim.
- Validation uses only 18 circuits at one bond length, so model selection uncertainty
  is substantial.
