# Kaggle experiment suite

These notebooks implement the requested publication-strength follow-up experiments.
Each notebook embeds `kaggle_qas_core.py`; it can therefore run without cloning the
repository.

## Recommended order

1. `00_build_hubbard_multiseed_dataset.ipynb`
2. `01_multiseed_bootstrap_statistics.ipynb`
3. `02_unseen_architecture_and_parameter_test.ipynb`
4. `03_vqe_convergence_check.ipynb`
5. `04_trainable_gat_kan_ablation.ipynb`
6. `05_dense_hubbard_sweep_analysis.ipynb`
7. `06_lih_bond_length_transfer.ipynb`

Notebook 00 produces `hubbard_dense_multiseed.csv`. Download its contents from
`/kaggle/working/qas_materials/`, create a Kaggle Dataset, and attach that dataset to
notebooks 01–05 to avoid recomputing VQE labels. Notebook 06 similarly produces
`lih_bond_multiseed.csv` for reuse.

Every notebook can generate a missing CSV itself. Label generation saves a checkpoint
after each interaction value or bond length and resumes an incomplete file. A file made
with `FAST_MODE=True` has a separate `_fast.csv` name and cannot be mistaken for the
publication dataset.

## Publication configuration

- Hubbard: 24 circuits × 5 architecture seeds × 9 values of U/t = 1080 labels.
- Base VQE: 60 steps, 2 restarts.
- Convergence audit: selected top-five circuits, 160 steps, 5 restarts.
- LiH: 18 circuits × 5 seeds × 9 bond lengths = 810 labels.
- LiH base VQE: 70 steps, 2 restarts.
- Neural comparison: circuit GAT, Hamiltonian-only negative control, legacy joint
  pairwise GAT, lossless Pauli-factor GAT, Pauli-factor GAT+KAN, and scalar ridge.
- VQE restart initializations are paired across Hamiltonian parameters for each fixed
  architecture, reducing optimizer-noise confounding in transfer comparisons.
- Every aggregated Pauli word is a factor node. Axis-labelled term–qubit edges and
  coefficient features make the Hamiltonian graph reconstructible, while interaction
  magnitudes enter edge-conditioned GAT attention.
- Surrogates train on within-Hamiltonian standardized VQE energies using training rows
  only. Exact ground energies are used solely to report benchmark gaps.

Keep `FAST_MODE=False` for results used in an abstract or paper. Quick mode exists only
to validate execution and uses too little data for scientific conclusions.

## Interpretation safeguards

- Report bootstrap intervals across architecture seeds, not across individual circuit
  rows, because rows from one architecture pool are not independent replicates.
- Five-seed percentile bootstrap intervals are requested and reported, but are
  necessarily coarse; retain the individual seed results and standard deviation.
- Kendall and Spearman correlations are calculated within each Hamiltonian and then
  macro-averaged. Never pool energy gaps from different Hamiltonians into one ranking.
- Claim architecture generalization only from splits that hold out complete circuits.
- Claim a Hamiltonian-encoding benefit only if joint models consistently outperform
  circuit-only GAT across model seeds.
- Molecular gaps are relative to exact diagonalization in the same STO-3G active space;
  they are not complete-basis or experimental errors.
- The legacy weighted pairwise graph is retained only as an explicit ablation. Automated
  round-trip tests require the factor graph to reconstruct every aggregated Pauli word
  and coefficient with error below `1e-10` before training.
- Every VQE label is a noiseless state-vector result. Nothing in this suite establishes
  hardware-noise robustness or quantum advantage.
