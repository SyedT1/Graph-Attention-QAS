# Graph-Attention-QAS

A research project implementing **predictor-guided Quantum Architecture Search (QAS)** using a Graph Attention Network (GAT). The goal is to efficiently find high-quality quantum circuit architectures (ansätze) for Variational Quantum Algorithms (VQAs) without evaluating every candidate — instead training a graph-attention model to rank circuits from their structure alone.

The benchmark task throughout is finding the ground-state energy of the **Transverse-Field Ising Model (TFIM)** via VQE (Variational Quantum Eigensolver).

---

## Background

Variational Quantum Algorithms depend heavily on the choice of circuit architecture. A poor ansatz either can't represent the target state or makes optimization intractable through barren plateaus. On NISQ hardware, every additional two-qubit (CNOT) gate adds noise, so circuits should be **shallow and sparse in entangling gates** while remaining expressive enough to solve the problem.

Evaluating a single candidate circuit normally requires a full variational optimization to convergence — making exhaustive search prohibitively expensive. This project replaces most evaluations with a cheap **performance predictor**: a GAT trained on a modest set of (circuit, VQE energy) pairs that estimates the score of an unseen circuit from its graph structure alone.

---

## Notebooks

The project evolved across five notebooks, each version refining the approach:

### `graph_attention_QAS_executed.ipynb` — Original baseline (executed)
The initial, fully-executed prototype. Introduces the core pipeline end-to-end:
- TFIM Hamiltonian construction and exact ground-state reference energy
- Circuit search space: gate slots drawn from `{RX, RY, RZ, CNOT}`
- Circuit-to-DAG encoding: gates as nodes, qubit-wire dependencies as edges
- GAT predictor (from scratch in PyTorch): graph attention with 4 heads, hidden dim 32
- Dataset of 220 circuits labeled by quick VQE (60 steps, 3 restarts, random init)
- Supervised training with MSE + pairwise ranking loss (Kendall τ metric)
- Multi-objective guided search: score 4000 candidates, validate top 8 with full VQE
- Greedy structural pruning: remove gates whose removal doesn't raise energy beyond tolerance
- Comparison against random search at a matched budget of full VQE evaluations
- Publishability self-assessment checklist

**Key config:** `N_QUBITS=4`, `N_CIRCUITS=220`, `VQE_STEPS=60`, `LAMBDA_GATES=0.015`, `LAMBDA_CNOT=0.040`

---

### `graph-attention-qas-v2__1_.ipynb` — v2: Three research-grade improvements
Builds on the baseline with three major additions:

1. **Cleaner VQE labels** — layerwise initialization (parameters start near zero with small noise) + local cost functions (average single-qubit Z cost) for the first 30% of VQE steps to suppress barren plateaus, then switching to global energy
2. **ZX-calculus data augmentation** — generates equivalent circuit variants (spider fusion, identity removal, phase-free simplification) to multiply the labeled training set without additional quantum simulation
3. **Self-supervised pre-training (SSL)** — contrastive pre-training (NT-Xent / SimCLR-style) of the GAT encoder on 2000 unlabeled circuits before fine-tuning on the small labeled set

Results reported over 5 random seeds with mean ± std.

**Key config:** `N_CIRCUITS=220`, `VQE_STEPS=60`, `ZX_VARIANTS=3`, `SSL_CIRCUITS=2000`, `SSL_EPOCHS=60`, `SSL_LR=3e-3`, `SSL_TEMP=0.07`

---

### `qas-v3.ipynb` — v3: Bug fixes from v2
Addresses four bugs identified in v2:

| # | Bug | Fix |
|---|-----|-----|
| 1 | Double forward pass in `run_one_seed` (called `_bp(enc, tr_idx)` twice) | Cache `pred_tr` and reuse for ranking loss |
| 2 | `GAT-guided CNOT std = 0.000` — `LAMBDA_CNOT=0.040` drowned energy signal, always selecting 0-CNOT circuits | Reduce `LAMBDA_CNOT` 0.040→0.012, `LAMBDA_GATES` 0.015→0.008 |
| 3 | Publishability check `Demonstrates efficiency objective` always `False` even when pruning worked | Compare total gate depth instead of CNOT means |
| 4 | Non-reproducible rank-loss: used global `np.random.randint` instead of seeded RNG | Replace with `rng_np.integers` |

**Key config:** same as v2 with corrected `LAMBDA_GATES=0.008`, `LAMBDA_CNOT=0.012`

---

### `Graph-Attention-QAS-v4.ipynb` — v4: Deeper fixes to labels and search
Continues from v3 with five more targeted fixes:

| Fix | Change |
|-----|--------|
| **A** | Enforce `MIN_CNOTS=2` per circuit — TFIM ground state requires ZZ correlations; 0-CNOT circuits can't represent it |
| **B** | Increase `VQE_STEPS` 60→120, `VQE_RESTARTS` 3→4 for accurate labels (best previously only reached −4.48 vs E0=−4.76) |
| **C** | Increase `SSL_EPOCHS` 60→150, lower `SSL_LR` 3e-3→8e-4 with cosine decay (loss was plateauing at 2.77 vs random baseline 4.14) |
| **D** | Set `LAMBDA_GATES=0.0`, `LAMBDA_CNOT=0.0` — use energy-only acquisition + hard depth filter (`MAX_DEPTH_FILTER=16`) |
| **E** | Increase `N_CIRCUITS` 220→300, reduce `ZX_VARIANTS` 3→1 (previously 75% of training data was synthetic) |

**Key config:** `N_CIRCUITS=300`, `VQE_STEPS=120`, `SSL_EPOCHS=150`, `SSL_LR=8e-4`, `MIN_CNOTS=2`, `LAMBDA_GATES=0.0`, `LAMBDA_CNOT=0.0`

---

### `q5-bench__1_.ipynb` — v5 (benchmark): Sample-efficiency curve
Adds a sample-efficiency analysis on top of the v3/v4 pipeline:
- Raised SSL temperature `SSL_TEMP` 0.07→0.12 (SimCLR uses 0.07 with batch=256; at `SSL_BATCH=32` with only 31 negatives, 0.07 is too cold and causes early gradient plateau)
- Adds `BUDGET_EVALS = [1, 2, 4, 6, 8, 12, 16]` for plotting solution quality vs. number of full VQE validations — the standard sample-efficiency curve for predictor-based QAS

---

## Architecture

### Circuit Representation
Each quantum circuit is stored as a list of `(gate_type, qubit_tuple)` slots. Gate types: `{RX, RY, RZ, CNOT}`. The circuit is encoded as a **directed graph**:
- **Nodes** = gates, with features: gate-type one-hot + control/target flags + qubit one-hot + phase encoding `(sin φ, cos φ)`
- **Edges** = qubit-wire dependencies (bidirectional) + self-loops

### GAT Predictor
A custom Graph Attention Network (`hidden_dim=32`, `heads=4`) with:
- Two graph attention layers with ELU activations
- Global mean pooling to a fixed-size circuit embedding
- Regression head predicting VQE energy
- Training loss: MSE + pairwise ranking hinge loss (weight 1.0, margin 0.10, 256 random pairs/epoch)

### Self-Supervised Pre-training
SimCLR-style contrastive pre-training on unlabeled circuits:
- Augmentation: random gate-drop (probability 0.15)
- NT-Xent loss with temperature 0.07–0.12 depending on version
- 2000 unlabeled circuits, batches of 32

### VQE Evaluation
- PennyLane `default.qubit` simulator
- Adam optimizer, multiple restarts
- **v2+**: layerwise near-zero initialization + local cost (single-qubit Z) warm-up for the first 30% of steps to suppress barren plateaus

### Guided Search
1. Score 4000 randomly sampled candidate circuits with the cheap GAT predictor
2. Rank by acquisition function: `predicted_energy + λ_gates × depth + λ_cnot × num_cnots`
3. Validate top 8 candidates with full VQE
4. Apply greedy structural pruning to the winner

---

## Dependencies

```
pennylane
torch
networkx
scipy
numpy
matplotlib
```

Install automatically on first run:
```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pennylane", "torch", "networkx", "scipy"])
```

No GPU required. Default settings are tuned to complete in a few minutes on CPU.

---

## Configuration

All hyperparameters live in the `CFG` class at the top of each notebook. The most impactful levers for stronger results:

| Parameter | Default | Recommended (stronger) |
|-----------|---------|------------------------|
| `N_CIRCUITS` | 220–300 | 600–2000 |
| `VQE_STEPS` | 60–120 | 150–200 |
| `VQE_RESTARTS` | 3–4 | 5–6 |
| `N_QUBITS` | 4 | 8–12 |
| `SSL_EPOCHS` | 60–150 | 200+ |

---

## Results & Publishability

The pipeline demonstrates:
- Positive held-out Kendall τ rank correlation (predictor learns circuit quality from structure)
- GAT-guided search outperforms random search at a matched VQE evaluation budget
- Structural pruning reduces gate count while preserving energy accuracy

The self-assessment checklist in `graph_attention_QAS_executed.ipynb` identifies what remains for a publishable contribution: statistical rigor over multiple seeds, strong baselines (DQAS, RL, Bayesian), scale to 8+ qubits, multiple Hamiltonians, noise-model validation, ablations, and a sample-efficiency curve. The v5 benchmark notebook (`q5-bench`) begins addressing the last item.

---

## References

1. **QAS Survey** — arXiv:2406.06210
2. **He et al. (2024)** — Neural Predictor based on Graph Measures — doi:10.1002/qute.202400223
3. **GSQAS** — Graph Self-supervised QAS — arXiv:2303.12381
4. **SA-DQAS** — Self-attention Enhanced DQAS — arXiv:2406.08882
5. **QuantumDARTS** — Differentiable QAS with Gumbel-Softmax — PMLR
6. **ZX-calculus predictor** — EPJ Quantum Technology (2025) — doi:10.1140/epjqt/s40507-025-00410-w
7. **QAS-Bench** — Standardized QAS benchmarks — PMLR v202
8. **Awesome-QAS** — github.com/Aqasch/awesome-QAS
