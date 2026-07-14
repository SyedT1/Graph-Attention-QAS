# Graph-Attention-QAS

A research notebook series implementing **predictor-guided Quantum Architecture Search (QAS)** with graph neural networks (GAT, GCN, GIN, KAN heads). The goal is to find good quantum circuit architectures (ansätze) for Variational Quantum Algorithms (VQAs) without running a full variational optimization on every candidate — instead, a graph neural network is trained to rank circuits from their structure alone, and expensive simulation is reserved for the few candidates the predictor likes best.

The primary benchmark is the ground-state energy of the **Transverse-Field Ising Model (TFIM)** via a Variational Quantum Eigensolver (VQE). Later notebooks add the **Heisenberg model**, and **v8** adds **MaxCut** (combinatorial / QAOA-style) plus a **depolarizing noise** check.

This repository contains **eleven notebooks**. Early files form a branching development history from baseline through two parallel v5/v6 branches. Later notebooks consolidate and extend that work:

| Milestone notebook | Role |
|--------------------|------|
| `8qubitrun.ipynb` | **v6 unified** — merges the `QAS-v6` and `DQAS+KANQAS` branches; fully executed multi-seed results at 4 qubits |
| `QASv7.ipynb` | **v7** — replaces REINFORCE-DQAS with a stronger **GIN + Bayesian Expected Improvement (GIN-BO)** baseline |
| `QASv8.ipynb` | **v8 production** — scales to **8 qubits**, adds depolarizing noise validation and a MaxCut third task |

**Recommended starting point for new work or citation of latest numbers: `QASv8.ipynb`.** For the cleanest 4-qubit multi-method comparison with DQAS still in the loop, use `8qubitrun.ipynb`.

This README treats the files as siblings rather than forcing a false linear order, derives equations from the code that implements them, and is explicit about which numbers come from stored, internally consistent execution output.

---

## Contents of this Repository

| File | Internal version label | What it adds |
|------|------------------------|--------------|
| `graph_attention_QAS_executed.ipynb` | v1 (baseline) | Original end-to-end pipeline |
| `graph-attention-qas-v2 (1).ipynb` | v2 | Cleaner VQE labels, ZX-calculus augmentation, SSL pre-training, multi-seed stats |
| `qas-v3.ipynb` | v3 | Four bug fixes to v2 |
| `Graph-Attention-QAS-v4.ipynb` | v4 | Five further fixes to the search space and labels |
| `q5-bench (1).ipynb` | "v4" (parallel branch) | GCN-vs-GAT ablation, sample-efficiency curve, Heisenberg cross-task check, joblib caching |
| `q5-optimized.ipynb` | v4→v5 | Layered hardware-efficient ansatz; pruning-bug fix |
| `QAS-v6.ipynb` | "v5" (branch from `q5-bench`) | Promotes GCN to primary predictor; 165-point circuit-level Spearman ρ |
| `DQAS+KANQAS.ipynb` | "v6" (branch from `q5-bench`) | KAN regression head + REINFORCE-DQAS baseline (stale multi-seed aggregate cell — see §10) |
| `8qubitrun.ipynb` | **v6 unified** | Merge of both v6 branches; KAN + DQAS in multi-seed loop; **executed** multi-seed results |
| `QASv7.ipynb` | **v7** | Replaces REINFORCE-DQAS with **GIN-BO** (GIN encoder + MC-Dropout + EI acquisition) |
| `QASv8.ipynb` | **v8** | **8-qubit TFIM**, depolarizing noise (`p=0.01`), **MaxCut** third task; GIN-BO retained |

All notebooks share the same skeleton (Hamiltonian → circuit search space → graph encoding → GNN predictor → guided search → pruning → baselines → publishability checklist). Sections below build the mathematical pipeline once, then report what changed numerically at each version.

---

## 1. Background and Problem Statement

A VQA prepares a parameterized state

$$|\psi(\boldsymbol\theta)\rangle = U(\boldsymbol\theta)\,|0\rangle^{\otimes n}, \qquad U(\boldsymbol\theta) = \prod_{l=1}^{L} U_l(\theta_l)$$

where each $U_l$ is either a single-qubit rotation $RX(\theta)$, $RY(\theta)$, $RZ(\theta)$ (one trainable angle) or a fixed two-qubit $\mathrm{CNOT}$ (no trainable angle). The cost function to minimize is the energy expectation value

$$C(\boldsymbol\theta) = \langle\psi(\boldsymbol\theta)|H|\psi(\boldsymbol\theta)\rangle = \langle 0|^{\otimes n}\,U(\boldsymbol\theta)^\dagger\, H\, U(\boldsymbol\theta)\,|0\rangle^{\otimes n}$$

for a target Hamiltonian $H$. The *architecture* is the discrete choice of which gates appear, on which qubits, in what order — i.e. the sequence $\{(\mathrm{gate}_l, \mathrm{wire}_l)\}_{l=1}^L$ independent of the continuous angles $\boldsymbol\theta$. A poor architecture either cannot represent the target state at all (insufficient expressivity) or makes the loss landscape pathological — gradients $\nabla_{\boldsymbol\theta} C$ that vanish exponentially in $n$ almost everywhere (a *barren plateau*, McClean et al. 2018). On NISQ hardware there is an extra pressure: every two-qubit gate injects physical noise, so the search wants circuits that are simultaneously low-energy, shallow, and CNOT-sparse.

**Quantum Architecture Search (QAS)** is the bi-level optimization

$$\mathrm{arch}^\star = \arg\min_{\mathrm{arch}} \;\min_{\boldsymbol\theta} C_{\mathrm{arch}}(\boldsymbol\theta)$$

over the *outer* discrete architecture space, where each evaluation of the objective requires solving the *inner* continuous VQE optimization to convergence. The idea across this series is to replace most of those inner-loop evaluations with a cheap **learned performance predictor** $\hat{E}_\phi(\mathrm{arch})$ — a graph neural network trained on a modest number of true (architecture, VQE energy) pairs — and reserve the expensive inner loop for only the handful of architectures the predictor ranks best.

---

## 2. The Benchmark Hamiltonians

### 2.1 Transverse-Field Ising Model (TFIM)

Used in every notebook, on an open chain of $n$ qubits:

$$H_{TFIM} = -J\sum_{i=0}^{n-2} Z_i Z_{i+1} \;-\; h\sum_{i=0}^{n-1} X_i$$

with $J = h = 1.0$. $H$ is built as a PennyLane `qml.Hamiltonian`, converted to a dense matrix, and diagonalized exactly via `numpy.linalg.eigvalsh`:

$$E_0 = \lambda_{\min}\bigl(H_{TFIM}\bigr).$$

| Setting | $n$ | $E_0$ (stored output) | Notebooks |
|---------|-----|------------------------|-----------|
| Default series | 4 | $-4.758770$ | v1–v7, `8qubitrun` |
| Scaled | 8 | $-9.837951$ | **`QASv8.ipynb` only** |

The **energy gap** is

$$\Delta E(\mathrm{arch}) = C_{\mathrm{arch}}(\boldsymbol\theta^\star) - E_0 \;\ge\; 0$$

where $\boldsymbol\theta^\star$ is the VQE-optimized parameter vector for that architecture; $\Delta E = 0$ iff the circuit reaches the exact ground state.

### 2.2 Heisenberg Model (XXX Chain)

Added in `q5-bench`, `QAS-v6`, `DQAS+KANQAS`, `8qubitrun`, `QASv7`, and `QASv8` as a second task:

$$H_{\mathrm{Heis}} = -J\sum_{i=0}^{n-2}\bigl(X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1}\bigr)$$

| $n$ | $E_0^{\mathrm{Heis}}$ |
|-----|------------------------|
| 4 | $-3.000000$ |
| 8 (`QASv8`) | $-7.000000$ |

Always run for a **single seed** ("for speed") — a sanity check on ranking quality ($\tau$), not a statistically powered second result. At both 4 and 8 qubits, stored runs often show **gap ≈ 0 for both GAT and random** (ceiling / product-state overlap); the informative number is $\tau$, not the tied energy gap.

### 2.3 MaxCut (Combinatorial / QAOA-style) — `QASv8` only

An 8-node Erdős–Rényi graph (`p_edge=0.4`) with cost Hamiltonian

$$H_C = \sum_{(i,j)\in E}\tfrac{1}{2}(I - Z_i Z_j).$$

Search minimizes $-\langle H_C\rangle$ so the existing VQE/predictor stack applies. Metric is the **approximation ratio**

$$r = \langle H_C\rangle_{\mathrm{circuit}}\,/\,\mathrm{MaxCut}^\star$$

with $\mathrm{MaxCut}^\star$ from brute-force enumeration ($2^{n-1}$ partitions). Stored single-seed output: 7 edges, $\mathrm{MaxCut}^\star = 6$, GAT/random/pruned all reach ratio $1.000$, predictor $\tau = 0.482$.

Two MaxCut-related bug notes are documented in the notebook: (1) reusing TFIM’s single-qubit $H_{\mathrm{local}}$ warm-up collapsed every circuit onto the same computational-basis state — fixed by setting $H_{\mathrm{local}} = H$ for MaxCut; (2) subsequent metric/offset corrections for cut counting.

### 2.4 Local Cost Function and Two-Phase VQE (v2 Onward)

From v2 onward, VQE uses a **two-phase schedule**. Phase 1 minimizes a local cost for `LOCAL_COST_FRAC = 0.30` of the budget:

$$C_{\mathrm{local}}(\boldsymbol\theta) = \langle\psi(\boldsymbol\theta)|H_{\mathrm{local}}|\psi(\boldsymbol\theta)\rangle, \qquad H_{\mathrm{local}} = -\frac{1}{n}\sum_{i=0}^{n-1} Z_i$$

Phase 2 switches to $C(\boldsymbol\theta) = \langle H\rangle$. Parameters use **layerwise** init $\theta_i^{(0)}\sim\mathcal{N}(0, 0.05^2)$ rather than $\mathcal{U}(0,2\pi)$. Both phases use PennyLane `AdamOptimizer` with parameter-shift gradients.

---

## 3. Search Space

### 3.1 Free-Form Gate Slots (Most Notebooks)

A circuit is a list of $L$ slots $\{(\mathrm{gate}_l, q_l)\}_{l=1}^L$, with $L\sim\mathcal{U}\{L_{\min},\ldots,L_{\max}\}$ and gates from $\mathcal{G}=\{RX,RY,RZ,\mathrm{CNOT}\}$. CNOTs use ring neighbours $(q_l,(q_l+1)\bmod n)$.

| Notebooks | `MIN_DEPTH`–`MAX_DEPTH` |
|-----------|-------------------------|
| v1–v7, `8qubitrun` (4q) | typically 8–18 |
| `QASv8` (8q) | **16–36** |

From v4 onward, free-form samplers enforce $|\{l:\mathrm{gate}_l=\mathrm{CNOT}\}|\ge 2$ (or equivalent floors) so “winning” product-state circuits with zero CNOTs cannot dominate TFIM search.

### 3.2 Layered Hardware-Efficient Ansatz (`q5-optimized.ipynb` Only)

$K\sim\mathcal{U}\{2,\ldots,5\}$ layers; each layer is a per-qubit rotation sublayer ($g_{k,i}\in\{RX,RY,RZ,I\}$, skip prob $0.15$) times an optional full nearest-neighbour CNOT ladder (Bernoulli $0.85$). Guarantees whole-chain entanglement when a layer entangles at all — the only design that drives post-prune $\Delta E$ near zero at 4 qubits (see §6.4, §11).

---

## 4. Circuit-to-Graph Encoding

Every notebook encodes a circuit as a directed graph $\mathcal{C}=(\mathcal{V},\mathcal{E})$.

**Nodes** are gates with features

$$x_l = \bigl[\mathrm{onehot}(\mathrm{gate}_l)\ \|\ c_l,\, t_l\ \|\ \mathrm{onehot}(q_l)\ \|\ \sin\phi_l,\, \cos\phi_l\bigr]$$

so $d=10$ in v1 and $d=12$ from v2 (phase pair). At 8 qubits the one-hot wire block grows with $n$.

**Edges** follow qubit wires bidirectionally (plus self-loops): next two ops on a shared wire get $a\leftrightarrow b$.

---

## 5. The Performance Predictors

### 5.1 Graph Attention Network (GAT)

Implemented from scratch (no PyG). Additive multi-head attention:

$$e_{ij}^{(k)} = \mathrm{LeakyReLU}_{0.2}\!\Bigl(\mathbf{a}_\mathrm{src}^{(k)\top} h_j^{(k)} \;+\; \mathbf{a}_\mathrm{dst}^{(k)\top} h_i^{(k)}\Bigr)$$

$$\alpha_{ij}^{(k)} = \mathrm{softmax}_{j\in\mathcal{N}(i)}(e_{ij}^{(k)}),\qquad h_i'^{(k)} = \sum_{j\in\mathcal{N}(i)}\alpha_{ij}^{(k)} h_j^{(k)}$$

Two layers × 4 heads, mean+max pool, MLP head → standardized energy. ~20.5k params at $d=12$, `hidden=32`.

### 5.2 Graph Convolutional Network (GCN) Ablation

Fixed mean aggregation over in-neighbours (no content-dependent attention), ~3.6k params. Isolates whether attention itself helps ranking on small circuit graphs.

### 5.3 KAN (Kolmogorov-Arnold Network) Head

GAT encoder unchanged; MLP head replaced by two `KANLinear` layers (B-spline activations via Cox–de Boor recursion, grid size 5, order 3, SiLU residual). Introduced in `DQAS+KANQAS`, multi-seeded in `8qubitrun` / v7 / v8. ~37k params total (probe output in notebooks).

### 5.4 GIN Encoder (v7+, inside GIN-BO)

2-layer Graph Isomorphism Network (Xu et al. 2019), SSL-pretrained with the same NT-Xent protocol as the GAT encoder. Used as the latent map for the Bayesian baseline, not as the primary guided-search predictor.

### 5.5 Training Objective

$$\mathcal{L}(\phi) = \mathrm{MSE}(\hat{y}, y) + \lambda_{\mathrm{rank}}\cdot\frac{1}{|P|}\sum_{(a,b)\in P}\max\bigl(0,\ m - \mathrm{sign}(y_a-y_b)(\hat{y}_a-\hat{y}_b)\bigr)$$

with `RANK_PAIRS=256`, `RANK_MARGIN=0.10`, `RANK_WEIGHT=1.0`. Model selection uses validation Kendall $\tau$. From v2 onward: SSL pre-train on 2000 unlabelled circuits (NT-Xent), then freeze backbone for the first half of fine-tuning epochs.

**SSL temperature:** raised from `0.07` → `0.12` from `q5-bench` onward (SimCLR’s $0.07$ is too cold at batch 32).

### 5.6 Best-of-GAT/KAN Active Predictor (v6+)

In `8qubitrun`, `QASv7`, and `QASv8`, each seed picks the **better of GAT and KAN** test $\tau$ as the acquisition model (`[active=GAT|KAN]` in per-seed logs). This is an intentional fix so the search is not locked to a weaker head on a given seed.

---

## 6. Predictor-Guided Search and the Acquisition Function

$$s(c) = \hat{E}(c) \;+\; \lambda_\text{gate}\,|c| \;+\; \lambda_\text{cnot}\,n_\text{CNOT}(c), \qquad c^\star = \arg\min_{c\in\mathcal{P}} s(c)$$

with $|\mathcal{P}|=$ `SEARCH_POOL`$=4000$, full VQE on top `TOPK_VALIDATE=8`.

### 6.1–6.3 Penalty bugs and fixes (v1→v4)

| Version | $\lambda_\text{gate}$, $\lambda_\text{cnot}$ | Effect |
|---------|-----------------------------------------------|--------|
| v1–v2 | 0.015, 0.040 | Penalty drowned energy → 0-CNOT product states |
| v3 | 0.008, 0.012 | Entangling circuits reappear |
| v4+ free-form | often 0 / 0 + `MIN_CNOTS` floor | Energy-only + structural floor |
| v2-style configs retained in many later notebooks | 0.008, 0.012 | Including `8qubitrun`, v7, v8 |

### 6.4 Layered ansatz breakthrough (`q5-optimized`)

Free-form floors still allowed minimally connected CNOT placements; energy stuck near $\Delta E\approx 0.3$. Layered HEA redesign → post-prune $\Delta E = 0.0045\pm 0.0057$ at 4 qubits (only notebook that essentially solves TFIM).

---

## 7. Structural Pruning

Greedy single-gate removal while $E(c_{-l})-E(c)\le\mathrm{tol}$. `q5-optimized` documents the v4 bug: `tol=5\times 10^{-3}$ below VQE noise at cheap budgets → systematic silent CNOT removal. Fix: `tol=10^{-3}` + high-precision recheck (`PRUNE_RECHECK_STEPS=250`, 5 restarts).

---

## 8. ZX-Calculus Data Augmentation (v2 Onward)

| Transformation | Rule |
|----------------|------|
| Spider fusion | $R_\sigma(\alpha)R_\sigma(\beta)\to R_\sigma(\alpha+\beta)$ |
| Identity removal | $R_\sigma(\theta)\to\mathbf{1}$ if $\theta\equiv 0\pmod{2\pi}$ (tol 0.08) |
| Phase-free simplify | $R_Z(\pi/2)R_X(\theta)R_Z(-\pi/2)\to R_Y(\theta)$ (tol 0.12) |
| Scalar reduction | Inverse of phase-free |

Variants inherit parent energy labels; ZX variants go **only** into the train split (`orig_val` / `orig_test` for val/test).

---

## 9. Search Baselines Beyond Random

### 9.1 REINFORCE-DQAS (`DQAS+KANQAS`, `8qubitrun`)

Architecture logits $\boldsymbol\eta\in\mathbb{R}^{L\times|\mathcal{G}|}$, qubit logits $\boldsymbol\xi\in\mathbb{R}^{L\times n}$; Gumbel-Softmax sampling with annealed temperature; **black-box REINFORCE** (not true differentiable DQAS through the simulator):

$$\nabla J = -(R_t - b_t)\,\nabla\log\pi,\qquad R_t=-E_t,\quad b_t=0.9\,b_{t-1}+0.1\,R_t.$$

### 9.2 GIN-BO — Bayesian EI (`QASv7`, `QASv8`)

Replaces REINFORCE-DQAS (arXiv:2512.09586 style):

1. **GIN encoder** — SSL pre-trained on unlabelled circuits  
2. **MC-Dropout surrogate** — dropout 0.1, $T=20$ stochastic passes → $(\mu,\sigma)$  
3. **Expected Improvement** — $\mathrm{EI}=(\mu_{\mathrm{best}}-\mu)\Phi(z)+\sigma\,\phi(z)$; cold start then iterative VQE updates  

Demo-scale single run (v7): ~48 VQE calls (8 cold + 40 BO steps). Multi-seed GIN-BO gaps are in §11.

---

## 10. What's Real vs. What's Stated but Unverified

**`QASv8.ipynb` is the recommended notebook for latest results** (8q TFIM, noise, MaxCut, GIN-BO, multi-seed GAT/GCN/KAN). Stored multi-seed and checklist outputs are present and internally consistent with the v8 code path.

**`8qubitrun.ipynb` is fully executed** (no longer source-only). Multi-seed aggregation, sample-efficiency, Heisenberg, component summary, and checklist all have matching stored output at 4 qubits with **DQAS** still as the non-random baseline.

**`QASv7.ipynb` is fully executed** at 4 qubits with **GIN-BO** replacing DQAS; multi-seed stats match its own aggregation cells.

**`DQAS+KANQAS.ipynb` still has the stale aggregate cell** described in earlier README revisions: the headline comparison cell can print old `q5-bench`-era strings, while the real per-seed loop output is the trustworthy source for GAT/GCN $\tau$. Prefer `8qubitrun` for the same design with clean aggregation.

**`QAS-v6.ipynb` still has the GCN-primary rationale vs. measured GAT win** contradiction (checklist correctly leaves the GCN-primary item unchecked).

**Heisenberg and MaxCut remain single-seed** everywhere they appear. Heisenberg energy gaps often floor at 0.0000 for both methods; use $\tau$ as the cross-task signal. MaxCut in v8 reaches ratio 1.0 for GAT and random on the stored seed — ranking $\tau=0.482$ is the non-trivial number.

**Noise comparison in v8 is a single circuit pair.** Stored run: GAT (2 CNOTs) degraded by $+0.776$ energy units under $p=0.01$ depolarizing noise; random (4 CNOTs) degraded by $+0.502$. The notebook itself notes that this single comparison does *not* support “fewer CNOTs ⇒ always more robust” and asks for multi-circuit/multi-seed follow-up.

**Energy formatting still outruns VQE precision** in places (e.g. prune changing gap by $10^{-4}$).

---

## 11. Results by Notebook

Unless noted: $\Delta E = E_{\mathrm{found}}-E_0$ on **4-qubit TFIM** ($E_0=-4.758770$), mean ± std over seeds `{7, 42, 137, 256, 512}`. ✓ = confirmed against stored, internally consistent notebook output.

### 11.1 Main multi-seed table (4 qubits unless noted)

| Notebook | Random gap | Guided gap | + pruning | CNOTs after prune | Test τ |
|----------|------------|------------|-----------|-------------------|--------|
| v1 (single seed) ✓ | 0.655 (tied) | 0.655 | 0.655 | 0 | 0.360 |
| v2 ✓ | 0.580 ± 0.168 | 0.355 ± 0.000 | 0.355 ± 0.000 | 0 | 0.644 ± 0.079 |
| v3 ✓ | 0.580 ± 0.168 | 0.344 ± 0.026 | 0.344 ± 0.026 | 0.2 ± 0.4 | 0.660 ± 0.078 |
| v4 ✓ | 0.588 ± 0.171 | 0.311 ± 0.018 | 0.311 ± 0.018 | 1.0 ± 0.0 | 0.625 ± 0.054 |
| `q5-bench` ✓ | 0.478 ± 0.212 | 0.337 ± 0.030 | 0.337 ± 0.030 | 0.6 ± 0.9 | GAT 0.650 ± 0.081 / GCN **0.730 ± 0.086** |
| `q5-optimized` ✓ | 0.299 ± 0.344 | 0.046 ± 0.093 | **0.0045 ± 0.0057** | 6.2 ± 2.2 | 0.602 ± 0.031 |
| `QAS-v6` ✓ | 0.478 ± 0.212 | 0.336 ± 0.030 | 0.336 ± 0.030 | 0.6 ± 0.9 | GCN 0.626 ± 0.074 / GAT 0.647 ± 0.098 |
| `DQAS+KANQAS` ✓ (partial) | 0.478 ± 0.212 ⁺ | 0.337 ± 0.030 ⁺ | n/a in aggregate | n/a | GAT 0.650 / GCN 0.730 ⁺ |
| **`8qubitrun` (v6)** ✓ | **0.580 ± 0.168** | **0.327 ± 0.039** (GAT) | **0.327 ± 0.039** | **1.0 ± 1.0** | GAT **0.671 ± 0.070** / GCN 0.644 ± 0.079 / KAN **0.683 ± 0.059** |
| **`QASv7`** ✓ | **0.580 ± 0.168** | **0.327 ± 0.039** (GAT) | **0.327 ± 0.039** | **1.0 ± 1.0** | same τ trio as `8qubitrun`; baseline = GIN-BO |
| **`QASv8` (8q TFIM)** ✓ | **0.998 ± 0.196** | **0.629 ± 0.207** | **0.592 ± 0.160** | **1.2 ± 1.3** | GAT **0.660 ± 0.044** / GCN 0.593 ± 0.121 / KAN **0.734 ± 0.100** |

⁺ Reconstructed from per-seed / related-cell output where the aggregate cell is stale.

### 11.2 Baseline search methods (multi-seed energy gap)

| Notebook | Random | DQAS | GIN-BO | GAT-guided |
|----------|--------|------|--------|------------|
| `8qubitrun` ✓ | 0.580 ± 0.168 | **0.332 ± 0.032** | — | 0.327 ± 0.039 |
| `QASv7` ✓ | 0.580 ± 0.168 | — | **0.378 ± 0.136** | 0.327 ± 0.039 |
| `QASv8` (8q) ✓ | 0.998 ± 0.196 | — | **0.754 ± 0.107** | 0.629 ± 0.207 |

At 4q, GAT vs DQAS is statistically equivalent ($\Delta\approx 0.005$, ~0.13σ). GAT vs GIN-BO at 4q: $\Delta\approx 0.051$ (~0.58σ). At 8q, GAT vs GIN-BO: $\Delta\approx 0.126$ (~0.80σ). The notebooks emphasize GAT’s **sample-efficiency at low VQE budget**, not large asymptotic energy wins.

### 11.3 Circuit-level Spearman ρ (predictor-as-surrogate)

| Notebook | $n$ pairs | Spearman ρ | $p$ |
|----------|-----------|------------|-----|
| `8qubitrun`, `QASv7` ✓ | 165 | **0.827** | ≈ 0 |
| `QASv8` (8q) ✓ | 165 | **0.836** | ≈ 0 |

### 11.4 Sample-efficiency (stored summaries)

**`8qubitrun` / `QASv7` (4q)** — fraction improvement of GAT gap vs random:

| Budget | GAT gap | Rand gap | Improvement |
|--------|---------|----------|-------------|
| 1 | 0.674 | 0.965 | 30.1% |
| 4 | 0.339 | 0.580 | 41.5% |
| 8 | 0.327 | 0.580 | 43.5% |
| 12 | 0.310 | 0.571 | **45.7%** |
| 16 | 0.310 | 0.483 | 35.7% |

**`QASv8` (8q):**

| Budget | GAT gap | Rand gap | Improvement |
|--------|---------|----------|-------------|
| 1 | 1.140 | 1.402 | 18.7% |
| 4 | 0.768 | 1.279 | 39.9% |
| 6 | 0.629 | 1.251 | **49.8%** |
| 8 | 0.629 | 0.998 | 37.0% |
| 16 | 0.615 | 0.827 | 25.6% |

### 11.5 Cross-task checks (single seed)

| Notebook | Task | τ | Gap / ratio note |
|----------|------|---|------------------|
| `q5-bench` | Heis 4q | 0.707 | gap 0.000 both methods |
| `QAS-v6` | Heis 4q | **0.612** (prose once claimed 0.707) | gap 0.000 |
| `8qubitrun` | Heis 4q | 0.650 | gap 0.000 |
| `QASv7` | Heis 4q | 0.612 | gap 0.000 |
| `QASv8` | Heis 8q | 0.646 | gap 0.000 both; ceiling effect noted |
| `QASv8` | MaxCut 8q | 0.482 | approx. ratio **1.000** GAT & random |

### 11.6 Noise (`QASv8` §14b, single pair)

Depolarizing $p=0.01$ per gate (`default.mixed`):

| Circuit | CNOTs | Noiseless $E$ | Noisy $E$ | Degradation |
|---------|-------|---------------|-----------|-------------|
| GAT-discovered | 2 | −9.031 | −8.255 | +0.776 |
| Random-search | 4 | −8.459 | −7.957 | +0.502 |

Random degraded *less* on this draw — multi-circuit study still open.

### 11.7 Headline takeaways

1. **Only the layered HEA search space (`q5-optimized`) closes 4q TFIM** to ~0.1% relative error after pruning. Free-form gate slots plateau near $\Delta E\approx 0.3$–$0.4$ even with mature predictors.
2. **At 8 qubits (`QASv8`) the predictor value proposition is substantive**: Hilbert space $2^8=256$, random gap ≈ 1.0, GAT+prune ≈ 0.59, peak sample-efficiency improvement ≈ 50% at budget 6.
3. **GCN vs GAT is unstable** across seeds/configs at 4q; GCN can win (`q5-bench`) or lose (`8qubitrun`/`QASv7`). GCN remains attractive for **parameter efficiency** (~5.7× fewer params).
4. **KAN is competitive or better on τ** in unified runs (4q: 0.683 vs GAT 0.671; 8q: 0.734 vs 0.660) and is selected as active head on 2/5 (4q) or **4/5** (8q) seeds.
5. **DQAS and GIN-BO are strong baselines** but do not clearly dominate GAT on final energy at matched multi-seed settings; GAT wins on cheap ranking + low-budget sample efficiency.

---

## 12. Mathematical and Methodological Limitations

**Acquisition scalarization is still hand-tuned.** $s(c)=\hat{E}+\lambda|c|+\lambda_{\mathrm{CNOT}}n_{\mathrm{CNOT}}$ mixes energy and gate counts without a common scale; v1–v2 bugs were the predictable consequence. Pareto or $\sigma_E$-normalized penalties are not implemented.

**Free-form expressivity remains a bottleneck at 4q.** Without the layered HEA redesign, multi-method sophistication (KAN, DQAS, GIN-BO, SSL) does not break the $\Delta E\approx 0.3$ plateau.

**ZX labels can be approximate.** Tolerance-based identity removal / phase-free rewrites assign parent energies without re-VQE; label noise is unquantified.

**KAN boundary B-spline patch is conventional but untested** for partition of unity on $[-1,1]$.

**DQAS is REINFORCE, not simulator-differentiable DQAS.** High-variance score-function gradients; weaker than literature DQAS with true quantum backprop.

**GIN-BO vs GAT cost accounting is asymmetric** (v8 note): GAT/GCN/KAN share one labelled dataset per seed; GIN-BO rebuilds SSL + cold-start independently — energy-gap tables ignore that amortized cost difference.

**Heisenberg / MaxCut are single-seed; Heisenberg gaps often uninformative** (ceiling at 0). Multi-seed cross-task remains open (checklist in v8).

**Noise model is simulated depolarizing only**, one circuit pair — not multi-seed, not amplitude damping, not real hardware. Real-device validation is still unchecked in the v8 publishability list (28/30 criteria met).

**No BeH₂ / chemistry task** in any notebook.

---

## 13. Architecture Summary

**Circuit representation:** free-form $(\mathrm{gate},q)$ slots, or layered HEA (`q5-optimized` only). Depths 8–18 (4q) or 16–36 (8q).

**Graph encoding:** gates as nodes; wire-following bidirectional edges + self-loops; features include gate one-hot, control/target flags, qubit one-hot, phase $(\sin\phi,\cos\phi)$.

**Predictors / baselines:**

| Component | Formula / role | Params (typical) |
|-----------|----------------|------------------|
| GAT | additive multi-head attention + mean/max pool + MLP | ~20.5k |
| GCN | mean aggregation ablation | ~3.6k |
| KAN head | B-spline `KANLinear` ×2 on GAT pool | ~37k total |
| DQAS | Gumbel + REINFORCE on architecture logits | — |
| GIN-BO | GIN SSL + MC-Dropout + EI | — |

**Training:** $\mathcal{L}=\mathrm{MSE}+\lambda_{\mathrm{rank}}\cdot$ hinge-rank; NT-Xent SSL on 2k circuits; freeze-then-tune.

**Search:** 4k candidates → predictor score → top-8 VQE; optional best-of-GAT/KAN.

**Pruning:** greedy deletion with tolerance + (in optimized notebook) high-precision recheck.

**Augmentation:** ZX rewrites, train-split only.

---

## 14. Dependencies

```
pennylane
torch
networkx
scipy
numpy
matplotlib
joblib   # caching from q5-bench onward (most later notebooks)
```

Notebooks typically auto-install missing packages in the first cell:

```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pennylane", "torch", "networkx", "scipy"])
```

No GPU required. Rough wall-clock (CPU): minutes–~15 min for early notebooks; multi-seed free-form 4q runs are longer; **`QASv8` multi-seed at 8 qubits is the heavyweight** (per-seed dataset builds on the order of ~25 minutes in stored logs).

---

## 15. Key Configuration Knobs

| Parameter | Range across notebooks | Effect |
|-----------|------------------------|--------|
| `N_QUBITS` | **4** (v1–v7, `8qubitrun`) → **8** (`QASv8`) | Problem size; exact $E_0$ and feature dim |
| `MIN_DEPTH` / `MAX_DEPTH` | 8–18 → **16–36** at 8q | Search-space depth |
| `N_CIRCUITS` | 220 (sometimes 300) | Labelled set before ZX |
| `VQE_STEPS` / `VQE_RESTARTS` | 60/3 → 120/4 | Label quality |
| `ZX_VARIANTS` | 3 → 1 (some mid versions) | Synthetic train fraction |
| `SSL_EPOCHS` / `SSL_LR` | 60/3e-3 → 150/8e-4 | SSL schedule |
| `SSL_TEMP` | 0.07 → **0.12** | NT-Xent temperature |
| `LAMBDA_GATES` / `LAMBDA_CNOT` | 0.015/0.040 → 0.008/0.012 or 0/0 | Acquisition penalties |
| `MIN_CNOTS` | 0 → 2 | Entanglement floor (free-form) |
| `PRUNE_TOL` | 5e-3 → 1e-3 | Pruning noise safety |
| `NOISE_P` | **0.01** (`QASv8` only) | Depolarizing strength |
| `EXPERIMENT_SEEDS` | `{7,42,137,256,512}` | Multi-seed protocol |
| `BUDGET_EVALS` | `[1,2,4,6,8,12,16]` | Sample-efficiency curve |

---

## 16. Version Lineage (Short)

```
v1 baseline
 └─ v2 (SSL, ZX, multi-seed, local cost)
     └─ v3 (penalty / bug fixes)
         └─ v4 (MIN_CNOTS, energy-only acquisition, …)
              ├─ q5-bench ──► QAS-v6 (GCN primary attempt)
              │                 └─ DQAS+KANQAS (KAN + REINFORCE-DQAS)
              │                        └─ 8qubitrun (merge, executed)
              │                               └─ QASv7 (GIN-BO)
              │                                      └─ QASv8 (8q + noise + MaxCut)
              └─ q5-optimized (layered HEA + prune fix)  [parallel expressivity branch]
```

---

## 17. References

1. **Quantum Architecture Search: A Survey.** arXiv:2406.06210  
2. **He et al. (2024).** Quantum Architecture Search with Neural Predictor Based on Graph Measures. *Advanced Quantum Technologies*. doi:10.1002/qute.202400223  
3. **GSQAS: Graph Self-supervised Quantum Architecture Search.** arXiv:2303.12381  
4. **SA-DQAS: Self-attention Enhanced Differentiable QAS.** arXiv:2406.08882  
5. **QuantumDARTS: Differentiable QAS for VQAs.** PMLR / OpenReview  
6. **DQAS** — Differentiable Quantum Architecture Search. Ye et al. 2021  
7. **Li et al. (2025).** Quantum Architecture Search with Neural Predictor Based on ZX-calculus. *EPJ Quantum Technology*. doi:10.1140/epjqt/s40507-025-00410-w  
8. **QGAT: Quantum Graph Attention Network.** arXiv:2508.17630  
9. **QAS-Bench: Rethinking QAS and a Benchmark.** PMLR v202  
10. **SimCLR.** Chen et al. 2020  
11. **KANQAS** — Kolmogorov-Arnold Network for QAS. arXiv:2406.02749  
12. **Errica et al. (2020).** A Fair Comparison of Graph Neural Networks for Graph Classification. arXiv:1912.09893  
13. **McClean et al. (2018).** Barren plateaus in quantum neural network training landscapes  
14. **Cerezo et al. (2021).** Cost function dependent barren plateaus in shallow parametrized quantum circuits  
15. **Graph-based Bayesian Optimization for QAS.** arXiv:2512.09586 *(GIN-BO baseline in v7/v8)*  
16. **NA-QAS** — Noise-Aware QAS. arXiv:2601.10965  
17. **Parameter transfer for QAS.** arXiv:2508.11914  
18. **TensorRL-QAS** — tensor-network + RL QAS, scalable to 20 qubits. NeurIPS 2025  
19. **CRLQAS** — curriculum RL for noise-aware QAS. 2024  
20. **Xu et al. (2019).** How Powerful are Graph Neural Networks? (GIN). arXiv:1810.00826  
21. **Gal & Ghahramani (2016).** Dropout as a Bayesian Approximation (MC-Dropout)  
22. **Awesome-QAS.** github.com/Aqasch/awesome-QAS  

---
