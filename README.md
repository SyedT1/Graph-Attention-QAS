# Graph-Attention-QAS

A research notebook series implementing **predictor-guided Quantum Architecture Search (QAS)** with a Graph Attention Network (GAT). The goal is to find good quantum circuit architectures (ansätze) for Variational Quantum Algorithms (VQAs) without running a full variational optimization on every candidate — instead, a graph neural network is trained to rank circuits from their structure alone, and expensive simulation is reserved for the few candidates the predictor likes best.

The benchmark task throughout is the ground-state energy of the **Transverse-Field Ising Model (TFIM)**, solved via a Variational Quantum Eigensolver (VQE); one notebook adds the **Heisenberg model** as a second task.

This repository is a sequence of eight notebooks, each one a snapshot of the same pipeline at a different stage of debugging and extension. They do not form a strictly linear v1→v8 chain — two of them (`q5-bench` and `q5-optimized`) branch off the same "v4" point in different directions, and the last two (`QAS-v6.ipynb` and `DQAS+KANQAS.ipynb`) are two further, *independent* continuations of `q5-bench` that never merge back together. This README treats them as siblings rather than forcing a false linear order, and is explicit about which numbers come from code that actually ran versus code that was written but never executed.

---

## Contents of this repository

| File | Internal version label | What it adds |
|---|---|---|
| `graph_attention_QAS_executed.ipynb` | v1 (baseline) | The original end-to-end pipeline |
| `graph-attention-qas-v2 (1).ipynb` | v2 | Cleaner VQE labels, ZX-calculus augmentation, SSL pre-training, multi-seed stats |
| `qas-v3.ipynb` | v3 | Four bug fixes to v2 |
| `Graph-Attention-QAS-v4.ipynb` | v4 | Five further fixes to the search space and labels |
| `q5-bench (1).ipynb` | "v4" (parallel branch) | GCN-vs-GAT ablation, sample-efficiency curve, Heisenberg cross-task check, joblib caching |
| `q5-optimized.ipynb` | v4→v5 | Replaces the free-form gate search space with a layered hardware-efficient ansatz; fixes a pruning bug |
| `QAS-v6.ipynb` | "v5" (branch from `q5-bench`) | Promotes GCN to the primary predictor; adds a 165-point circuit-level Spearman ρ |
| `DQAS+KANQAS.ipynb` | "v6" (branch from `q5-bench`) | Adds a KAN regression head and a DQAS (Gumbel-Softmax/REINFORCE) baseline |

All eight notebooks share the same skeleton (TFIM Hamiltonian → circuit search space → graph encoding → GNN predictor → guided search → pruning → baselines → an automated "publishability checklist"), so the sections below describe the pipeline once and then go through what changed, numerically, at each step.

---

## 1. Background

A VQA prepares a parameterized state $|\psi(\boldsymbol\theta)\rangle = U(\boldsymbol\theta)\,|0\rangle^{\otimes n}$ and minimizes the energy $\langle\psi(\boldsymbol\theta)|H|\psi(\boldsymbol\theta)\rangle$. The structure of $U$ — which gates act on which qubits, in what order — is the *architecture*. A poor architecture either cannot represent the target state or makes optimization intractable (e.g. through barren plateaus). On NISQ hardware every two-qubit gate adds noise, so a good architecture should be shallow and sparse in entangling gates while remaining expressive enough to solve the problem.

Evaluating one candidate circuit normally means running VQE to convergence, which is expensive. QAS automates architecture selection; the family of methods used here replaces most of those expensive evaluations with a **performance predictor** — a model trained on a modest set of (circuit, energy) pairs that estimates an unseen circuit's quality from its structure alone, so a full VQE run is reserved for the handful of circuits the predictor ranks highest.

---

## 2. The benchmark Hamiltonians

### 2.1 Transverse-Field Ising Model (TFIM)

Used in all eight notebooks, on an open chain of $n$ qubits:

$$ H_{\text{TFIM}} = -J\sum_{i=0}^{n-2} Z_i Z_{i+1} \;-\; h\sum_{i=0}^{n-1} X_i $$

with $J = h = 1.0$ and $n = 4$ throughout (so the Hilbert space has dimension $2^4 = 16$). $H$ is built as a PennyLane `qml.Hamiltonian`, diagonalized exactly with `numpy.linalg.eigvalsh`, and its smallest eigenvalue $E_0$ is used as ground truth. At $n=4$, $E_0 = -4.758770$ in every notebook (it depends only on $J,h,n$, which never change). The "energy gap" reported everywhere is $E_{\text{found}} - E_0 \ge 0$.

### 2.2 Heisenberg model (XXX chain)

Added in `q5-bench`, `QAS-v6.ipynb`, and `DQAS+KANQAS.ipynb` as a second task to probe whether the predictor generalizes beyond TFIM:

$$ H_{\text{Heis}} = -J\sum_{i=0}^{n-2}\big(X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1}\big) $$

At $n=4, J=1$, $E_0^{\text{Heis}} = -3.000000$. This benchmark is only ever run for a **single seed** ("for speed"), so it is a sanity check, not a statistically powered second result — see Limitations.

### 2.3 Local cost function (barren-plateau mitigation)

From v2 onward, VQE optimization is "warmed up" with a cheap local cost before switching to the real energy:

$$ H_{\text{local}} = -\frac{1}{n}\sum_{i=0}^{n-1} Z_i $$

This is an average of single-qubit $Z$ expectations, used because its gradients do not vanish exponentially with system size the way a global, all-qubit observable's do (McClean et al. 2018; Cerezo et al. 2021). VQE runs `LOCAL_COST_FRAC` (30%) of its steps against $H_{\text{local}}$, then the remainder against the true Hamiltonian.

---

## 3. Search space

### 3.1 Free-form gate slots (v1 through `QAS-v6.ipynb` and `DQAS+KANQAS.ipynb`)

A circuit is a list of `(gate_type, qubit_tuple)` slots, with gate types drawn from $\{RX, RY, RZ, CNOT\}$. Rotation gates act on one randomly chosen qubit and carry one trainable angle; CNOTs act on a randomly chosen control qubit and its ring-neighbour. Circuit depth is drawn uniformly from `[MIN_DEPTH, MAX_DEPTH]` (8–18 in most notebooks). The number of trainable parameters equals the number of rotation gates.

From v4 onward, `sample_circuit` enforces `MIN_CNOTS=2`: if a randomly sampled circuit has fewer CNOTs than the floor, early non-CNOT slots are overwritten with CNOTs until the floor is met. This exists because v1–v3 routinely produced "winning" circuits with **zero** CNOTs — a circuit with no two-qubit gates is a product state and cannot represent the ZZ correlations TFIM's ground state needs, so a CNOT-free circuit is provably suboptimal for this Hamiltonian regardless of how well its single-qubit rotations are tuned.

### 3.2 Layered hardware-efficient ansatz (`q5-optimized.ipynb` only)

This notebook redesigns the search space because the `MIN_CNOTS` floor turned out to be an incomplete fix (see §6.2). Each circuit is now a sequence of layers, each consisting of:

1. A **rotation sublayer** — every qubit independently receives `RX`, `RY`, `RZ`, or no gate at all (skip probability `ROT_SKIP_PROB=0.15`).
2. An **entangling sublayer** — with probability `ENTANGLE_PROB=0.85`, a *fixed* nearest-neighbor CNOT ladder spanning the whole chain (`0–1, 1–2, 2–3`) is appended in full or not at all.

The number of layers is drawn from `[MIN_LAYERS, MAX_LAYERS] = [2, 5]`. QAS now searches over layer count and per-qubit rotation choice; the entangling structure itself is fixed once it is decided to include it. This guarantees that whenever a layer entangles, the entanglement spans the *entire* chain, rather than relying on a CNOT-count floor that could be satisfied with disconnected, redundant, or otherwise weakly-entangling gate placements. Downstream code (graph encoding, VQE, ZX augmentation, pruning) is unchanged — every circuit is still flattened to the same `(gate, qubits)` slot list.

---

## 4. Circuit-to-graph encoding

Every notebook encodes a circuit as a directed graph for the GNN predictor:

- **Nodes** are gates. Node features are a one-hot gate-type vector, two binary control/target flags (set together whenever the gate is a CNOT), a one-hot qubit-index vector, and — from v2 onward — a phase encoding $(\sin\phi, \cos\phi)$ used when ZX-augmented circuits carry an explicit rotation angle. Feature dimension is `len(GATE_TYPES) + 2 + N_QUBITS` (= 10 in v1) or `+2` more for the phase pair (= 12 from v2 onward).
- **Edges** follow the qubit wires: if gate $a$ and then gate $b$ are the next two operations touching a shared qubit, both $a\to b$ and $b\to a$ are added (so attention can pass information in either temporal direction), plus a self-loop on every node.

---

## 5. The performance predictors

### 5.1 Graph Attention Network (GAT) — primary predictor in v1–v4, `q5-optimized`, and an explicit ablation baseline in `q5-bench`, `QAS-v6.ipynb`, `DQAS+KANQAS.ipynb`

Implemented from scratch (no PyTorch Geometric). For a directed edge $j \to i$, a single attention head computes

$$ e_{ij} = \text{LeakyReLU}\big(\mathbf{a}_{\text{src}}^\top \mathbf{W}h_j + \mathbf{a}_{\text{dst}}^\top \mathbf{W}h_i\big) $$

normalizes it with an edge-softmax over all of $i$'s incoming edges to get the attention coefficient $\alpha_{ij}$,

$$ \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})} $$

and aggregates

$$ h_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij}\,\mathbf{W}h_j $$

`HEADS=4` heads run in parallel and are concatenated in the first layer, averaged in the second. The full predictor is two such `GATLayer`s with ELU activations (`hidden=32`), followed by **mean + max** global pooling concatenated into one vector, and a small MLP regression head (`Linear → ReLU → Linear`) predicting the standardized VQE energy. Parameter count: 20,289 (v1, no phase features) or 20,545 (v2 onward, with phase features).

### 5.2 Graph Convolutional Network (GCN) ablation

Introduced in `q5-bench` as a no-attention baseline, reused in `QAS-v6.ipynb` and `DQAS+KANQAS.ipynb`. Each `GCNLayer` does unweighted mean aggregation over in-edges,

$$ h_i' = \text{ELU}\!\left(\mathbf{W}\cdot\frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} h_j\right) $$

with the same two-layer + mean/max-pool + MLP-head structure as the GAT predictor, but only 3,585 parameters (no per-head attention projections). The intent of including it was to isolate whether *attention specifically* — not just "any graph network" — is responsible for the predictor's ranking quality.

### 5.3 KAN (Kolmogorov-Arnold Network) head

Introduced in `DQAS+KANQAS.ipynb` as a drop-in replacement for the MLP regression head, with the GAT encoder left unchanged. Each `KANLinear` layer replaces fixed node activations with learnable B-spline activations applied to the *edges* (weights) of the layer: for an input $x$, the layer evaluates a degree-3 B-spline basis over a fixed grid of 5 knots on $[-1,1]$ (extended by the spline order at each end) and combines it with a residual SiLU term,

$$ \text{KANLinear}(x) = \sum_{n} c_{n}\, B_n(x) \;+\; \mathbf{W}_{\text{res}}\cdot\text{SiLU}(x) $$

where $B_n$ are the B-spline basis functions (computed via the Cox–de Boor recursion) and $c_n$ are learned spline coefficients (`spline_weight`). Two `KANLinear` layers (pool-dim → hidden → 1) replace the two-layer MLP head.

### 5.4 Training objective (all predictors, identical formula everywhere)

$$ \mathcal{L} = \text{MSE}(\hat E, E) \;+\; \lambda_{\text{rank}}\cdot\frac{1}{|P|}\sum_{(a,b)\in P} \max\!\Big(0,\ m - \text{sign}(E_a - E_b)\cdot(\hat E_a - \hat E_b)\Big) $$

The second term is a pairwise ranking hinge loss over `RANK_PAIRS=256` randomly sampled circuit pairs per epoch, margin $m=$ `RANK_MARGIN=0.10`, weight $\lambda_{\text{rank}}=$ `RANK_WEIGHT=1.0`. It rewards getting the *order* of two circuits right, independent of getting their absolute energy right, which matters because the downstream search only needs a ranking. Model selection uses the validation-set Kendall's τ (best-τ checkpoint is restored before final evaluation), and from v2 onward the GNN backbone is frozen for the first half of fine-tuning epochs and unfrozen for the second half, after being initialized from self-supervised pre-training.

### 5.5 Self-supervised pre-training (v2 onward)

A SimCLR-style contrastive objective pre-trains the encoder on 2,000 *unlabelled* circuits (no VQE simulation needed) before any energy labels are used. Two augmented views of each circuit are created by randomly dropping gates (`SSL_AUG_DROP=0.15`) and permuting qubit labels; their pooled embeddings $z_i, z_j$ are pulled together and pushed apart from all other circuits in the batch with the NT-Xent loss,

$$ \mathcal{L}_{\text{NT-Xent}} = -\log\frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k\neq i}\exp(\text{sim}(z_i,z_k)/\tau)} $$

where $\text{sim}(\cdot,\cdot)$ is cosine similarity and $\tau$ is the temperature (`SSL_TEMP`). The temperature was raised from 0.07 to 0.12 starting in `q5-bench`, because SimCLR's original 0.07 was tuned for batch sizes of 256 (255 negatives); at this project's batch size of 32 (31 negatives), 0.07 made the loss too "cold" and training plateaued early — this is a legitimate, well-reasoned fix, not a guess.

---

## 6. Predictor-guided search and the acquisition function

The search loop is the same in every notebook: sample a large pool of candidates (`SEARCH_POOL=4000`), score every one with a single cheap forward pass through the trained predictor (no quantum simulation), rank by an acquisition score, and run real VQE only on the top `TOPK_VALIDATE=8` finalists. The acquisition function is

$$ s(c) = \hat E(c) \;+\; \lambda_{\text{gate}}\,|\text{gates}(c)| \;+\; \lambda_{\text{cnot}}\,|\text{CNOTs}(c)| $$

where $\hat E(c)$ is the predictor's (de-standardized) energy estimate. The two penalty terms are meant to push the search toward shallow, entangler-sparse circuits — the actual "efficient architecture" objective — but their weights were a major, repeated source of bugs (below).

### 6.1 Bug: the efficiency penalty drowned the energy signal (v1, v2)

In v1 and v2, `LAMBDA_GATES=0.015` and `LAMBDA_CNOT=0.040`. Because a CNOT costs 0.040 in the acquisition score while the actual energy differences between candidate circuits are typically a few tenths, the penalty term dominated the energy term for any circuit containing CNOTs. The result, confirmed by the executed output of both notebooks: **every top-8 finalist in v1 had exactly 0 CNOTs**, and the v2 multi-seed run reported a CNOT standard deviation of exactly `0.000` across all 5 seeds — the search had degenerated into "find the best product state," which cannot represent TFIM's entangled ground state no matter how well-tuned. v1's best circuit (`depth=8, params=8, cnots=0`) reached an energy gap of 0.655 (13.8% relative error) and was, by chance, numerically *tied* with random search at the same budget.

### 6.2 Fix #1: reduce the penalty weights (v3)

v3 reduces `LAMBDA_GATES` to 0.008 and `LAMBDA_CNOT` to 0.012. This let the search explore CNOT-containing circuits again: the v3 multi-seed mean energy gap improved to 0.344 ± 0.026 (vs. 0.580 ± 0.168 for random search), with the GAT-guided circuits now averaging 0.6 ± 0.5 CNOTs before pruning.

### 6.3 Fix #2: force a CNOT floor, then drop the penalty entirely (v4)

v4 goes further: it sets `LAMBDA_GATES = LAMBDA_CNOT = 0.0` (an **energy-only** acquisition function), enforces `MIN_CNOTS=2` directly in the sampler (§3.1), and adds a hard post-hoc filter `MAX_DEPTH_FILTER=16` that discards any sampled circuit deeper than 16 gates before scoring. This combination reached a mean gap of 0.311 ± 0.018, with GAT-guided circuits averaging 2.4 ± 0.9 CNOTs (1.0 ± 0.0 after pruning) — the first version where pruned circuits reliably keep at least one CNOT across every seed.

### 6.4 Why the v4 fix was still incomplete, and the v5 redesign (`q5-optimized.ipynb`)

`q5-optimized.ipynb` reports, plainly, that even with `MIN_CNOTS=2` every seed converged to circuits with *exactly* 2 CNOTs on a 4-qubit ring — just enough to clear the floor, not enough to connect the whole chain (2 CNOTs span at most 3 of the 4 qubits in a line). The energy gap stayed pinned at roughly 0.31 regardless of how much extra VQE budget was thrown at it, because the bottleneck was the **ansatz family's expressivity**, not its optimization: no amount of additional gradient descent can make a circuit represent a state outside the circuit's reachable manifold. This is the rationale behind the layered hardware-efficient redesign in §3.2, which guarantees whole-chain entanglement whenever a layer entangles at all. With that change, the mean energy gap dropped to 0.046 ± 0.093 before pruning and **0.0045 ± 0.0057 after pruning** — i.e., after pruning, the discovered circuits sit within about 0.1% of the exact ground-state energy on average, at the cost of much deeper raw circuits (14.0 ± 5.2 gates, 6.2 ± 2.2 CNOTs after pruning, versus single digits in earlier versions).

---

## 7. Structural pruning

A greedy, model-agnostic pass: try removing each gate one at a time, accept the removal if the re-evaluated energy doesn't rise by more than a tolerance `tol`, repeat until no gate can be dropped.

### 7.1 Bug: pruning silently discarded a CNOT every time (`q5-optimized.ipynb`'s account of v4)

`q5-optimized.ipynb` documents a bug it inherited from v4: the pruning tolerance (`tol=5e-3`) was *smaller than the VQE noise floor* at the cheap step budget used during pruning (120 steps, 3 restarts), so a genuinely necessary CNOT could appear "safe to remove" purely because the noisy re-evaluation happened to land within tolerance — not because the circuit was actually equally good without it. This is why v4's reported pruned-circuit CNOT counts (1.0 ± 0.0) are one less than its pre-pruning counts (2.4 ± 0.9) in every single seed, which is a suspicious, too-consistent pattern for a process that's supposed to depend on circuit-specific redundancy.

### 7.2 Fix (`q5-optimized.ipynb`)

The tolerance is tightened to `PRUNE_TOL=1e-3`, and — more importantly — both the pre-removal base energy and the post-removal candidate energy are **re-validated at high precision** (`PRUNE_RECHECK_STEPS=250`, `PRUNE_RECHECK_RESTARTS=5`) before a removal is accepted, using the cheap evaluation only as a fast pre-screen. This stops optimization noise at the cheap step budget from masquerading as genuine redundancy.

---

## 8. ZX-calculus data augmentation (v2 onward)

To multiply the labelled training set without any additional quantum simulation, four equivalence-preserving rewrites are applied to sampled circuits:

| Transformation | Effect |
|---|---|
| Spider fusion | Two adjacent same-axis rotations on the same qubit merge: $R_X(\alpha)\cdot R_X(\beta) \to R_X(\alpha+\beta)$ |
| Identity removal | A rotation with phase $\approx 0 \pmod{2\pi}$ is deleted |
| Phase-free simplification | $R_Z(\pi/2)\cdot R_X(\theta)\cdot R_Z(-\pi/2) \to R_Y(\theta)$ |
| Scalar reduction | The inverse substitution: replace a chosen $R_Y(\theta)$ with $R_Z(\pi/2)\cdot R_X(\theta)\cdot R_Z(-\pi/2)$ |

Each labelled circuit generates `ZX_VARIANTS` variants (3 in v2/v3, reduced to 1 from v4 onward because v3's training set was 75% synthetic — see Limitations) that **inherit the original circuit's energy label** at zero extra VQE cost. Validation and test splits are always drawn from the *original*, pre-augmentation circuit indices only (`orig_val`, `orig_test`) — ZX variants are added exclusively to the training split — which is a deliberate anti-leakage design documented explicitly in `q5-bench`'s code comments.

---

## 9. What's real vs. what's stated but unverified

Because this is a chain of research notebooks rather than a single audited pipeline, a careful read of the raw `.ipynb` files (not just the markdown prose) turns up real discrepancies worth knowing about before citing any of these numbers.

**`DQAS+KANQAS.ipynb`'s headline multi-seed comparison cell prints stale output that does not match its own code.** The cell that aggregates GAT/GCN/KAN τ and the DQAS energy gap across 5 seeds (and computes the 165-point Spearman ρ) has stored output that is verbatim the *old* `q5-bench` printout (it even prints the string `"[v4] ablation"` and a 5-point Pearson-r — concepts this version of the code has already replaced with Spearman ρ and a circuit-level comparison). This is the unmistakable signature of a notebook cell whose source was edited after it was last actually run end-to-end: the code computes one thing, the displayed output is left over from a previous version of that same cell. Reconstructing the true numbers from the one piece of output in this notebook that *is* internally consistent — the per-seed printout from the multi-seed loop itself — gives GAT τ = 0.650 ± 0.081 and GCN τ = 0.730 ± 0.086 across the 5 experiment seeds, matching `q5-bench`'s numbers exactly (expected, since both files reuse the same seeds and config up to this point). The KAN τ, DQAS energy gap, and Spearman ρ values that the current code computes are not visible in any stored output anywhere in the file.

**Several of `DQAS+KANQAS.ipynb`'s own demonstration cells never executed at all.** Inspecting the notebook JSON directly (not just reading the rendered text) shows that the standalone `KANLinear`/`KANPredictor` class-definition cell, the standalone `DQASSupercircuit`/`run_dqas` definition-and-demo cell, the single-run KAN-vs-MLP fine-tuning comparison cell, the final "GAT vs KAN vs DQAS vs Random" summary table, and the final publishability checklist all have zero stored outputs. The classes are defined and presumably used by the multi-seed loop (`run_one_seed`, which *does* have real output and does call `run_dqas()` and instantiate `KANPredictor` internally — see below), but the notebook's own narrative demonstrations of those components in isolation were never run, so anyone reading top-to-bottom for "does the KAN head work" or "does DQAS converge" will find code and a written rationale but no confirming numbers in this file.

**The real multi-seed numbers, where they exist, come from a cell most readers would not expect to contain them.** The function `run_one_seed` (used by the multi-seed loop) does internally build a GAT predictor, a GCN ablation, a KAN predictor with an independently SSL-pretrained backbone, and a DQAS run via `run_dqas(n_steps=100, ...)`, and this cell's execution **is** captured in the file with five real per-seed print lines (one per seed in `EXPERIMENT_SEEDS`). From those five lines: GAT τ ranged 0.542–0.752 (mean ≈0.650), GCN τ ranged 0.599–0.822 (mean ≈0.730), the GAT-guided energy gap ranged 0.287–0.355 (mean ≈0.337), and CNOT counts after the search ranged 0–2. The aggregated KAN τ and DQAS gap values that this same cell computes per seed are not printed in the verbose per-seed line and so cannot be reconstructed from the stored output — they exist somewhere in the run that produced this output, but are not visible in the file.

**The Heisenberg cross-task check is a single seed with a borderline-meaningless result.** In both `q5-bench` and `QAS-v6.ipynb`, the printed Heisenberg gap is exactly `0.0000` for *both* the GAT-guided circuit and the random-search baseline. Read naively this looks like "both methods solved the problem exactly," but a more likely explanation is that the discovered circuits (depth 8, pruned to depth 1) happened to express the same trivial low-energy product state that the random baseline also found, and the print formatting (`.4f`) is rounding a genuinely-near-zero-but-nonzero gap to display as zero, or the energy landscape at this depth/step budget has a degenerate minimum reachable from generic initializations. Either way, a single seed where two different methods both report exactly 0.0000 is a result that needs a second seed and a tighter print precision before it supports the "predictor generalizes across Hamiltonians" claim the notebooks draw from it.

**`QAS-v6.ipynb`'s design rationale and its own measured result disagree.** Section 8 of that notebook states, as justification for promoting GCN to the primary predictor, that "GCN (τ=0.730) beats GAT (τ=0.650)" — copying `q5-bench`'s finding. But the multi-seed loop in the *same notebook*, run with its own (different) random seeds and SSL schedule, reports GCN τ = 0.626 ± 0.074 versus GAT τ = 0.647 ± 0.098 — i.e., in this run **GAT wins**, the opposite of the stated rationale. The notebook's own publishability checklist correctly marks "`[v5] GCN is primary (beats GAT ablation in τ)`" as unmet (`[ ]`) precisely because of this contradiction, which is honest, but it does mean the central architectural decision of this notebook is not supported by its own evidence.

None of this means the underlying ideas (GAT/GCN/KAN encoders, DQAS, ZX augmentation, SSL pre-training) are wrong — the mechanics are implemented correctly wherever they do have verified output — but the per-notebook summary tables and conclusions below are annotated to distinguish numbers that come from a verified, matching execution from numbers that are stated in markdown prose without a matching, internally-consistent printed result.

---

## 10. Results by notebook

All energy gaps are $E_{\text{found}} - E_0$ on 4-qubit TFIM ($E_0=-4.758770$), mean ± standard deviation over the 5 seeds `{7, 42, 137, 256, 512}` unless noted otherwise. ✓ marks numbers directly confirmed against the notebook's own stored, internally-consistent output; ⚠ marks numbers stated in prose that could not be cross-checked against a matching execution.

| Notebook | Random gap | GAT/GCN-guided gap | + pruning | CNOTs after pruning | Test τ |
|---|---|---|---|---|---|
| v1 (baseline, single seed) ✓ | 0.655 (tied) | 0.655 | 0.655 | 0 | 0.360 |
| v2 ✓ | 0.580 ± 0.168 | 0.355 ± 0.000 | 0.355 ± 0.000 | 0 | 0.644 ± 0.079 |
| v3 ✓ | 0.580 ± 0.168 | 0.344 ± 0.026 | 0.344 ± 0.026 | 0.2 ± 0.4 | 0.660 ± 0.078 |
| v4 ✓ | 0.588 ± 0.171 | 0.311 ± 0.018 | 0.311 ± 0.018 | 1.0 ± 0.0 | 0.625 ± 0.054 |
| `q5-bench` ✓ | 0.478 ± 0.212 | 0.337 ± 0.030 | 0.337 ± 0.030 | 0.6 ± 0.9 | GAT 0.650 ± 0.081 / GCN 0.730 ± 0.086 |
| `q5-optimized` ✓ | 0.299 ± 0.344 | 0.046 ± 0.093 | **0.0045 ± 0.0057** | 6.2 ± 2.2 | 0.602 ± 0.031 |
| `QAS-v6.ipynb` ✓ | 0.478 ± 0.212 | 0.336 ± 0.030 | 0.336 ± 0.030 | 0.6 ± 0.9 | GCN 0.626 ± 0.074 / GAT 0.647 ± 0.098 |
| `DQAS+KANQAS.ipynb` ✓ (partial) | 0.478 ± 0.212⁺ | 0.337 ± 0.030⁺ | n/a in stored output | n/a in stored output | GAT 0.650 ± 0.081 / GCN 0.730 ± 0.086⁺ |

⁺ Reconstructed from `DQAS+KANQAS.ipynb`'s stale/mismatched output cell (see §9); not independently re-verified for this file. KAN τ and DQAS gap are not present in any stored output in this file.

Two results stand out. First, **only the v5 layered-ansatz redesign (`q5-optimized.ipynb`) actually closes the gap to the exact ground state** — every free-form-gate-slot version plateaus around a 0.3–0.4 energy gap (roughly 6–8% relative error) no matter how the acquisition function, penalty weights, or VQE budget are tuned, because the bottleneck there is the search space's expressivity, not its optimization. Second, **the GCN-vs-GAT finding is itself unstable across reruns** — `q5-bench` and `DQAS+KANQAS.ipynb` (same seeds, closely related code) both show GCN ahead by a similar margin, but `QAS-v6.ipynb`, run with what is nominally the same experimental design, shows GAT ahead instead. At $n=4$ qubits with a 220-circuit dataset, this comparison does not appear to be settled.

---

## 11. Mathematical and methodological limitations

**The TFIM benchmark is small enough to make the predictor's value proposition hard to demonstrate.** At 4 qubits the Hamiltonian is exactly diagonalizable in milliseconds, and a single VQE evaluation costs only a few seconds. The entire motivation for a learned predictor — that full evaluation is too expensive to apply to every candidate — is far more pressing at 8–20 qubits, where these notebooks never go. Every "publishability checklist" across all eight files independently flags `Scale beyond a few qubits (>= 8 qubits)` as unmet.

**The acquisition function's energy term and efficiency penalty terms are not on a common, principled scale.** $\hat E(c) + \lambda_{\text{gate}}|{\text{gates}(c)}| + \lambda_{\text{cnot}}|\text{CNOTs}(c)|$ adds an energy (in Hartree-like units, here just the dimensionless TFIM energy scale) to integer gate counts with hand-tuned weights $\lambda$. The v1→v2 bug (§6.1) was a direct, predictable consequence of this: there is no a priori reason a CNOT should cost exactly 0.040 energy-units rather than 0.012 or 0.0, and three different values were tried across the version history before landing on "drop the penalty and use a hard structural constraint instead" (v4) or "redesign the search space so the constraint is structural by construction" (`q5-optimized`). A more principled approach — e.g. a Pareto frontier over (energy, gate count) rather than a single scalarized score, or normalizing the penalty by the empirical energy spread of the candidate pool — is not implemented anywhere in this series.

**ZX-calculus augmentation assigns a parent circuit's energy label to a structurally different circuit without re-running VQE.** Spider fusion, identity removal, and the phase-free/scalar-reduction substitutions are valid gate-level identities for the *specific rotation angles* they're derived from, but the augmented circuit's VQE-optimal angles are not actually re-optimized — the variant simply inherits the original circuit's *converged energy* as its label. This is a reasonable approximation when the rewrite is a true identity (e.g. spider fusion is exact for any angle), but `_identity_remove`'s tolerance (`tol=0.08` radians) and `_phase_free_simplify`'s pattern-matching tolerance (`0.12` radians) both accept *approximate* matches, meaning some fraction of "augmented" circuits are not exactly equivalent to their parent and are receiving a label that doesn't correspond to their own true converged energy. None of the notebooks quantify how much label noise this introduces.

**The KAN B-spline implementation has a boundary-condition patch whose correctness is asserted, not tested.** The de Boor recursion's order-0 basis uses a half-open interval test (`x >= grid[i] & x < grid[i+1]`), which by construction assigns zero basis weight to any input landing exactly on the rightmost grid point; the code adds a manual patch line to push that mass into the last bin. This is a known, standard edge case in B-spline implementations, and the fix is a reasonable one, but no unit test verifies the basis sums to 1 across the input domain, so this is a plausible-looking fix rather than a verified one.

**DQAS as implemented is a REINFORCE policy-gradient variant, not differentiable architecture search.** The notebook's own `DQASSupercircuit.forward()` docstring states this directly: true DQAS backpropagates through the quantum circuit itself (via a differentiable simulator or the parameter-shift rule), which the codebase's joblib-cached, non-differentiable `evaluate_circuit` cannot support. The implementation instead treats VQE energy as a black-box reward and updates architecture logits with a REINFORCE estimator and an EMA baseline. This is a legitimate, commonly used simplification, but it means the "DQAS baseline" in this repository is weaker than the gradient-based method the name usually refers to in the literature, and any energy-gap comparison between "GAT-guided" and "DQAS" is really a comparison against this lighter-weight variant.

**The Heisenberg cross-task result is statistically unpowered by construction.** It is run for exactly one seed "for speed," with no error bars, no significance test, and (per §9) a reported gap of exactly 0.0000 for both methods that is more likely a degenerate-minimum or rounding artifact than evidence of generalization.

**Energy-gap "ties" between methods are reported as if they were meaningfully different.** Several notebooks report e.g. `GAT-guided: 0.3551` and `GAT-guided + prune: 0.3552` to four decimal places and discuss them as distinct results, when the difference (0.0001) is far smaller than the VQE optimization noise floor at the step budgets used (a few×10⁻³ at best, per `q5-optimized`'s own pruning-bug discussion in §7.1). The high-precision formatting throughout the codebase outpaces the actual numerical precision of the underlying VQE estimates.

**No noise model or real-hardware validation exists anywhere in the series.** The entire "efficiency" motivation — fewer CNOTs and shallower circuits matter because two-qubit gates are noisy on real devices — is argued from first principles but never demonstrated: every VQE evaluation in all eight notebooks uses PennyLane's noiseless `default.qubit` simulator. A circuit with fewer CNOTs is only confirmed cheaper in gate count, not in any measured or simulated robustness to noise.

**Baselines stop at random search**, except for the REINFORCE-based DQAS variant in one notebook. None of the eight files compare against reinforcement-learning search, Bayesian optimization, or evolutionary search, despite all eight publishability checklists listing "strong baselines" as an open gap.

---

## 12. Architecture summary (for quick reference)

**Circuit representation:** list of `(gate_type, qubit_tuple)` slots; gate types $\{RX, RY, RZ, CNOT\}$ (free-form) or a layered rotation/fixed-CNOT-ladder structure (`q5-optimized` only).

**Graph encoding:** directed graph, gates as nodes (one-hot type + control/target flags + one-hot qubit + optional phase $(\sin\phi,\cos\phi)$), bidirectional wire-following edges + self-loops.

**Predictors:** GAT (2 layers × 4 heads, hidden=32, mean+max pool, MLP head, 20.3–20.5k params) · GCN ablation (2 mean-aggregation layers, same pool/head, 3.6k params) · KAN variant (same GAT encoder, B-spline head).

**Training:** MSE + pairwise ranking hinge loss; NT-Xent self-supervised pre-training (v2 onward) on 2,000 unlabelled circuits; backbone frozen for the first half of fine-tuning epochs.

**Search:** sample 4,000 candidates → score with one predictor forward pass each → rank by $\hat E + \lambda_{\text{gate}}|{\text{gates}}| + \lambda_{\text{cnot}}|\text{CNOTs}|$ (or energy-only from v4 onward) → validate top 8 with full VQE.

**Pruning:** greedy single-gate removal, accept if energy doesn't rise beyond a tolerance; tolerance and re-validation precision tightened in `q5-optimized` to avoid mistaking VQE noise for redundancy.

**Augmentation:** ZX-calculus rewrites (spider fusion, identity removal, phase-free simplification, scalar reduction) generating free training-label variants, restricted to the training split only.

**Baselines:** random search at a matched full-VQE-evaluation budget (all notebooks); REINFORCE/Gumbel-Softmax DQAS (`DQAS+KANQAS.ipynb` only, multi-seed numbers not recoverable from stored output — see §9).

---

## 13. Dependencies

```
pennylane
torch
networkx
scipy
numpy
matplotlib
joblib   # q5-bench, q5-optimized, QAS-v6, DQAS+KANQAS only (VQE result caching)
```

Every notebook installs missing packages automatically in its first code cell:

```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pennylane", "torch", "networkx", "scipy"])
```

No GPU is required; default settings are tuned to run in a few minutes to ~15 minutes on CPU per notebook (the multi-seed sections are the slow part — each of the 5 seeds repeats dataset generation, SSL pre-training, and fine-tuning from scratch).

---

## 14. Key configuration knobs

| Parameter | Typical range across notebooks | Effect |
|---|---|---|
| `N_QUBITS` | 4 (fixed everywhere) | Problem size; scaling this up is the single most-requested fix in every publishability checklist |
| `N_CIRCUITS` | 220 → 300 | Labelled dataset size before ZX augmentation |
| `VQE_STEPS` / `VQE_RESTARTS` | 60/3 → 120/4 | VQE label quality; v4 found 60 steps left the best labelled circuit 0.28 above its 120-step value |
| `ZX_VARIANTS` | 3 → 1 | v3's training set was 75% ZX-synthetic; reduced from v4 onward |
| `SSL_EPOCHS` / `SSL_LR` | 60/3e-3 → 150/8e-4 | SSL pre-training; v3's loss plateaued near the random-init baseline, prompting longer training at a lower rate |
| `SSL_TEMP` | 0.07 → 0.12 | NT-Xent temperature; raised because the original SimCLR value assumes far larger batches |
| `LAMBDA_GATES` / `LAMBDA_CNOT` | 0.015/0.040 → 0.0/0.0 | Acquisition penalty weights; see §6 for the full bug history |
| `MIN_CNOTS` | 0 → 2 | Entanglement floor (superseded by the layered-ansatz redesign in `q5-optimized`) |
| `PRUNE_TOL` | 5e-3 → 1e-3 | Pruning tolerance; tightened after the silent-CNOT-removal bug |

---

## 15. References

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
12. **Errica et al. (2020).** A Fair Comparison of Graph Neural Networks for Graph Classification (GCN-vs-GAT-on-small-graphs evidence). arXiv:1912.09893
13. **McClean et al. (2018).** Barren plateaus in quantum neural network training landscapes.
14. **Cerezo et al. (2021).** Cost function dependent barren plateaus in shallow parametrized quantum circuits.
15. **Graph-based Bayesian Optimization for QAS.** arXiv:2512.09586
16. **NA-QAS** — Noise-Aware QAS. arXiv:2601.10965
17. **Parameter transfer for QAS.** arXiv:2508.11914
18. **TensorRL-QAS** — tensor-network + RL QAS, scalable to 20 qubits. NeurIPS 2025
19. **CRLQAS** — curriculum RL for noise-aware QAS. 2024
20. **Awesome-QAS.** github.com/Aqasch/awesome-QAS

---

## 16. Summary

This series demonstrates, correctly and reproducibly at small scale, that a graph-based performance predictor can learn to rank quantum circuits well enough to guide a search that beats random search at a matched evaluation budget — the core hypothesis holds up across every single one of the eight notebooks. It also demonstrates, through its own version history, several instructive failure modes worth taking seriously rather than glossing over: a scalarized multi-objective acquisition function is fragile to its weight choices (it took three iterations to stop accidentally selecting unentangled circuits); a hard structural constraint can satisfy its letter while missing its spirit (the `MIN_CNOTS` floor was met while the chain stayed unconnected); a pruning tolerance can be silently violated by the very optimization noise it's supposed to be robust to; and an architecture-vs-architecture ablation (GAT vs. GCN) that looks decisive in one run can flip in the next. None of these are fatal to the underlying research direction, but a reader citing any single number from this repository should know which notebook it came from, whether that notebook's own output actually confirms it, and what changed in the next version because that number wasn't good enough yet.
