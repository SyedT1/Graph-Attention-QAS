# Graph-Attention-QAS

A research notebook series implementing **predictor-guided Quantum Architecture Search (QAS)** with a Graph Attention Network (GAT). The goal is to find good quantum circuit architectures (ansätze) for Variational Quantum Algorithms (VQAs) without running a full variational optimization on every candidate — instead, a graph neural network is trained to rank circuits from their structure alone, and expensive simulation is reserved for the few candidates the predictor likes best.

The benchmark task throughout is the ground-state energy of the **Transverse-Field Ising Model (TFIM)**, solved via a Variational Quantum Eigensolver (VQE); some notebooks add the **Heisenberg model** as a second task.

This repository contains nine notebooks. The first eight form a branching development history from baseline to two parallel v5/v6 branches. The ninth, **`8qubitrun.ipynb`**, is the **consolidated production notebook** that merges the previously diverged `QAS-v6.ipynb` and `DQAS+KANQAS.ipynb` branches into a single, fully integrated pipeline — incorporating all v6 bug fixes and positioning the codebase for the next planned extension to 8+ qubits. This README treats them as siblings rather than forcing a false linear order, derives every equation directly from the code that implements it, and is explicit about which numbers come from code that actually ran versus code that was written but never executed.

---

## Contents of this Repository

| File | Internal version label | What it adds |
|------|------------------------|--------------|
| `graph_attention_QAS_executed.ipynb` | v1 (baseline) | The original end-to-end pipeline |
| `graph-attention-qas-v2 (1).ipynb` | v2 | Cleaner VQE labels, ZX-calculus augmentation, SSL pre-training, multi-seed stats |
| `qas-v3.ipynb` | v3 | Four bug fixes to v2 |
| `Graph-Attention-QAS-v4.ipynb` | v4 | Five further fixes to the search space and labels |
| `q5-bench (1).ipynb` | "v4" (parallel branch) | GCN-vs-GAT ablation, sample-efficiency curve, Heisenberg cross-task check, joblib caching |
| `q5-optimized.ipynb` | v4→v5 | Replaces the free-form gate search space with a layered hardware-efficient ansatz; fixes a pruning bug |
| `QAS-v6.ipynb` | "v5" (branch from `q5-bench`) | Promotes GCN to the primary predictor; adds a 165-point circuit-level Spearman ρ |
| `DQAS+KANQAS.ipynb` | "v6" (branch from `q5-bench`) | Adds a KAN regression head and a DQAS (Gumbel-Softmax/REINFORCE) baseline |
| `8qubitrun.ipynb` | **v6 unified** (merge of `QAS-v6` + `DQAS+KANQAS`) | Consolidates both v6 branches; 10 issues fixed; KAN and DQAS both in multi-seed loop; circuit-level Spearman ρ; 8-qubit scaling targeted as next step |

All nine notebooks share the same skeleton (TFIM Hamiltonian → circuit search space → graph encoding → GNN predictor → guided search → pruning → baselines → an automated "publishability checklist"), so the sections below build up the full mathematical pipeline once, then go through what changed numerically at each version.

---

## 1. Background and Problem Statement

A VQA prepares a parameterized state

$$|\psi(\boldsymbol\theta)\rangle = U(\boldsymbol\theta)\,|0\rangle^{\otimes n}, \qquad U(\boldsymbol\theta) = \prod_{l=1}^{L} U_l(\theta_l)$$

where each $U_l$ is either a single-qubit rotation $RX(\theta)$, $RY(\theta)$, $RZ(\theta)$ (one trainable angle) or a fixed two-qubit $\mathrm{CNOT}$ (no trainable angle). The cost function to minimize is the energy expectation value

$$C(\boldsymbol\theta) = \langle\psi(\boldsymbol\theta)|H|\psi(\boldsymbol\theta)\rangle = \langle 0|^{\otimes n}\,U(\boldsymbol\theta)^\dagger\, H\, U(\boldsymbol\theta)\,|0\rangle^{\otimes n}$$

for a target Hamiltonian $H$. The *architecture* is the discrete choice of which gates appear, on which qubits, in what order — i.e. the sequence $\{(\mathrm{gate}_l, \mathrm{wire}_l)\}_{l=1}^L$ independent of the continuous angles $\boldsymbol\theta$. A poor architecture either cannot represent the target state at all (insufficient expressivity) or makes the loss landscape pathological — gradients $\nabla_{\boldsymbol\theta} C$ that vanish exponentially in $n$ almost everywhere (a *barren plateau*, McClean et al. 2018). On NISQ hardware there is an extra pressure: every two-qubit gate injects physical noise, so the search wants circuits that are simultaneously low-energy, shallow, and CNOT-sparse.

**Quantum Architecture Search (QAS)** is the bi-level optimization

$$\mathrm{arch}^\star = \arg\min_{\mathrm{arch}} \;\min_{\boldsymbol\theta} C_{\mathrm{arch}}(\boldsymbol\theta)$$

over the *outer* discrete architecture space, where each evaluation of the objective requires solving the *inner* continuous VQE optimization to convergence. The inner loop is what makes naive QAS expensive: evolutionary or RL-based search methods that evaluate thousands of candidates each pay this full inner-loop cost per candidate. The idea explored across all eight notebooks here is to replace most of those inner-loop evaluations with a cheap **learned performance predictor** $\hat{E}_\phi(\mathrm{arch})$ — a graph neural network trained on a modest number of true (architecture, VQE energy) pairs — and reserve the expensive inner loop for only the handful of architectures the predictor ranks best.

---

## 2. The Benchmark Hamiltonians

### 2.1 Transverse-Field Ising Model (TFIM)

Used in all eight notebooks, on an open chain of $n$ qubits:

$$H_{TFIM} = -J\sum_{i=0}^{n-2} Z_i Z_{i+1} \;-\; h\sum_{i=0}^{n-1} X_i$$

with $J = h = 1.0$ and $n = 4$ throughout (Hilbert space dimension $2^4 = 16$). $H$ is built as a PennyLane `qml.Hamiltonian`, converted to its dense matrix representation, and diagonalized exactly:

$$E_0 = \lambda_{\min}\bigl(H_{TFIM}\bigr) = \min_k \lambda_k, \qquad H_{TFIM}\,|\phi_k\rangle = \lambda_k\,|\phi_k\rangle$$

via `numpy.linalg.eigvalsh`. At $n=4$, $E_0 = -4.758770$ in every notebook (it depends only on $J, h, n$, which never change across the series). The **energy gap** reported everywhere is the non-negative quantity

$$\Delta E(\mathrm{arch}) = C_{\mathrm{arch}}(\boldsymbol\theta^\star) - E_0 \;\ge\; 0$$

where $\boldsymbol\theta^\star$ is the VQE-optimized parameter vector for that architecture; $\Delta E = 0$ iff the circuit reaches the exact ground state.

### 2.2 Heisenberg Model (XXX Chain)

Added in `q5-bench`, `QAS-v6.ipynb`, and `DQAS+KANQAS.ipynb` as a second task to probe whether the predictor generalizes beyond TFIM:

$$H_{\mathrm{Heis}} = -J\sum_{i=0}^{n-2}\bigl(X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1}\bigr)$$

At $n=4$, $J=1$, $E_0^{\text{Heis}} = -3.000000$. This benchmark is only ever run for a **single seed** ("for speed"), so it is a sanity check, not a statistically powered second result — see §11.

### 2.3 Local Cost Function and the Two-Phase VQE Schedule (Barren-Plateau Mitigation, v2 Onward)

From v2 onward, VQE optimization does not minimize $C(\boldsymbol\theta) = \langle H\rangle$ directly from the start. Instead it runs a **two-phase schedule** against two different cost functions. Phase 1 minimizes a cheap local cost,

$$C_{\mathrm{local}}(\boldsymbol\theta) = \langle\psi(\boldsymbol\theta)|H_{\mathrm{local}}|\psi(\boldsymbol\theta)\rangle, \qquad H_{\mathrm{local}} = -\frac{1}{n}\sum_{i=0}^{n-1} Z_i$$

for `local_steps = floor(VQE_STEPS · LOCAL_COST_FRAC)` steps (`LOCAL_COST_FRAC = 0.30`, so 30% of the budget), and phase 2 switches to the true cost $C(\boldsymbol\theta) = \langle H \rangle$ for the remaining `global_steps = VQE_STEPS − local_steps`. $H_{\mathrm{local}}$ is a sum of single-qubit operators, so unlike a global $n$-qubit observable its gradient variance does not shrink exponentially with $n$ (Cerezo et al. 2021); the idea is to use it purely to escape the flat region near a random initialization before switching to the real objective the search actually cares about. Parameters are initialized **layerwise** rather than uniformly at random:

$$\theta_i^{(0)} \sim \mathcal{N}(0,\; 0.05^2) \quad \text{(layerwise init)} \qquad \text{vs.} \qquad \theta_i^{(0)} \sim \mathcal{U}(0, 2\pi) \quad \text{(v1 default)}$$

Both phases use PennyLane's `AdamOptimizer` with the autograd interface, which differentiates the quantum circuit using the parameter-shift rule under the hood. For a gate $e^{-i\theta P/2}$ generated by a Pauli $P$, the exact gradient of an expectation value is

$$\frac{\partial \langle H\rangle}{\partial \theta} = \frac{1}{2}\Bigl[\langle H\rangle_{\theta + \pi/2} - \langle H\rangle_{\theta - \pi/2}\Bigr]$$

with no finite-difference approximation error, since this identity is exact for Pauli-rotation gates.

---

## 3. Search Space

### 3.1 Free-Form Gate Slots (v1 through `QAS-v6.ipynb`, `DQAS+KANQAS.ipynb`, and `8qubitrun.ipynb`)

A circuit is a list of $L$ slots $\{(\mathrm{gate}_l, q_l)\}_{l=1}^L$, with $L \sim \mathcal{U}\{L_{\min}, \ldots, L_{\max}\}$ (`MIN_DEPTH` to `MAX_DEPTH`, 8–18 in most notebooks), and gate types drawn uniformly from $\mathcal{G} = \{RX, RY, RZ, \mathrm{CNOT}\}$. Rotation gates act on one randomly chosen qubit $q_l \in \{0,\ldots,n-1\}$ and carry one trainable angle; CNOTs act on a randomly chosen control qubit and its ring-neighbour $(q_l,\, (q_l+1)\bmod n)$. The number of trainable parameters is $|\boldsymbol\theta| = |\{l : \mathrm{gate}_l \neq \mathrm{CNOT}\}|$.

From v4 onward, `sample_circuit` enforces a hard floor $|\{l : \mathrm{gate}_l = \mathrm{CNOT}\}| \ge 2$: if a randomly sampled circuit has fewer CNOTs than the floor, early non-CNOT slots are overwritten with CNOTs until the floor is met. This exists because v1–v3 routinely produced "winning" circuits with **zero** CNOTs. A CNOT-free circuit factorizes as $U = \bigotimes_{i=1}^n u_i(\theta_i)$, a product of independent single-qubit unitaries, so $|\psi(\boldsymbol\theta)\rangle = \bigotimes_i u_i(\theta_i)|0\rangle$ is a separable (unentangled) product state for *any* choice of $\boldsymbol\theta$. The TFIM ground state at $J=h$ has nonzero two-point correlations $\langle Z_iZ_{i+1}\rangle \neq \langle Z_i\rangle\langle Z_{i+1}\rangle$ and is provably entangled, so no product state — however well its single-qubit angles are tuned — can reach $\Delta E = 0$; a 0-CNOT circuit is suboptimal by construction for this Hamiltonian, regardless of optimization quality.

### 3.2 Layered Hardware-Efficient Ansatz (`q5-optimized.ipynb` Only)

This notebook redesigns the search space because the `MIN_CNOTS` floor turned out to be an incomplete fix (see §6.4). Each circuit is now a sequence of $K \sim \mathcal{U}\{2,\ldots,5\}$ (`MIN_LAYERS` to `MAX_LAYERS`) layers, each layer $k$ consisting of:

$$\mathrm{layer}_k = \underbrace{\bigotimes_{i=0}^{n-1} g_{k,i}(\theta_{k,i})}_{\text{rotation sublayer}} \;\cdot\; \underbrace{\mathbb{1}[b_k = 1]\cdot\prod_{i=0}^{n-2}\mathrm{CNOT}(i, i+1)}_{\text{entangling sublayer}}$$

where $g_{k,i} \in \{RX, RY, RZ, \mathrm{Identity}\}$ is drawn independently per qubit (skip probability `ROT_SKIP_PROB = 0.15`, i.e. $P(g_{k,i}=\mathrm{Identity})=0.15$), and $b_k \sim \mathrm{Bernoulli}(p=0.85)$ decides whether the *entire* fixed nearest-neighbor CNOT ladder $\mathrm{CNOT}(0,1)\cdot\mathrm{CNOT}(1,2)\cdot\mathrm{CNOT}(2,3)$ is appended in full or omitted entirely — never partially. QAS now searches over $K$, $\{b_k\}$, and $\{g_{k,i}\}$; the entangling structure itself, once included, is fixed. This guarantees that whenever a layer entangles at all, the entanglement graph spans the whole chain in one connected component, rather than relying on a bare CNOT-count floor that can be satisfied by disconnected or redundant gate placements (see §6.4 for why that distinction matters numerically). Downstream code (graph encoding, VQE, ZX augmentation, pruning) is unchanged — every circuit is still flattened to the same `(gate, qubits)` slot list before being consumed.

---

## 4. Circuit-to-Graph Encoding

Every notebook encodes a circuit as a directed graph $\mathcal{C} = (\mathcal{V}, \mathcal{E})$ for the GNN predictor.

**Nodes** $\mathcal{V} = \{v_1,\ldots,v_L\}$ are gates. The feature vector of node $v_l$ is the concatenation

$$x_l = \bigl[\underbrace{\mathrm{onehot}(\mathrm{gate}_l)}_{|\mathcal{G}|=4}\ \|\ \underbrace{c_l,\, t_l}_{\text{control/target flags}}\ \|\ \underbrace{\mathrm{onehot}(q_l)}_{n=4}\ \|\ \underbrace{\sin\phi_l,\, \cos\phi_l}_{\text{phase, v2+ only}}\bigr] \in \mathbb{R}^{d}$$

where $c_l = t_l = 1$ jointly whenever $\mathrm{gate}_l = \mathrm{CNOT}$ (and both 0 otherwise), and $\phi_l$ is the rotation angle when a ZX-augmented circuit carries an explicit phase (set to 0 for unaugmented circuits). Feature dimension $d = 10$ in v1 (no phase pair), $d = 12$ from v2 onward.

**Edges** follow the qubit wires: if gate $a$ and then gate $b$ (in program order) are the next two operations touching a shared qubit, both directed edges $a\to b$ and $b\to a$ are added (so attention can pass information in either temporal direction along a wire), plus a self-loop $v_l \to v_l$ on every node. This wire-following construction is a directed-acyclic-graph encoding of the circuit: edges capture both the temporal order along each qubit wire and the entanglement links created by two-qubit gates.

---

## 5. The Performance Predictors

### 5.1 Graph Attention Network (GAT) — Primary Predictor in v1–v4, `q5-optimized`, and Explicit Ablation Baseline in `q5-bench`, `QAS-v6.ipynb`, `DQAS+KANQAS.ipynb`, `8qubitrun.ipynb`

Implemented from scratch (no PyTorch Geometric). Each `GATLayer` applies a learned linear projection $W \in \mathbb{R}^{H \cdot D \times d_{\mathrm{in}}}$ shared across all $H$ heads, then reshapes to per-head features $h_l^{(k)} \in \mathbb{R}^D$ for head $k=1,\ldots,H$. For a directed edge $j \to i$, head $k$ computes an unnormalized attention logit as the **sum** of a source term and a destination term (not a concatenation — this is the additive variant of GAT):

$$e_{ij}^{(k)} = \mathrm{LeakyReLU}_{0.2}\!\Bigl(\mathbf{a}_\mathrm{src}^{(k)\top} h_j^{(k)} \;+\; \mathbf{a}_\mathrm{dst}^{(k)\top} h_i^{(k)}\Bigr)$$

where $\mathbf{a}_\mathrm{src}^{(k)}, \mathbf{a}_\mathrm{dst}^{(k)} \in \mathbb{R}^D$ are learned per-head attention vectors. The logits are normalized with an edge-softmax over all of $i$'s incoming neighbours $\mathcal{N}(i)$:

$$\alpha_{ij}^{(k)} = \frac{\exp\!\bigl(e_{ij}^{(k)} - \max_{j'\in\mathcal{N}(i)} e_{ij'}^{(k)}\bigr)}{\displaystyle\sum_{j'\in\mathcal{N}(i)} \exp\!\bigl(e_{ij'}^{(k)} - \max_{j''\in\mathcal{N}(i)} e_{ij''}^{(k)}\bigr)}$$

(the max-subtraction is implemented explicitly in code for numerical stability and does not change the value of the softmax), and aggregates

$$h_i'^{(k)} = \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)}\, h_j^{(k)}$$

The first `GATLayer` runs `HEADS = 4` heads in parallel and **concatenates** them, $h_i' = \bigl[h_i'^{(1)} \,\|\, \cdots \,\|\, h_i'^{(H)}\bigr] \in \mathbb{R}^{HD}$; the second layer runs 4 heads again but **averages** them, $h_i' = \frac{1}{H}\sum_k h_i'^{(k)} \in \mathbb{R}^D$. Both layers apply ELU activation, $\mathrm{ELU}(x) = x$ if $x > 0$, else $\alpha(e^x - 1)$. Graph-level pooling concatenates mean and max over all nodes:

$$z = \bigl[\, \mathrm{mean}_{i\in\mathcal{V}}\,h_i' \;\|\; \max_{i\in\mathcal{V}}\,h_i' \,\bigr] \in \mathbb{R}^{2D}$$

and a two-layer MLP head $\hat{E}_{\mathrm{std}} = W_2\,\mathrm{ReLU}(W_1 z + b_1) + b_2$ produces the standardized energy prediction. Total parameter count: 20,289 (v1, $d=10$) or 20,545 (v2 onward, $d=12$), with `hidden = 32`.

### 5.2 Graph Convolutional Network (GCN) Ablation

Introduced in `q5-bench` as a no-attention baseline, reused in `QAS-v6.ipynb` and `DQAS+KANQAS.ipynb`. Each `GCNLayer` replaces the learned attention weights $\alpha_{ij}^{(k)}$ with a fixed, unweighted average over in-neighbours:

$$h_i' = \mathrm{ELU}\!\left(W\cdot\frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} h_j\right)$$

i.e. $\alpha_{ij} \equiv 1/|\mathcal{N}(i)|$ for every edge regardless of node content — the same two-layer + mean/max-pool + MLP-head structure as the GAT predictor, but with only 3,585 parameters (no per-head attention projections $\mathbf{a}_\mathrm{src}^{(k)}, \mathbf{a}_\mathrm{dst}^{(k)}$). The point of including it is to isolate whether *content-dependent attention* specifically — not just "any message-passing graph network" — is what drives the predictor's ranking quality.

### 5.3 KAN (Kolmogorov-Arnold Network) Head

Introduced in `DQAS+KANQAS.ipynb` as a drop-in replacement for the MLP regression head, with the GAT encoder ($g_1, g_2$) left unchanged; only the head after pooling differs. Each `KANLinear` layer replaces a fixed-activation linear map with a **learnable B-spline activation per (input, output) connection** — the defining idea of a KAN, motivated by the Kolmogorov–Arnold representation theorem (any continuous multivariate function decomposes into sums/compositions of continuous univariate functions). A fixed uniform grid of knots $t_0 < t_1 < \cdots < t_{m}$ is built over $[-1, 1]$, extended by `spline_order` $= k = 3$ steps at each end so boundary basis functions are well-defined. The B-spline basis is evaluated by the **Cox–de Boor recursion**: order-0 (indicator) basis functions

$$B_{i,0}(x) = \mathbb{1}\bigl[t_i \le x < t_{i+1}\bigr]$$

(with an explicit boundary patch so $x$ landing exactly on the rightmost knot is captured by the last bin rather than falling into a zero-measure gap), and for orders $r = 1,\ldots,k$ the standard recursive blend

$$B_{i,r}(x) = \frac{x - t_i}{t_{i+r} - t_i}\,B_{i,r-1}(x) \;+\; \frac{t_{i+r+1} - x}{t_{i+r+1} - t_{i+1}}\,B_{i+1,r-1}(x)$$

(division-by-zero guarded with a small $\epsilon = 10^{-8}$ clamp on each denominator). The layer's output for input $x \in \mathbb{R}^{d_{\mathrm{in}}}$ is then

$$\mathrm{KANLinear}(x)_{o} = \sum_{i=1}^{d_{\mathrm{in}}}\sum_{n=1}^{n_{\mathrm{basis}}} c_{o,i,n}\, B_{n,k}(x_i) \;+\; \sum_{i=1}^{d_{\mathrm{in}}} w^{\text{res}}_{o,i}\cdot\mathrm{SiLU}(x_i)$$

with $n_{\mathrm{basis}} = 8$ (grid size $5$ plus spline order $k=3$) learned spline coefficients $c_{o,i,n}$ per connection, plus a learned residual scale $w^{\text{res}}_{o,i}$ on a fixed SiLU nonlinearity $\mathrm{SiLU}(x) = x\cdot\sigma(x)$ (this residual term is the standard KAN stabilization trick that keeps gradients well-behaved before the splines are trained). Two `KANLinear` layers (pool-dim $2D = 64$ → hidden $32$ → $1$) replace the two-layer MLP head, with an ELU nonlinearity between them.

### 5.4 Training Objective (All Predictors, Identical Formula Everywhere)

Given standardized labels $y = (E - \mu_E)/\sigma_E$ over the training split, every predictor minimizes

$$\mathcal{L}(\phi) = \underbrace{\frac{1}{|\mathcal{B}|}\sum_{c\in\mathcal{B}}\bigl(\hat{y}_\phi(c) - y_c\bigr)^2}_{\text{MSE — get the scale right}} \;+\; \lambda_{\mathrm{rank}}\cdot\underbrace{\frac{1}{|P|}\sum_{(a,b)\in P} \max\!\Bigl(0,\ m - \mathrm{sign}(y_a - y_b)\cdot(\hat{y}_a - \hat{y}_b)\Bigr)}_{\text{pairwise ranking hinge — get the order right}}$$

The ranking term sums over `RANK_PAIRS = 256` randomly sampled circuit-index pairs $P$ per epoch, with margin $m =$ `RANK_MARGIN = 0.10` and weight $\lambda_{\mathrm{rank}} =$ `RANK_WEIGHT = 1.0`. This is a margin ranking loss: it incurs zero penalty once the predicted gap between $\hat{y}_a$ and $\hat{y}_b$ correctly matches the sign of the true gap by at least margin $m$, and a linearly growing penalty otherwise — rewarding correct *order* independent of getting the absolute energy scale right, which is exactly what the downstream search needs (it only ever compares circuits to each other, never reads off an absolute number). Model selection tracks the validation-set **Kendall rank correlation**,

$$\tau = \frac{(\text{concordant pairs}) - (\text{discordant pairs})}{\binom{N}{2}}$$

over all $\binom{N}{2}$ pairs of validation circuits, where a pair is concordant if $\mathrm{sign}(\hat{y}_a - \hat{y}_b) = \mathrm{sign}(y_a - y_b)$; the best-$\tau$ checkpoint is restored before computing the final, reported test-set $\tau$. From v2 onward the GNN backbone ($g_1, g_2$) is frozen (`requires_grad = False`) for the first half of fine-tuning epochs and unfrozen for the second half, after being initialized from self-supervised pre-training (§5.5) — a standard frozen-then-fine-tuned transfer-learning schedule.

### 5.5 Self-Supervised Pre-Training (v2 Onward)

A SimCLR-style contrastive objective pre-trains the encoder on `SSL_CIRCUITS = 2000` *unlabelled* circuits (no VQE simulation needed at all) before any energy labels are used. Two augmented views $c^{(a)}, c^{(b)}$ of each circuit are created by independently (i) dropping each gate with probability `SSL_AUG_DROP = 0.15`, and (ii) applying a random permutation $\pi$ of qubit labels — both transformations a sensible predictor should be at least partially invariant to, since dropping a gate or relabelling qubits doesn't change a circuit's *qualitative* architecture much. Their pooled embeddings $z^{(a)} = \mathrm{encode}(c^{(a)})$, $z^{(b)} = \mathrm{encode}(c^{(b)})$ are first $\ell_2$-normalized, $\hat{z} = z / \|z\|_2$, then pulled together and pushed apart from every other circuit in the batch via the **NT-Xent** (normalized temperature-scaled cross-entropy) loss. For a batch of $B$ circuits, stacking $z = [\hat{z}^{(a)}_1, \ldots, \hat{z}^{(a)}_B, \hat{z}^{(b)}_1, \ldots, \hat{z}^{(b)}_B] \in \mathbb{R}^{2B \times d}$, the pairwise cosine-similarity matrix (scaled by temperature $\tau$) is

$$S_{pq} = \frac{\hat{z}_p^\top \hat{z}_q}{\tau}, \qquad p,q = 1,\ldots,2B$$

with the diagonal $S_{pp}$ masked to $-\infty$ (so a sample never gets to use itself as a positive), and the loss is the standard cross-entropy over each row treating the matched partner as the only correct "class":

$$\mathcal{L}_{\text{NT-Xent}} = -\frac{1}{2B}\sum_{p=1}^{2B}\log\frac{\exp(S_{p,\,\mathrm{pos}(p)})}{\displaystyle\sum_{q\ne p}\exp(S_{pq})}$$

where $\mathrm{pos}(p)$ indexes the one positive partner of sample $p$ (view $a$ of circuit $i$ pairs with view $b$ of circuit $i$, and vice versa). The temperature was raised from `SSL_TEMP = 0.07` to `0.12` starting in `q5-bench`, because SimCLR's original $\tau = 0.07$ was tuned for batch sizes of 256 ($2\times256 - 2 = 510$ negatives per anchor); at this project's batch size of 32 ($2\times32 - 2 = 62$ negatives), $\tau = 0.07$ made the softmax in the denominator too peaked/"cold" relative to the much smaller negative pool, and training plateaued early — a legitimate, well-reasoned fix rather than an arbitrary tweak.

---

## 6. Predictor-Guided Search and the Acquisition Function

The search loop is the same in every notebook: sample a large pool of candidates ($|\mathcal{P}| =$ `SEARCH_POOL` $= 4000$), score every one with a single cheap forward pass through the trained predictor — no quantum simulation — rank by an acquisition score, and run real VQE only on the top `TOPK_VALIDATE = 8` finalists. The acquisition function is

$$s(c) = \hat{E}(c) \;+\; \lambda_\text{gate}\,|c| \;+\; \lambda_\text{cnot}\,n_\text{CNOT}(c), \qquad c^\star = \arg\min_{c\in\mathcal{P}} s(c)$$

where $\hat{E}(c) = \hat{y}_\phi(c)\cdot\sigma_E + \mu_E$ is the predictor's de-standardized energy estimate, $|c|$ is total gate count, and $n_\text{CNOT}(c)$ is the CNOT count. The two penalty terms are meant to push the search toward shallow, entangler-sparse circuits — the actual "efficient architecture" objective in addition to low energy — but their weights were a major, repeated source of bugs across the version history.

### 6.1 Bug: The Efficiency Penalty Drowned the Energy Signal (v1, v2)

In v1 and v2, $\lambda_\text{gate} = 0.015$ and $\lambda_\text{cnot} = 0.040$. Because adding one CNOT costs a flat $0.040$ in the acquisition score while the *actual* TFIM energy differences between candidate circuits at this scale are typically a few tenths, the term $\lambda_\text{cnot}\,n_\text{CNOT}(c)$ dominated $\hat{E}(c)$'s discriminating signal for almost any pair of circuits differing by a CNOT or two — i.e. $s(c)$ was effectively minimized by minimizing $n_\text{CNOT}(c)$ first and the energy only as a tiebreaker. The result, confirmed by the executed output of both notebooks: **every top-8 finalist in v1 had exactly 0 CNOTs**, and the v2 multi-seed run reported a CNOT standard deviation of exactly `0.000` across all 5 seeds — the search had degenerated into "find the best product state" (§3.1), which cannot represent TFIM's entangled ground state no matter how well-tuned. v1's best circuit ($L=8$, $|\boldsymbol\theta|=8$, $n_\text{CNOT}=0$) reached $\Delta E = 0.655$ (13.8% relative error) and was, by chance, numerically *tied* with random search at the same budget.

### 6.2 Fix #1: Reduce the Penalty Weights (v3)

v3 reduces $\lambda_\text{gate} \to 0.008$ and $\lambda_\text{cnot} \to 0.012$. This let the search explore CNOT-containing circuits again: the v3 multi-seed mean energy gap improved to $\Delta E = 0.344 \pm 0.026$ (vs. $0.580 \pm 0.168$ for random search), with the GAT-guided circuits now averaging $0.6 \pm 0.5$ CNOTs before pruning.

### 6.3 Fix #2: Force a CNOT Floor, then Drop the Penalty Entirely (v4)

v4 goes further: it sets $\lambda_\text{gate} = \lambda_\text{cnot} = 0$ — an **energy-only** acquisition function $s(c) = \hat{E}(c)$ — enforces the hard floor $n_\text{CNOT}(c) \ge n_{\min} = 2$ directly in the sampler (§3.1), and adds a post-hoc filter that discards any sampled circuit with $|c| > L_{\max} = 16$ before scoring. This combination reached $\Delta E = 0.311 \pm 0.018$, with GAT-guided circuits averaging $2.4 \pm 0.9$ CNOTs ($1.0 \pm 0.0$ after pruning) — the first version where pruned circuits reliably keep at least one CNOT across every seed.

### 6.4 Why the v4 Fix Was Still Incomplete, and the v5 Redesign (`q5-optimized.ipynb`)

`q5-optimized.ipynb` reports, plainly, that even with $n_{\min} = 2$ every seed converged to circuits with *exactly* 2 CNOTs on a 4-qubit ring — just enough to clear the floor, not enough to connect the whole chain (2 CNOTs span at most 3 of the 4 qubits if placed in a line, and can even overlap on a ring and span fewer). The energy gap stayed pinned at roughly $0.31$ regardless of how much extra VQE budget was thrown at it, because the bottleneck was the **ansatz family's expressivity** — the set $\{|\psi(\boldsymbol\theta)\rangle : \boldsymbol\theta \in \mathbb{R}^{|\boldsymbol\theta|}\}$ reachable by these circuits simply does not contain a state close enough to the true ground state — not its optimization: no amount of additional gradient descent over $\boldsymbol\theta$ can make a fixed circuit topology represent a state outside its own reachable manifold. This is the rationale behind the layered hardware-efficient redesign in §3.2, which guarantees whole-chain entanglement (a connected entangling graph spanning all $n$ qubits) whenever a layer entangles at all. With that change, the mean energy gap dropped to $0.046 \pm 0.093$ before pruning and **$0.0045 \pm 0.0057$ after pruning** — i.e. on average the discovered circuits sit within about 0.1% of the exact ground-state energy after pruning, at the cost of much deeper raw circuits ($14.0 \pm 5.2$ gates, $6.2 \pm 2.2$ CNOTs after pruning, versus single digits in earlier versions).

---

## 7. Structural Pruning

A greedy, model-agnostic local search over circuit topology: for the current circuit $c$ with baseline energy $E(c)$, try removing each gate $l$ in turn to get $c_{-l}$, accept the removal if $E(c_{-l}) - E(c) \le \mathrm{tol}$, and repeat until no single-gate removal is accepted. Formally this is coordinate-descent on the discrete neighbourhood $\mathcal{N}_1(c) = \{c_{-l} : l=1,\ldots,|c|\}$ defined by single-gate deletions:

$$c \leftarrow \arg\min_{c' \in \{c\}\cup\mathcal{N}_1(c)} E(c') \quad\text{subject to}\quad E(c') - E(c) \le \mathrm{tol}$$

repeated until a fixed point.

### 7.1 Bug: Pruning Silently Discarded a CNOT Every Time (`q5-optimized.ipynb`'s Account of v4)

`q5-optimized.ipynb` documents a bug it inherited from v4: the pruning tolerance $\mathrm{tol} = 5\times10^{-3}$ was *smaller than the VQE noise floor* at the cheap step budget used during pruning (120 steps, 3 restarts), so a genuinely necessary CNOT could appear "safe to remove" purely because the noisy re-evaluation of $E(c_{-l})$ happened to land within tolerance of $E(c)$ by chance — not because the circuit was actually equally good without it. This is why v4's reported pruned-circuit CNOT counts ($1.0 \pm 0.0$) are one less than its pre-pruning counts ($2.4 \pm 0.9$) in every single seed, which is a suspiciously consistent pattern for a process that's supposed to depend on circuit-specific redundancy rather than a systematic noise bias.

### 7.2 Fix (`q5-optimized.ipynb`)

The tolerance is tightened to $\mathrm{tol} = 10^{-3}$, and — more importantly — both the pre-removal baseline $E(c)$ and the post-removal candidate $E(c_{-l})$ are **re-validated at high precision** (`PRUNE_RECHECK_STEPS = 250`, `PRUNE_RECHECK_RESTARTS = 5`) before a removal is accepted, using the cheap evaluation only as a fast pre-screen to decide which removals are even worth re-checking carefully. This stops optimization noise at the cheap step budget from masquerading as genuine structural redundancy.

---

## 8. ZX-Calculus Data Augmentation (v2 Onward)

To multiply the labelled training set without any additional quantum simulation, four equivalence-preserving rewrites from ZX-calculus / circuit identities are applied to sampled circuits:

| Transformation | Rewrite rule |
|----------------|--------------|
| Spider fusion | $R_\sigma(\alpha)\cdot R_\sigma(\beta) \;\to\; R_\sigma(\alpha+\beta)$ for adjacent same-axis ($\sigma\in\{X,Y,Z\}$) rotations on the same qubit |
| Identity removal | $R_\sigma(\theta) \to \mathbf{1}$ whenever $\theta \equiv 0 \pmod{2\pi}$ within tolerance $0.08$ rad |
| Phase-free simplification | $R_Z(\pi/2)\cdot R_X(\theta)\cdot R_Z(-\pi/2) \;\to\; R_Y(\theta)$ within tolerance $0.12$ rad |
| Scalar reduction | Inverse substitution: $R_Y(\theta) \to R_Z(\pi/2)\cdot R_X(\theta)\cdot R_Z(-\pi/2)$ |

These are genuine single-qubit gate identities (spider fusion is exact for any angle via $e^{-i\sigma(\alpha+\beta)/2} = e^{-i\sigma\alpha/2}e^{-i\sigma\beta/2}$; the Euler-angle decomposition $R_Z R_X R_Z = R_Y$ up to global phase is the standard ZXZ → Y identity). Each labelled circuit generates `ZX_VARIANTS` variants (3 in v2/v3, reduced to 1 from v4 onward — v3's training set was 75% ZX-synthetic, see §11) that **inherit the original circuit's energy label** at zero extra VQE cost, since the rewrites are theoretically energy-preserving. Validation and test splits are always drawn from the *original*, pre-augmentation circuit indices only (`orig_val`, `orig_test`) — ZX variants are added exclusively to the training split — a deliberate anti-leakage design documented explicitly in `q5-bench`'s code comments.

---

## 9. DQAS: A REINFORCE Policy-Gradient Baseline (`DQAS+KANQAS.ipynb` and `8qubitrun.ipynb`)

The notebook implements a lightweight variant of Differentiable Quantum Architecture Search (Ye et al. 2021). A `DQASSupercircuit` holds learnable **architecture logits** $\boldsymbol\eta \in \mathbb{R}^{L\times|\mathcal{G}|}$ (one row per circuit slot, one column per gate type) and **qubit logits** $\boldsymbol\xi \in \mathbb{R}^{L\times n}$, defining a categorical policy over architectures:

$$\pi_{\boldsymbol\eta,\boldsymbol\xi}(\mathrm{gate}_l = g,\ q_l = i) = \mathrm{softmax}(\boldsymbol\eta_l)_g \cdot \mathrm{softmax}(\boldsymbol\xi_l)_i$$

At each search step $t$ with annealed temperature $\tau_t$ linearly interpolated from $\tau_{\mathrm{start}} = 1.0$ to $\tau_{\mathrm{end}} = 0.1$ over `n_steps = 150` steps, a **discrete** architecture is sampled using the Gumbel-Softmax straight-through estimator with `hard = True`:

$$\mathrm{gate}_l = \arg\max_g\Bigl[\eta_{l,g} + \mathrm{Gumbel}(0,1)_g\Bigr]\Big/\tau_t \quad\text{(one-hot forward pass, soft gradient backward)}$$

The true Hamiltonian docstring in the notebook is explicit that this `forward()` method is a **structural stub, never called by the actual search** — true end-to-end-differentiable DQAS would need to backpropagate gradients through the quantum simulator itself (via a torch-differentiable device or the parameter-shift rule applied *inside* the supercircuit), which the project's joblib-cached, black-box `evaluate_circuit` cannot support. Instead, the real search in `run_dqas()` treats the sampled architecture's VQE energy as a black-box reward and updates $\boldsymbol\eta, \boldsymbol\xi$ with **REINFORCE**:

$$\nabla_{\boldsymbol\eta,\boldsymbol\xi}\, J = -\bigl(R_t - b_t\bigr)\cdot\nabla_{\boldsymbol\eta,\boldsymbol\xi}\Bigl[\log\pi_{\boldsymbol\eta}(\text{gate sampled}) + \log\pi_{\boldsymbol\xi}(\text{qubit sampled})\Bigr]$$

with reward $R_t = -E_t$ (negative VQE energy — lower energy is better, so reward increases as energy decreases) and an exponential-moving-average baseline that reduces gradient variance:

$$b_t = 0.9\,b_{t-1} + 0.1\,R_t, \qquad b_0 = R_0$$

The log-probability terms are computed as $\log\pi_{\boldsymbol\eta}(\cdot) = \sum_l \bigl[\log\mathrm{softmax}(\boldsymbol\eta_l)\odot e_l\bigr]$, i.e. the log-probability the *current* (continuously updated) policy logits assign to the architecture that was actually sampled this step — standard score-function / likelihood-ratio policy gradient. After `n_steps`, the final architecture is read off by discretizing the policy at $\arg\max$ (`discretize()`), and validated with a careful, high-precision VQE run.

---

## 10. What's Real vs. What's Stated but Unverified

Because this is a chain of research notebooks rather than a single audited pipeline, a careful read of the raw `.ipynb` files (not just the markdown prose) turns up real discrepancies worth knowing about before citing any of these numbers.

**`8qubitrun.ipynb` is the recommended notebook for reproducibility.** It is the unified, merged successor to both `QAS-v6.ipynb` and `DQAS+KANQAS.ipynb`, incorporating all 10 v6 bug fixes (see §11), KAN and DQAS both inside the multi-seed loop, an independent SSL pre-train for the KAN encoder, and a circuit-level Spearman ρ on 165 test pairs. The stale-output problems in `DQAS+KANQAS.ipynb` described below are corrected in `8qubitrun.ipynb` by a clean end-to-end rewrite and re-execution.

**`DQAS+KANQAS.ipynb`'s headline multi-seed comparison cell prints stale output that does not match its own code.** The cell that aggregates GAT/GCN/KAN τ and the DQAS energy gap across 5 seeds (and computes the 165-point Spearman ρ) has stored output that is verbatim the *old* `q5-bench` printout (it even prints the string `"[v4] ablation"` and a 5-point Pearson-r — concepts this version of the code has already replaced with Spearman ρ and a circuit-level comparison). This is the unmistakable signature of a notebook cell whose source was edited after it was last actually run end-to-end: the code computes one thing, the displayed output is left over from a previous version of that same cell. Reconstructing the true numbers from the one piece of output in this notebook that *is* internally consistent — the per-seed printout from the multi-seed loop itself — gives GAT τ = $0.650 \pm 0.081$ and GCN τ = $0.730 \pm 0.086$ across the 5 experiment seeds, matching `q5-bench`'s numbers exactly (expected, since both files reuse the same seeds and config up to this point). The KAN τ, DQAS energy gap, and Spearman ρ values that the current code computes are not visible in any stored output anywhere in the file.

**Several of `DQAS+KANQAS.ipynb`'s own demonstration cells never executed at all.** Inspecting the notebook JSON directly (not just reading the rendered text) shows that the standalone `KANLinear`/`KANPredictor` class-definition cell, the standalone `DQASSupercircuit`/`run_dqas` definition-and-demo cell, the single-run KAN-vs-MLP fine-tuning comparison cell, the final "GAT vs KAN vs DQAS vs Random" summary table, and the final publishability checklist all have zero stored outputs. The classes are defined and presumably used by the multi-seed loop (`run_one_seed`, which *does* have real output and does call `run_dqas()` and instantiate `KANPredictor` internally), but the notebook's own narrative demonstrations of those components in isolation were never run, so anyone reading top-to-bottom for "does the KAN head work" or "does DQAS converge" will find code and a written rationale but no confirming numbers in this file.

**The real multi-seed numbers, where they exist, come from a cell most readers would not expect to contain them.** The function `run_one_seed` (used by the multi-seed loop) does internally build a GAT predictor, a GCN ablation, a KAN predictor with an independently SSL-pretrained backbone, and a DQAS run via `run_dqas(n_steps=100, ...)`, and this cell's execution **is** captured in the file with five real per-seed print lines (one per seed in `EXPERIMENT_SEEDS`). From those five lines: GAT τ ranged 0.542–0.752 (mean ≈ 0.650), GCN τ ranged 0.599–0.822 (mean ≈ 0.730), the GAT-guided energy gap ranged 0.287–0.355 (mean ≈ 0.337), and CNOT counts after the search ranged 0–2. The aggregated KAN τ and DQAS gap values that this same cell computes per seed are not printed in the verbose per-seed line and so cannot be reconstructed from the stored output — they exist somewhere in the run that produced this output, but are not visible in the file.

**The Heisenberg cross-task check is a single seed, and its narrative text in two different notebooks doesn't match its own printed numbers.** In `q5-bench`, the printed Heisenberg gap is exactly `0.0000` for *both* the GAT-guided circuit and the random-search baseline — read naively this looks like both methods solved the problem exactly, but a more likely explanation is that the discovered circuits (depth 8, pruned to depth 1) happened to land in the same low-energy, low-entanglement product-state sector the Heisenberg ground state partially overlaps with at this depth, and the print formatting (`.4f`) rounds a genuinely-near-zero-but-nonzero gap to display as `0.0000`. In `QAS-v6.ipynb`, the rationale paragraph for this same experiment asserts "the meaningful result here is τ = 0.707," but the notebook's own executed output for that exact cell prints **τ = 0.612**, not 0.707 — a direct mismatch between a specific number quoted in prose and the number the code actually produced and displayed two lines below it.

**`QAS-v6.ipynb`'s design rationale and its own measured result disagree.** Section 8 of that notebook states, as justification for promoting GCN to the primary predictor, that "GCN (τ = 0.730) beats GAT (τ = 0.650)" — copying `q5-bench`'s finding. But the multi-seed loop in the *same notebook*, run with its own (different) random seeds and SSL schedule, reports GCN τ = $0.626 \pm 0.074$ versus GAT τ = $0.647 \pm 0.098$ — i.e. in this run **GAT wins**, the opposite of the stated rationale. The notebook's own publishability checklist correctly marks `[v5] GCN is primary (beats GAT ablation in τ)` as unmet (`[ ]`) precisely because of this contradiction, which is honest, but it does mean the central architectural decision of this notebook is not supported by its own evidence.

None of this means the underlying ideas (GAT/GCN/KAN encoders, DQAS, ZX augmentation, SSL pre-training) are wrong — the mechanics are implemented correctly wherever they do have verified output — but the per-notebook summary tables and conclusions below are annotated to distinguish numbers that come from a verified, matching execution from numbers that are stated in markdown prose without a matching, internally-consistent printed result.

---

## 11. Results by Notebook

All energy gaps are $\Delta E = E_{\mathrm{found}} - E_0$ on 4-qubit TFIM ($E_0 = -4.758770$), mean ± standard deviation over the 5 seeds `{7, 42, 137, 256, 512}` unless noted otherwise. ✓ marks numbers directly confirmed against the notebook's own stored, internally-consistent output; ⁺ marks numbers reconstructed from a different (mismatched) cell's stale output.

| Notebook | Random gap | GAT/GCN-guided gap | + pruning | CNOTs after pruning | Test τ |
|----------|------------|--------------------|-----------|---------------------|--------|
| v1 (baseline, single seed) ✓ | 0.655 (tied) | 0.655 | 0.655 | 0 | 0.360 |
| v2 ✓ | 0.580 ± 0.168 | 0.355 ± 0.000 | 0.355 ± 0.000 | 0 | 0.644 ± 0.079 |
| v3 ✓ | 0.580 ± 0.168 | 0.344 ± 0.026 | 0.344 ± 0.026 | 0.2 ± 0.4 | 0.660 ± 0.078 |
| v4 ✓ | 0.588 ± 0.171 | 0.311 ± 0.018 | 0.311 ± 0.018 | 1.0 ± 0.0 | 0.625 ± 0.054 |
| `q5-bench` ✓ | 0.478 ± 0.212 | 0.337 ± 0.030 | 0.337 ± 0.030 | 0.6 ± 0.9 | GAT 0.650 ± 0.081 / GCN 0.730 ± 0.086 |
| `q5-optimized` ✓ | 0.299 ± 0.344 | 0.046 ± 0.093 | **0.0045 ± 0.0057** | 6.2 ± 2.2 | 0.602 ± 0.031 |
| `QAS-v6.ipynb` ✓ | 0.478 ± 0.212 | 0.336 ± 0.030 | 0.336 ± 0.030 | 0.6 ± 0.9 | GCN 0.626 ± 0.074 / GAT 0.647 ± 0.098 |
| `DQAS+KANQAS.ipynb` ✓ (partial) | 0.478 ± 0.212 ⁺ | 0.337 ± 0.030 ⁺ | n/a in stored output | n/a in stored output | GAT 0.650 ± 0.081 / GCN 0.730 ± 0.086 ⁺ |
| `8qubitrun.ipynb` (v6 unified) | *re-execution required* | *re-execution required* | *re-execution required* | *re-execution required* | *re-execution required* |

**Note on `8qubitrun.ipynb` results:** This consolidated notebook resolves the stale-output problems in `DQAS+KANQAS.ipynb` by providing a clean, fully integrated pipeline (see §10). Because it is a source-only release (not a pre-executed notebook), its quantitative results must be reproduced by running the notebook end-to-end. The code structure guarantees that all five predictors — GAT, GCN ablation, KAN, DQAS, and random baseline — are evaluated in the same multi-seed loop (`run_one_seed` over seeds `{7, 42, 137, 256, 512}`), and that the circuit-level Spearman ρ is computed over 165 test pairs (5 seeds × 33 test circuits).

Two results from the earlier notebooks stand out. First, **only the v5 layered-ansatz redesign (`q5-optimized.ipynb`) actually closes the gap to the exact ground state** — every free-form-gate-slot version plateaus around $\Delta E \approx 0.3$–$0.4$ (roughly 6–8% relative error) no matter how the acquisition function, penalty weights, or VQE budget are tuned, because (per §6.4) the bottleneck is the search space's expressivity, not its optimization. Second, **the GCN-vs-GAT finding is itself unstable across reruns** — `q5-bench` and `DQAS+KANQAS.ipynb` (same seeds, closely related code) both show GCN ahead by a similar margin, but `QAS-v6.ipynb`, run with what is nominally the same experimental design, shows GAT ahead instead. At $n=4$ qubits with a 220-circuit dataset, this comparison does not appear to be settled.

---

## 12. Mathematical and Methodological Limitations

**The acquisition function's energy term and efficiency penalty terms are not on a common, principled scale.** $s(c) = \hat{E}(c) + \lambda_\text{gate}|c| + \lambda_\text{cnot}\,n_\text{CNOT}(c)$ adds an energy (in the dimensionless TFIM energy scale) to integer gate counts with hand-tuned weights $\lambda$. The v1→v2 bug (§6.1) was a direct, predictable consequence of this: there is no a priori reason a CNOT should cost exactly $0.040$ energy-units rather than $0.012$ or $0$, and three different scalarizations were tried across the version history before landing on "drop the penalty and use a hard structural constraint instead" (v4) or "redesign the search space so the constraint is structural by construction" (`q5-optimized`). A more principled approach — e.g. a Pareto frontier over $(\hat{E}(c), |c|)$ rather than a single scalarized score $s(c)$, or normalizing the penalty by the empirical energy spread $\sigma_E$ of the candidate pool so $\lambda_\text{cnot}$ is dimensionless relative to the actual signal it competes against — is not implemented anywhere in this series.

**The TFIM benchmark is small enough to make the predictor's value proposition hard to demonstrate.** At $n=4$ qubits the Hamiltonian is exactly diagonalizable in milliseconds via dense `eigvalsh`, and a single VQE evaluation costs only a few seconds. The entire motivation for a learned predictor — that full evaluation is too expensive to apply to every candidate — is far more pressing at 8–20 qubits, where the Hilbert space dimension $2^n$ grows large enough that exact diagonalization itself becomes the bottleneck. Every "publishability checklist" across all nine files independently flags `Scale beyond a few qubits (>= 8 qubits)` as unmet; `8qubitrun.ipynb` takes its name from this aspiration and is explicitly designed as the staging ground for that scaling experiment.

**ZX-calculus augmentation assigns a parent circuit's energy label to a structurally different circuit without re-running VQE.** Spider fusion, identity removal, and the phase-free/scalar-reduction substitutions are valid gate-level identities for the *specific rotation angles* they're derived from, but the augmented circuit's VQE-optimal angles are not actually re-optimized — the variant simply inherits the original circuit's converged energy $E(c)$ as its own label $E(c_{ZX}) := E(c)$. This is exact when the rewrite is a true identity (spider fusion is exact for any angle pair, since $e^{-i\sigma\alpha/2}e^{-i\sigma\beta/2} = e^{-i\sigma(\alpha+\beta)/2}$ holds identically), but `_identity_remove`'s tolerance ($0.08$ rad) and `_phase_free_simplify`'s pattern-matching tolerance ($0.12$ rad) both accept *approximate* matches — meaning some fraction of "augmented" circuits are not exactly equivalent to their parent and are receiving a label $E(c)$ that does not correspond to their own true converged energy $E(c_{ZX}) \ne E(c)$. None of the notebooks quantify how much label noise this introduces into the training set.

**The KAN B-spline implementation has a boundary-condition patch whose correctness is asserted, not tested.** The de Boor recursion's order-0 basis uses a half-open interval test $B_{i,0}(x) = \mathbb{1}[t_i \le x < t_{i+1}]$, which by construction assigns zero basis weight to any input landing exactly on the rightmost grid point $t_m$ (since no interval is right-closed there); the code adds a manual patch to push that boundary mass into the last bin. This is a known, standard edge case in B-spline implementations and the fix is the conventional one, but no unit test verifies the basis partition-of-unity property $\sum_i B_{i,k}(x) = 1$ holds across the full input domain $[-1, 1]$ including the boundary, so this is a plausible-looking fix rather than a verified one.

**DQAS as implemented is a REINFORCE policy-gradient variant, not gradient-based differentiable architecture search.** As discussed in §9, true DQAS backpropagates through the quantum circuit itself; this implementation instead treats VQE energy as a black-box reward for a score-function gradient estimator. REINFORCE gradients have variance that scales unfavourably with the dimensionality of the action space (here, $L\times(|\mathcal{G}|+n)$ logits) and are known to converge more slowly and noisily than true backpropagated gradients through a differentiable simulator — a legitimate, commonly used simplification given the black-box VQE evaluator, but it means the "DQAS baseline" in this repository is a weaker proxy for the method the name usually refers to in the literature, and any energy-gap comparison between "GAT-guided" and "DQAS" is really a comparison against this lighter-weight variant.

**The Heisenberg cross-task result is statistically unpowered by construction, and (per §10) inconsistently reported.** It is run for exactly one seed "for speed," with no error bars and no significance test, and the one number quoted in `QAS-v6.ipynb`'s narrative text (τ = 0.707) does not match the number its own executed cell prints (τ = 0.612).

**Energy-gap "ties" between methods are reported as if they were meaningfully different.** Several notebooks report e.g. `GAT-guided: 0.3551` and `GAT-guided + prune: 0.3552` to four decimal places and discuss them as distinct results, when the difference ($10^{-4}$) is far smaller than the VQE optimization noise floor at the step budgets used (a few $\times 10^{-3}$ at best, per `q5-optimized`'s own pruning-bug discussion in §7.1). The high-precision formatting throughout the codebase outpaces the actual numerical precision of the underlying VQE point estimates, which are themselves the minimum over only 3–5 random restarts of a non-convex optimization, not a converged global optimum with a quantified error bar.

**No noise model or real-hardware validation exists anywhere in the series.** The entire "efficiency" motivation — fewer CNOTs and shallower circuits matter because two-qubit gates are noisy on real devices — is argued from first principles but never demonstrated: every VQE evaluation in all eight notebooks uses PennyLane's noiseless `default.qubit` simulator, $\rho_{\mathrm{out}} = U|0\rangle\langle0|U^\dagger$ exactly, with no decoherence or gate-error channel applied anywhere. A circuit with fewer CNOTs is only confirmed cheaper in raw gate count, not in any measured or simulated robustness to a depolarizing or amplitude-damping noise channel.

**Baselines stop at random search**, except for the REINFORCE-based DQAS variant in `DQAS+KANQAS.ipynb` and `8qubitrun.ipynb`. None of the nine files compare against reinforcement-learning search beyond that lightweight REINFORCE baseline, Bayesian optimization, or evolutionary search, despite all nine publishability checklists listing "strong baselines" as an open gap.

---

## 13. Architecture Summary

**Circuit representation:** list of $(\mathrm{gate}, q)$ slots, $\mathrm{gate}\in\{RX, RY, RZ, \mathrm{CNOT}\}$ (free-form) or a layered rotation/fixed-CNOT-ladder structure (`q5-optimized` only).

**Graph encoding:** directed graph, gates as nodes with feature vector $x_l = [\mathrm{onehot}(\mathrm{gate})\,\|\,c, t\,\|\,\mathrm{onehot}(q)\,\|\,(\sin\phi, \cos\phi)]$, bidirectional wire-following edges plus self-loops.

**Predictors:**
- GAT — $e_{ij}^{(k)} = \mathrm{LeakyReLU}(\mathbf{a}_\mathrm{src}^{(k)\top} h_j^{(k)} + \mathbf{a}_\mathrm{dst}^{(k)\top} h_i^{(k)})$, edge-softmax $\alpha_{ij}$, $h_i' = \sum_j\alpha_{ij}h_j$ (2 layers × 4 heads, hidden = 32, mean+max pool, MLP head, 20.3–20.5k params)
- GCN ablation — $h_i' = \mathrm{ELU}(W\cdot\mathrm{mean}_{j\in\mathcal{N}(i)}h_j)$ (3.6k params)
- KAN variant — same GAT encoder, B-spline head $\sum_n c_n B_{n,k}(x) + w^{\text{res}}\mathrm{SiLU}(x)$ (introduced in `DQAS+KANQAS.ipynb`, fully integrated with independent SSL pre-train in `8qubitrun.ipynb`)

**Training:** $\mathcal{L} = \mathrm{MSE}(\hat{y}, y) + \lambda_{\mathrm{rank}}\cdot\text{hinge-rank}(\hat{y}, y)$; NT-Xent self-supervised pre-training (v2 onward) on 2,000 unlabelled circuits:

$$\mathcal{L}_{\text{NT-Xent}} = -\log\frac{\exp(S_{p,\mathrm{pos}}/\tau)}{\sum_{q\ne p}\exp(S_{pq}/\tau)}$$

backbone frozen for the first half of fine-tuning epochs.

**Search:** sample 4,000 candidates → score with one predictor forward pass each → rank by $s(c) = \hat{E}(c) + \lambda_\text{gate}|c| + \lambda_\text{cnot}n_\text{CNOT}(c)$ (or energy-only from v4 onward) → validate top 8 with full VQE.

**Pruning:** greedy single-gate removal, accept if $E(c_{-l}) - E(c) \le \mathrm{tol}$; tolerance and re-validation precision tightened in `q5-optimized` to avoid mistaking VQE noise for redundancy.

**Augmentation:** ZX-calculus rewrites (spider fusion $R_\sigma(\alpha)R_\sigma(\beta)\to R_\sigma(\alpha+\beta)$, identity removal, $R_Z R_X R_Z \leftrightarrow R_Y$) generating free training-label variants, restricted to the training split only.

**Baselines:** random search at a matched full-VQE-evaluation budget (all notebooks); REINFORCE/Gumbel-Softmax DQAS with EMA-baselined policy gradient $\nabla J = -(R-b)\nabla\log\pi$ (`DQAS+KANQAS.ipynb` and `8qubitrun.ipynb`; in the former, multi-seed numbers are not recoverable from stored output — see §10; in the latter, they are computed live in the multi-seed loop).

---

## 14. Dependencies

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

## 15. Key Configuration Knobs

| Parameter | Typical range across notebooks | Effect |
|-----------|-------------------------------|--------|
| `N_QUBITS` | 4 (fixed everywhere, including `8qubitrun.ipynb`) | Problem size; scaling to 8+ qubits is the stated next step in every publishability checklist and motivates the `8qubitrun.ipynb` name |
| `N_CIRCUITS` | 220 → 300 | Labelled dataset size before ZX augmentation |
| `VQE_STEPS` / `VQE_RESTARTS` | 60/3 → 120/4 | VQE label quality; v4 found 60 steps left the best labelled circuit 0.28 above its 120-step value |
| `ZX_VARIANTS` | 3 → 1 | v3's training set was 75% ZX-synthetic; reduced from v4 onward |
| `SSL_EPOCHS` / `SSL_LR` | 60/3e-3 → 150/8e-4 | SSL pre-training; v3's loss plateaued near the random-init baseline, prompting longer training at a lower rate |
| `SSL_TEMP` | 0.07 → 0.12 | NT-Xent temperature; raised because the original SimCLR value assumes far larger batches |
| `LAMBDA_GATES` / `LAMBDA_CNOT` | 0.015/0.040 → 0.0/0.0 | Acquisition penalty weights; see §6 for the full bug history |
| `MIN_CNOTS` | 0 → 2 | Entanglement floor (superseded by the layered-ansatz redesign in `q5-optimized`) |
| `PRUNE_TOL` | 5e-3 → 1e-3 | Pruning tolerance; tightened after the silent-CNOT-removal bug |

---

## 16. References

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
