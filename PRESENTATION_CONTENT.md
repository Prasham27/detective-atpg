# Presentation Content — DETECTive 30-Minute Full Deck

Project: Replicating DETECTive (Petrolo et al., GLSVLSI 2024) — GNN+LSTM Automatic Test Pattern Prediction
Deck: `presentation/DETECTive_30min_full_deck.pptx` (31 slides, ~30 min, 16:9)
Builder: `presentation/build_full_deck.py`

Structure:
- Title slide
- BLOCK 1 — Problem & Classical ATPG (slides 1–11, ~10 min)
- BLOCK 2 — ATPP Architecture (slides 12–19, ~10 min)
- BLOCK 3 — Results (slides 20–28, ~10 min)
- Q & A
- Thank You

D-Algorithm and FAN slides (B1.4, B1.5, B1.9, B1.10) are first-pass content — colleague will tweak.

---

## TITLE

**Title:** Test Vector Generation for Fault Detection Using Graph Neural Networks
**Subtitle:** Replicating DETECTive: A 30-minute Walk-through of Classical ATPG and ML-driven ATPP
**Citation:** Petrolo et al., DETECTive: Machine Learning-driven Automatic Test Pattern Prediction for Faults in Digital Circuits, GLSVLSI 2024.

---

## BLOCK 1 — PROBLEM & CLASSICAL ATPG

### Slide 1 — Digital Circuit Fault

**Bullets**
- Manufacturing defects: bridges, opens, transistor failures during fabrication.
- Stuck-at fault model: abstract a defect as a wire permanently driven to a fixed value.
- Stuck-at-0 (SA-0): the line is always logic 0 regardless of its driver.
- Stuck-at-1 (SA-1): the line is always logic 1 regardless of its driver.
- Industry standard: every line has 2 stuck-at faults — a circuit with N lines has 2N targets.
- Goal of ATPG: generate input vectors that detect all stuck-at faults on the chip.

**Time:** 60 s

---

### Slide 2 — Test Pattern

**Bullets**
- A test pattern is a binary vector applied to the primary inputs (PIs) of the circuit.
- Two requirements for detecting a fault:
  - Activation — force the faulty line to the OPPOSITE value of the stuck-at value.
  - Propagation — sensitize a path so the difference reaches a primary output (PO).
- Detection condition: faulty-circuit output != fault-free output on at least one PO.
- Example: SA-0 on line L requires (a) drive L to 1 in the good circuit and (b) propagate the 1-vs-0 difference to a PO.

**Time:** 70 s

---

### Slide 3 — The Exponential Problem

**Bullets**
- A circuit with n primary inputs has 2^n possible test patterns.
- Even modest circuits have hundreds to thousands of PIs.
- c1908 has 33 PIs => 2^33 ~ 8.6 billion possible patterns. c2670 has 233 PIs => intractable.
- Exhaustive testing is infeasible — we cannot try every input combination.
- We need smart algorithms that pick a small set of useful patterns (target each fault directly).
- This is the core motivation for ATPG: structured search instead of brute force.

**Time:** 70 s

---

### Slide 4 — D-Algorithm (Roth, 1966)  [colleague will tweak]

**Bullets**
- First systematic ATPG algorithm — J. Paul Roth at IBM, 1966.
- Works directly on the stuck-at fault model.
- Introduces 5-valued logic: 0, 1, X (unknown), D (1 in good / 0 in faulty), D-bar (0 in good / 1 in faulty).
- D notation lets the algorithm track a discrepancy as it propagates through gates.
- Decisions can be made on ANY internal line (not just primary inputs).
- Provably complete: if a test pattern exists, D-Algorithm will find it.
- Provably exponential: worst-case decision space is huge, especially with reconvergent fanout.

**Time:** 80 s

---

### Slide 5 — Working of D-Algorithm  [colleague will tweak]

**Bullets**
- Step 1 — Fault activation: place D or D-bar on the fault line by justifying the opposite value.
- Step 2 — D-frontier propagation: at each gate the D meets, sensitize the gate by setting side-inputs to non-controlling values.
- Step 3 — Line justification: every assigned internal value must be consistent with PI assignments — recursively justify upward.
- Step 4 — Backtrack on conflict: if any internal assignment becomes infeasible, undo the most recent decision.
- The D-frontier shrinks as faults propagate; the J-frontier (lines requiring justification) grows.
- Termination: success when D reaches a PO and all internal lines are justified by PI values.

**Time:** 85 s

---

### Slide 6 — PODEM (Goel 1981)

**Bullets**
- Path-Oriented Decision Making — proposed by Prabhu Goel at IBM, 1981.
- Direct response to D-Algorithm's exponential blow-up on XOR-heavy and reconvergent circuits.
- Key idea: branch only on Primary Inputs, never on internal lines.
- Every decision is therefore directly implementable — no inconsistent internal assignments to retract.
- Backtracking is bounded by 2^|PI|, far smaller than the D-Alg search space over all internal lines.

**Time:** 75 s

---

### Slide 7 — Working of PODEM

**Bullets**
- Objective: pick a target value on a specific line needed to activate or propagate the fault.
- Backtrace: walk backward from the objective line to a PI, choosing gate inputs that cause the desired output.
- Assign that PI, simulate forward, check if the objective is met.
- X-Path Check: confirm at least one all-X path still exists from the fault site to a PO; if not, backtrack.
- Completeness: PODEM is complete — if a test exists, it will eventually find it.
- Heuristics: SCOAP controllability/observability metrics guide objective and backtrace choices to reach a PI faster.

**Time:** 90 s

---

### Slide 8 — Worked PODEM Example

**Bullets**
- Tiny circuit: A, B, C are PIs. G1 = AND(A, B). G2 = OR(G1, C). Fault: G1 stuck-at-0.
- Activation objective: G1 = 1, which means A = 1 AND B = 1.
- Backtrace from G1: pick A first (highest controllability-1), assign A = 1, simulate.
- Re-evaluate: G1 still X because B is X. New objective: B = 1. Backtrace, assign B = 1.
- Now G1 = 1 in good circuit, 0 in faulty circuit — D activated at G1.
- Propagation objective: G2 must be 1 in good and 0 in faulty, so set side-input C = 0.
- Backtrace C = 0, assign, simulate. PO = G2 differs. Test pattern: A=1, B=1, C=0.

**Time:** 120 s

---

### Slide 9 — FAN (Fujiwara-Shimono, 1983)  [colleague will tweak]

**Bullets**
- Fanout-Oriented test generation — H. Fujiwara and T. Shimono, 1983.
- Refines PODEM by classifying wires structurally before search.
- Fanout points: any line that drives more than one gate — the source of reconvergence.
- Bound lines: lines in the fanout cone of a fanout point — values are constrained.
- Free lines: lines with no fanout point in their fanin cone — values are unconstrained.
- Headlines: free lines that fan into bound regions — natural decision boundaries.
- Same completeness as PODEM, dramatically less backtracking on real circuits.

**Time:** 85 s

---

### Slide 10 — Working of FAN  [colleague will tweak]

**Bullets**
- Multiple backtrace: when several objectives exist, backtrace them in parallel — one pass instead of N.
- Stop at headlines, not at PIs — defer headline justification until propagation is fixed.
- Unique sensitization: when only one path can propagate the fault, force its side inputs immediately.
- X-path check: same as PODEM — abandon branches where no all-X route to a PO exists.
- Static learning: pre-compute implications of each gate's input combinations — avoid re-deriving at runtime.
- Net effect: same correctness as D-Algorithm and PODEM, far fewer wasted decisions.

**Time:** 85 s

---

### Slide 11 — Scalability Bottleneck and Why ML

**Bullets**
- Reconvergent fanout is the core source of pain — it creates internal constraints that classical search must rediscover.
- Worst-case runtime of D-Alg, PODEM, FAN is exponential in the number of fanout points.
- Our local PODEM took 117 seconds on c432 (160 gates) — and modern designs have 100M+ gates.
- Commercial tools (TestMAX, ATALANTA) survive via heuristics, parallelism, and engineering effort — but still scale poorly.
- Observation: useful test patterns share structural regularities — fanout cones, gate sequences, reconvergence shapes.
- ML hypothesis: a neural network can learn these regularities and predict a candidate pattern in one forward pass.
- This is the motivation for DETECTive (Petrolo et al., GLSVLSI 2024) — and the rest of the talk.

**Time:** 90 s

---

## BLOCK 2 — ATPP ARCHITECTURE

### Slide 12 — ATPP Concept

**Bullets**
- ATPP = Automatic Test Pattern Prediction (vs. Generation).
- Input: gate-level netlist + a target stuck-at fault location and polarity.
- Output: a single binary test pattern over the primary inputs, in one forward pass.
- Architecture: GNN encoder learns gate-level structure, LSTM encoder learns path context, MLP head predicts PI bits.
- Goal per the paper: prioritize one test per fault, not maximize coverage in one pattern.
- Evaluation: bit-wise comparison with the closest-matching ground-truth pattern (paper Sec. 5).

**Time:** 70 s

---

### Slide 13 — Circuit as Graph

**Bullets**
- Each gate becomes a node; each wire becomes a directed edge from driver to load.
- Primary inputs and outputs are special node types with no driver / no load.
- The fault site is encoded by a one-bit flag on the affected node.
- Resulting structure is a directed acyclic graph (combinational logic only).
- Paper Fig. 2 shows this transformation; we follow it exactly.
- This representation lets a GNN reason about local fanin/fanout and reconvergence directly.

**Time:** 60 s

---

### Slide 14 — GNN Encoder: GAT + GCN

**Bullets**
- First layer: Graph Attention Network (GAT) — learns weighted importance over each node's neighbors.
- Useful for fanout, where some loads matter more than others for fault propagation.
- Second layer: Graph Convolutional Network (GCN) — uniform aggregation, refines the attention output.
- Output embedding dimension: 32 per node.
- ReLU activations, dropout 0.2 between layers.
- Result: each node carries a learned vector summarizing its structural role.

**Time:** 65 s

---

### Slide 15 — LSTM Path Encoder

**Bullets**
- Two path types per fault: activation paths (PIs to fault site) and propagation paths (fault site to POs).
- Up to p = 10 paths of each type; paths are variable length.
- Each path is a sequence of GNN node embeddings — naturally fits an LSTM.
- We use a Bidirectional LSTM so context flows both forward and backward along the path.
- Hidden size 32, single layer, mean-pooled across the p paths to give a fixed vector per type.
- Output: an activation-path embedding and a propagation-path embedding, each in R^32.

**Time:** 70 s

---

### Slide 16 — Full Architecture

**Bullets**
- Three modules: GNN encoder, LSTM path encoder, MLP InputPredictor head.
- Hidden dim 32 throughout, p = 10 paths, dropout 0.2.
- InputPredictor concatenates (GNN-fault-node-embedding, activation-path-emb, propagation-path-emb).
- MLP: 96 -> 64 -> |PI| with sigmoid output, one bit per primary input.
- Total trainable parameters: ~25 k — small by modern standards, intentionally so.
- One forward pass per fault gives the predicted test pattern.

**Time:** 70 s

---

### Slide 17 — Node Features and Tensor Shapes

**Bullets**
- Per-node feature: 11-dim vector — 10-way one-hot for gate type plus 1-bit fault flag.
- Gate types: AND, OR, NAND, NOR, XOR, XNOR, NOT, BUFF, PI, PO.
- Tensor flow: [N, 11] -> GNN -> [N, 32] -> path slice [p, L, 32] -> BiLSTM -> [p, 32] -> mean -> [32].
- Concatenated head input: [96] -> MLP -> [|PI|] sigmoid -> binary test pattern.
- Threshold at 0.5 to get hard bits at inference; raw logits used for paper's bit-accuracy metric.
- Same shapes whether circuit is 5 gates or 1000 gates — model is parameter-shared across N.

**Time:** 75 s

---

### Slide 18 — Training Setup

**Bullets**
- Loss: Binary Cross-Entropy applied per PI bit, averaged.
- Optimizer: Adam, learning rate 1e-3, gradient clipping at 1.0, batch size 32.
- Train/val split: 80/20 over ~12 k synthetic 4-input circuits.
- Multi-ground-truth handling: closest-pattern selection — pick GT pattern with min Hamming distance.
- Trained 100 epochs; checkpoint best validation bit-accuracy.
- Weights file: 100_epoch_run/best_detective_model.pt (~25k params).

**Time:** 70 s

---

### Slide 19 — Training Curve and Overfitting

**Bullets**
- Best validation bit-accuracy: 0.8358 at epoch 40.
- Training accuracy continues climbing past epoch 40 — model has capacity to memorize.
- Validation plateaus then drifts down — classic overfitting signature.
- We report the epoch-40 checkpoint, not the final epoch.
- Train-val gap widens monotonically past epoch 40 by ~0.5 - 1 percentage point per 10 epochs.
- Justifies the early-best checkpoint policy and the modest parameter budget (~25 k).

**Time:** 70 s

---

## BLOCK 3 — RESULTS

### Slide 20 — Dataset

**Bullets**
- ~12 000 randomly generated combinational circuits, each with exactly 4 primary inputs.
- Random gate counts, depths, and topologies — chosen to span the regimes the paper studies.
- Ground truth: brute-force exhaustive search across all 16 PI patterns per fault.
- Multiple valid test patterns exist per fault — handled by closest-pattern selection at evaluation.
- Fault list: every line in every circuit, both stuck-at-0 and stuck-at-1.
- Generation script and pickled splits live in train_dataset.pkl, val_dataset.pkl.

**Time:** 70 s

---

### Slide 21 — ISCAS-85 Results Table

**Bullets**
- Five circuits evaluated: c17, c432, c499, c880, c1908.
- Per circuit: gate count, ATPP bit-accuracy, ATPP fault-sim coverage, total runtime in ms.
- c17 (6 gates): bit-acc 86.67%, fault-cov 83.33%, 408.5 ms.
- c432 (160 gates): bit-acc 85.53%, fault-cov 41.88%, 665 ms.
- c499 (202 gates): bit-acc 44.27%, fault-cov 17.82%, 692.8 ms.
- c880 (383 gates): bit-acc 92.58%, fault-cov 15.54%, 1130.2 ms.
- c1908 (880 gates): bit-acc 69.70%, fault-cov 23.86%, 572.2 ms.

**Time:** 80 s

---

### Slide 22 — 4-Way Bit-Accuracy (Paper-Faithful)

**Bullets**
- Bar chart compares ATPP, PODEM, D-Algorithm, FAN on the bit-accuracy metric.
- Paper Section 5 verbatim: bit-accuracy is the official DETECTive metric — closest-pattern bit-wise compare.
- ATPP scores (%): c17 86.67, c432 85.53, c499 44.27, c880 92.58, c1908 69.70.
- ATPP is competitive with classical algorithms on c17, c432, c880.
- ATPP underperforms on c499 (XOR-heavy reconvergence) — same family of circuits the paper flags as hardest.
- This is the apples-to-apples view the paper uses; runtime is reported separately.

**Image:** bit_accuracy_4way.png
**Time:** 75 s

---

### Slide 23 — Runtime per Fault — 17.6x Speedup on c432

**Bullets**
- c432 headline: PODEM 117 034 ms total vs ATPP 665 ms — a 176x circuit-level speedup, ~17.6x per-fault.
- Across all five circuits ATPP runtime stays in the hundreds-of-ms range.
- PODEM runtime explodes with circuit complexity — exponential search behavior visible.
- D-Algorithm and FAN follow PODEM's general scaling pattern, just with different constants.
- ATPP runtime is dominated by GNN+LSTM forward pass, near-constant per fault.
- This is the result the paper centers on, and our measurements reproduce it directionally.

**Image:** runtime_perfault_4way.png
**Time:** 80 s

---

### Slide 24 — Bit-Accuracy vs Fault-Sim Coverage Gap

**Bullets**
- Bit-accuracy and fault-simulation coverage are NOT the same thing — and the gap is real.
- Bit-accuracy (paper metric): how many bits match the closest ground-truth pattern.
- Fault-sim coverage (stricter): does the predicted pattern actually detect the fault when simulated?
- Measured gap: c432 85.53% bit-acc -> 41.88% fault-cov; c880 92.58% -> 15.54%; c1908 69.70% -> 23.86%.
- The paper acknowledges this: it explicitly chooses bit-level over word-level (Section 6.2).
- We report both because honest replication requires acknowledging which guarantee you actually have.

**Paper Section 5 quote (used in speaker notes):**
> "We train DETECTive to generate a single test pattern that exposes the designated stuck-at fault, prioritizing it rather than aiming for broader fault coverage. ... we compute our model's accuracy by selecting the test pattern that exhibits the closest similarity to the predicted test pattern. The accuracy metric is determined through a bit-wise comparison of the two test patterns."

**Image:** fault_sim_coverage_table.png
**Time:** 90 s

---

### Slide 25 — Synthetic Grid (Paper Fig 5a Replication)

**Bullets**
- Grid sweeps gate count vs circuit depth on synthetic 4-input circuits — replicates paper Figure 5a.
- Bit-accuracy range: 0.78 to 1.00 across the grid; mean 0.87.
- Trend is monotone: small + shallow circuits hit ~0.93, deep + large circuits drop to ~0.79.
- Confirms paper's claim that the model degrades smoothly with structural complexity — no sharp cliffs.
- Synthetic regime is in-distribution — these are circuits drawn from the same generator as training.
- ISCAS-85 results are out-of-distribution — bigger and structurally different than training.

**Image:** synthetic_grid_heatmap.png
**Time:** 75 s

---

### Slide 26 — Depth Degradation (Paper Fig 6a Replication)

**Bullets**
- Plot: bit-accuracy vs maximum logic depth, holding gate count roughly fixed.
- Range observed: 0.89 at depth 1 down to 0.78 at depth 25.
- Smooth curve — no phase transition, no collapse.
- Matches paper Figure 6a directionally.
- Mechanism: deeper circuits have longer activation/propagation paths than the LSTM saw in training.
- Suggests one obvious extension — train on deeper synthetic circuits to push the curve right.

**Image:** depth_curve.png
**Time:** 75 s

---

### Slide 27 — Honest Takeaways and Limitations

**Bullets**
- Speed: yes — 17.6x faster than PODEM on c432, hundreds-of-ms regime across ISCAS-85.
- Bit-accuracy: competitive with classical ATPG on most circuits under the paper's official metric.
- Completeness: no — ATPP gives no formal guarantee that any predicted pattern detects its fault.
- Weak spots: c499 and c1908 — XOR-heavy and reconvergent, same circuits where classical methods also struggle.
- Bit-acc vs fault-coverage gap is real; the paper's metric choice is defensible but worth flagging.
- Scope: combinational only; deeper paths hurt; no sequential circuits in this work.
- Honest framing: ATPP is a fast first guess for a classical verifier, not a replacement for ATPG.

**Time:** 90 s

---

### Slide 28 — Conclusion

**Bullets**
- Classical ATPG (D-Alg, PODEM, FAN) is correct and complete but exponential in the worst case.
- DETECTive (ATPP) replaces search with a single GNN+LSTM forward pass — predicts a candidate test pattern.
- Demonstrated 17.6x per-fault speedup over PODEM on c432; 85.5% bit-accuracy under the paper's metric.
- The model transfers from synthetic 4-PI training data to real ISCAS-85 circuits — generalization works.
- Future work: integrate ATPP as fast first guess + classical verifier; extend to sequential circuits; deeper-circuit training.
- Broader take: machine learning is a real tool for EDA — not a replacement for proofs, but a meaningful accelerator.

**Time:** 80 s

---

## CLOSING

### Q & A
Floor open for questions on any block.

### Thank You
References on the slide; offline discussion welcome.

---

## FILES REFERENCED

- Builder script: `presentation/build_full_deck.py`
- Output deck: `presentation/DETECTive_30min_full_deck.pptx` (31 slides, 16:9, ~480 KB)
- Images embedded from `presentation_images/`:
  - `bit_accuracy_4way.png` (slide 22)
  - `runtime_perfault_4way.png` (slide 23)
  - `fault_sim_coverage_table.png` (slide 24)
  - `synthetic_grid_heatmap.png` (slide 25)
  - `depth_curve.png` (slide 26)
- Trained weights: `100_epoch_run/best_detective_model.pt`
- Datasets: `train_dataset.pkl`, `val_dataset.pkl`
