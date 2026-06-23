# plane_warp Decoder Benchmarks

**Noise model:** STIM circuit-level noise  
**Rounds:** 5  
**Metrics:** Logical Error Rate (LER) with Wilson 95% confidence intervals.

- **plain** = unconstrained decoder
- **cap-auto** = adaptive correction-weight cap
- `*` = best setting for that circuit

---

## Overview

| Circuit | Grid | Baseline | Best | Result |
|--------|------|---------:|-----:|--------|
| CZ-based (basis matched) | 6×6 | 1.65% | **0.20%** | ✅ Corrects |
| Phenomenological | 6×6 | 1.10% | **0.00%** | ✅ Corrects |
| Correlated pair | 6×6 | 1.45% | **0.20%** | ✅ Corrects |
| Asymmetric (10× hot sublattice) | 6×6 | 13.00% | **4.30%** | ✅ Corrects |
| CNOT-based (basis mismatched) | 6×6 | 1.90% | **2.00%** | ⚠️ Baseline |
| CZ-based (basis matched) | 20×20 | 3.67% | **1.67%** | ≈ Baseline |
| CNOT-based (basis mismatched) | 20×20 | 2.33% | **2.33%** | ≈ Baseline |

---

# Detailed Results

<details>
<summary><b>CZ-based (basis matched), 6×6</b></summary>

```
n        = 36
p_g      = 5.0e-04
p_meas   = 1e-03
rounds   = 5
shots    = 2000

baseline (no correction)      1.65%  [1.18, 2.31]

* plain                       0.20%  [0.08, 0.51]   corrects
  cap-auto 0.015 (cap=2)      0.25%  [0.11, 0.58]   corrects
  cap-auto 0.030 (cap=3)      0.20%  [0.08, 0.51]   corrects
```

</details>

---

<details>
<summary><b>Phenomenological (data + measurement), 6×6</b></summary>

```
n        = 36
p_g      = 8.0e-04
p_meas   = 1e-03
rounds   = 5
shots    = 2000

baseline (no correction)      1.10%  [0.73, 1.66]

* plain                       0.00%  [0.00, 0.19]   corrects
  cap-auto 0.015 (cap=2)      0.00%  [0.00, 0.19]   corrects
  cap-auto 0.030 (cap=3)      0.00%  [0.00, 0.19]   corrects
```

</details>

---

<details>
<summary><b>Correlated-pair noise, 6×6</b></summary>

```
n        = 36
p_g      = 5.0e-04
p_meas   = 1e-03
rounds   = 5
shots    = 2000

baseline (no correction)      1.45%  [1.01, 2.07]

* plain                       0.20%  [0.08, 0.51]   corrects
  cap-auto 0.015 (cap=2)      0.20%  [0.08, 0.51]   corrects
  cap-auto 0.030 (cap=3)      0.20%  [0.08, 0.51]   corrects
```

</details>

---

<details>
<summary><b>Asymmetric noise (10× hot sublattice), 6×6</b></summary>

```
n        = 36
p_g      = 5.0e-04
p_meas   = 1e-03
rounds   = 5
shots    = 1000

baseline (no correction)     13.00%  [11.06, 15.23]

* plain                       4.30%  [3.21, 5.74]   corrects
  cap-auto 0.015 (cap=2)      4.60%  [3.47, 6.08]   corrects
  cap-auto 0.030 (cap=3)      4.30%  [3.21, 5.74]   corrects
```

</details>

---

<details>
<summary><b>CNOT-based (basis mismatched), 6×6</b></summary>

```
n        = 36
p_g      = 5.0e-04
p_meas   = 1e-03
rounds   = 5
shots    = 1000

baseline (no correction)      1.90%  [1.22, 2.95]

plain                        44.10%  [41.05, 47.19]  WORSE

* cap-auto 0.015 (cap=2)      2.00%  [1.30, 3.07]   ≈ baseline
  cap-auto 0.030 (cap=3)      2.90%  [2.03, 4.13]   ≈ baseline
```

</details>

---

<details>
<summary><b>CZ-based (basis matched), 20×20</b></summary>

```
n        = 400
p_g      = 4.0e-04
p_meas   = 1e-03
rounds   = 5
shots    = 300

baseline (no correction)      3.67%  [2.06, 6.45]

* plain                       1.67%  [0.71, 3.84]   ≈ baseline
  cap-auto 0.015 (cap=11)     1.67%  [0.71, 3.84]   ≈ baseline
  cap-auto 0.030 (cap=19)     1.67%  [0.71, 3.84]   ≈ baseline
```

</details>

---

<details>
<summary><b>CNOT-based (basis mismatched), 20×20</b></summary>

```
n        = 400
p_g      = 2.0e-04
p_meas   = 1e-03
rounds   = 5
shots    = 300

baseline (no correction)      2.33%  [1.13, 4.74]

plain                        53.33%  [47.68, 58.90]  WORSE

* cap-auto 0.015 (cap=11)     2.33%  [1.13, 4.74]   ≈ baseline
  cap-auto 0.030 (cap=19)     2.33%  [1.13, 4.74]   ≈ baseline
```

</details>

---

# Interpretation

### Honest syndromes → correction

For **basis-matched** or otherwise trustworthy syndromes:

- CZ circuits
- Phenomenological noise
- Correlated-pair noise
- Strongly asymmetric noise

the unconstrained decoder (`plain`) consistently reduces logical error below baseline.

The adaptive cap (`cap-auto`) provides little benefit here and can slightly reduce performance by abstaining on corrections that are actually valid.

---

### Dishonest syndromes → damage control

For **basis-mismatched** CNOT circuits:

- `plain` decoding catastrophically over-corrects
- Logical error rises to **44–53%**
- `cap-auto` detects implausibly heavy corrections and abstains

The result is recovery to approximately baseline performance:

> A bad syndrome cannot be converted into correction.  
> The cap is therefore a **safety mechanism**, not a rescue decoder.

---

# Takeaways

- ✅ `plane_warp` corrects whenever the syndrome is physically consistent.
- ✅ The adaptive cap prevents catastrophic failure on inconsistent syndromes.
- ⚠️ The cap should be enabled conditionally, using a syndrome trust metric.
- ⚠️ These tests are **single-frame** benchmarks; measurement-dominated noise is better evaluated with full spacetime decoding over the syndrome history.
- 🚧 A detector-based multi-round benchmark remains future work.
