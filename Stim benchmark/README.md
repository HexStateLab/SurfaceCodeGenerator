# plane_warp — STIM Circuit-Level Bench

Fault-tolerance benchmark for the plane_warp decoder on the 2D BB code.
Runs 5-round error-correction circuits under five noise models, decodes
every shot, and reports logical error rates with 95% Wilson CIs.

## Quick Start

```bash
gcc -std=gnu11 -O3 -o plane_warp plane_warp.c -lm
python3 bench.py
```

## Circuit Types

Every circuit uses an R×S torus of physical data qubits with N = R×S
ancilla qubits per round, 5 measurement rounds, and a Z-basis logical
observable at even column indices.

| `ctype` | Gate | Ancilla prep | Measures |
|---------|------|-------------|----------|
| `cz` | CZ(anc, q) | \|+⟩ → CZ → H → M | Z₁Z₂Z₃Z₄ |
| `cn` | CNOT(q, anc) | \|0⟩ → CNOT → M | Z₁Z₂Z₃Z₄ |
| `phenom` | CZ(anc, q) | same as cz | Z₁Z₂Z₃Z₄ |
| `correlated` | CZ(anc, q) | same as cz | Z₁Z₂Z₃Z₄ |
| `asymmetric` | CZ(anc, q) | same as cz | Z₁Z₂Z₃Z₄ |

- **cz** — Standard CZ-based stabiliser extraction.
- **cn** — CNOT-based extraction using CNOT(data, anc) with ancilla
  prepared in \|0⟩ (no Hadamard). Each CNOT XORs the data qubit's Z value
  onto the ancilla, which accumulates the parity. Direct Z-basis
  measurement gives the same Z₁Z₂Z₃Z₄ outcome as CZ, differing only in
  the gate-noise profile.
- **phenom** — Phenomenological noise: data X errors each round +
  measurement flips. No gate noise (DEPOLARIZE2 omitted).
- **correlated** — Like cz, plus random DEPOLARIZE2 between pairs of
  data qubits within each check at 2×p_g correlation rate.
- **asymmetric** — One parity sector (px=0, py=0) has 10× the gate error
  rate. Models spatially non-uniform noise.

## Noise Model (per round)

| Source | Operation | Rate | Target |
|--------|-----------|------|--------|
| Idle | DEPOLARIZE1 | p_g / 10 | All data qubits |
| Gate (×4 per check) | DEPOLARIZE2 | p_g (lognormal, σ=0.2) | anc–data pair |
| Anc idle | DEPOLARIZE1 | p_g / 10 | All ancillae |
| Readout | X_ERROR | p_meas | Measurement outcomes |

After the final round, data qubits are measured in the Z basis. The
per-shot observable parity is computed from these measurements. A logical
error occurs when the observed parity differs from the decoder's
predicted parity.

## Decoder Settings Panel

Each circuit is tested under a panel of decoder settings, all on the
*same set of shots* for paired comparison.

| Setting | Flags | Description |
|---------|-------|-------------|
| `plain` | `--decode` | 5-pass residual loop with product-code preprocessor + per-sector exact solver |
| `cap-auto R` | `--cap-auto R --decode` | Same pipeline, abstains when correction exceeds ceil(R·n + 2√(R·n·(1−R))) |

## Output

For each circuit config, `bench.py` prints a block:

```
── CZ-based (basis-matched) ──  grid 6×6 (n=36), p_g=5.0e-04, p_meas=1e-03, 5 rounds, 2000 shots
     baseline (no correction)         1.50%  [ 1.05,  2.13]
  *  plain                            0.40%  [ 0.20,  0.79]  corrects
     cap-auto 0.015 (cap=2)           0.45%  [ 0.24,  0.85]  corrects
```

- **baseline** — logical error rate with no correction applied
- **plain** — decoder LER
- **95% CI** — Wilson score interval (valid near 0 and 1)
- **FT mark** — `corrects` = decoder strictly beats baseline,
  `=baseline` = CIs overlap, `WORSE` = decoder strictly hurts
- `*` marks the best setting

## Results

### 6×6 — 36 data qubits, distance 3, 20 logical qubits

p_g = 0.05%, p_meas = 0.1%, 2000 shots (1000 for asymmetric/CNOT).

| Circuit | Baseline LER | Decoded LER |
|---------|-------------|-------------|
| CZ | 1.50% | **0.40%** |
| Phenom | 1.10% | **0.00%** |
| Correlated | 2.15% | **0.35%** |
| Asymmetric (10× hot) | 13.30% | **5.80%** |
| CNOT | 1.40% | **0.20%** |

### 20×20 — 400 data qubits, distance 10, 76 logical qubits

p_g = 0.04% (CZ) / 0.02% (CNOT), p_meas = 0.1%, 300 shots.

| Circuit | Baseline LER | Decoded LER |
|---------|-------------|-------------|
| CZ | 2.67% | **0.00%** |
| CNOT | 2.33% | **0.67%** |

## Decoder Architecture

The decoder exploits the BB code's parity-sector structure: the +2 offsets
in the check polynomial `a(x,y) = (x²+1)(y²+1)` never mix parity classes,
splitting the r×s problem into 4 independent (r/2)×(s/2) unit-plaquette
toric codes.

Per sector, the (1+x)(1+y) check matrix is inverted via forward-pass
prefix XOR (the discrete 2D integral over GF(2)), then the hr+hs−1
dimensional kernel (column-flip and row-flip operators) is swept to
find the minimum-weight correction. Two boundary seeds per sector
(E[0][0] = 0 or 1) × 4 sectors = 8 total starting points.

A product-code preprocessor handles measurement errors by detecting and
correcting odd row/column parity violations in each parity sector using a
2×2 block scan (catches DEPOLARIZE2 ancilla–data correlations) followed
by iterative edge-flip matching (pairs odd rows/columns only at positions
where the syndrome bit is already 1 — genuine measurement-error sites).

The full pipeline:

1. **Preprocessor** — 2×2 scan for correlated events + edge-flip for
   standalone measurement errors. Iterates to convergence.
2. **ML solver** — per-sector algebraic exact solution.
3. **5-pass residual loop** — each pass decodes, accumulates the
   correction, computes the residual syndrome, and re-decodes.

## Configurations

Defined at the top of `bench.py`:

```python
CONFIGS = [
    (6,  6,  0.0005, 0.001, "cz",         2000, "CZ-based  (basis-matched)"),
    (6,  6,  0.0008, 0.001, "phenom",     2000, "phenomenological (data+meas)"),
    (6,  6,  0.0005, 0.001, "correlated", 2000, "correlated-pair"),
    (6,  6,  0.0005, 0.001, "asymmetric", 1000, "asymmetric (10x hot sub)"),
    (6,  6,  0.0005, 0.001, "cn",         1000, "CNOT-based (Z-check, basis-matched)"),
    (20, 20, 0.0004, 0.001, "cz",          300, "CZ-based  (basis-matched)"),
    (20, 20, 0.0002, 0.001, "cn",          300, "CNOT-based (Z-check, basis-matched)"),
]
```

Format: `(R, S, p_g, p_meas, ctype, shots, label)`.

## Requirements

- Python 3 with `numpy` and `stim`
- Compiled `plane_warp` binary in the same directory

```bash
pip install numpy stim
```
