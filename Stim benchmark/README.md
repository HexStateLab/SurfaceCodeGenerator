# Bench — STIM Circuit-Level Benchmark for plane_warp

`bench.py` is a comprehensive fault-tolerance benchmark for the plane_warp
decoder on the 2D BB code. It samples 5-round error-correction circuits
under five noise models, runs the decoder on each shot, and reports logical
error rates with 95% Wilson confidence intervals.

## Quick Start

```bash
gcc -std=gnu11 -O3 -o plane_warp plane_warp.c -lm
python3 bench.py
```

Expected output: a table for each circuit type showing baseline LER, decoded
LER, percentage reduction, and a fault-tolerance classification ("corrects",
"=baseline", or "WORSE").

## Circuit Types

All circuits use an R×S torus with N = R×S data qubits, N ancilla qubits per
round, and 5 measurement rounds. The logical observable is the parity of Z-basis
data measurements at even column indices q ∈ {0, 2, 4, …}.

| `ctype` | Gate | Description |
|---------|------|-------------|
| `cz` | CZ | Standard CZ-based ZZZZ stabilizer extraction. Ancilla in \|+⟩, CZ(anc,q) for each data qubit, H on anc, measure. |
| `cn` | CNOT | CNOT-based ZZZZ extraction. **Ancilla in \|0⟩, CNOT(q,anc) for each data qubit, measure directly.** No Hadamard gates. Identical syndrome structure to CZ; only the gate noise profile differs. |
| `phenom` | CZ | Phenomenological noise: X_ERROR on data each round + X_ERROR on measurement outcomes. No gate noise (no DEPOLARIZE2). |
| `correlated` | CZ | Like `cz`, plus random DEPOLARIZE2 between pairs of data qubits within each check, at 2×p_g correlation rate. |
| `asymmetric` | CZ | One parity sector (px=0, py=0) has 10× the gate error rate. Models spatially non-uniform noise. |

### Why the CNOT Circuit Works

Early versions used `CNOT(anc, q)` with ancilla in \|+⟩, which measures
X₁X₂X₃X₄ — an X-check that detects Z errors, invisible to the Z-basis
observable. The decoder found Z-error patterns, applied them as X flips,
and introduced new errors (LER 2% → 45%).

The fix swaps control and target: `CNOT(q, anc)` with ancilla prepared in
\|0⟩ (R gate only, no H). Each CNOT XORs the data qubit's Z value onto the
ancilla. After all four CNOTs, the ancilla holds the parity d₁⊕d₂⊕d₃⊕d₄.
Direct Z-basis measurement gives the ZZZZ outcome — identical to CZ-based
extraction, so the existing `--decode` path works without modification.

## Noise Model

Per round, every circuit applies in order:

| Step | Error | Rate | Applies to |
|------|-------|------|------------|
| Idle | `DEPOLARIZE1` | p_g / 10 | All data qubits |
| Gate (×4 per check) | `DEPOLARIZE2` | p_g (lognormal, σ=0.2) | Each ancilla–data pair after gate |
| Ancilla idle | `DEPOLARIZE1` | p_g / 10 | All ancilla qubits |
| Readout | `X_ERROR` | p_meas | All measurement outcomes |

After the final round, data qubits are measured in the Z basis. The
per-shot observable parity `ov` is computed from these measurements. A
logical error occurs when `ov ⊕ dp = 1`, where `dp` is the decoder's
predicted parity from its output correction.

## Decoder Settings Panel

Each circuit is tested under a panel of decoder settings, all run on the
**same set of shots** for paired comparison:

| Setting | Flags | Description |
|---------|-------|-------------|
| `plain` | `--decode` | Standard pipeline: product-code preprocessor + per-sector exact solver + 10-pass residual loop. |
| `cap-auto R` | `--cap-auto R --decode` | Same pipeline, but abstains (returns empty correction) when the found correction weight exceeds ceil(R×n + 2√(R×n×(1−R))) — the ~2σ upper bound for plausible data errors at rate R. Mitigates untrustworthy syndromes. |
| `spacetime` | `--st` | Spacetime decoder using all 5 rounds of syndrome data with temporal bias. |

## Output Format

For each circuit config, `bench.py` prints a block:

```
── CZ-based (basis-matched) ──  grid 6×6 (n=36), p_g=5.0e-04, p_meas=1.0e-03
setting                          dec_LER   [95% Wilson CI]   %reduction  baseline  [CI]            FT
plain (--decode)                 0.00350   [0.00169,0.00718]    76.7%    0.01500  [0.01045,0.02138] corrects
cap-auto 0.015 (cap=3)           0.01000   [0.00661,0.01508]    17.3%    0.01500  [0.01045,0.02138] =baseline
cap-auto 0.030 (cap=5)           0.00600   [0.00344,0.01040]    60.0%    0.01500  [0.01045,0.02138] corrects
```

- **dec_LER**: decoded logical error rate
- **95% Wilson CI**: Wilson score interval (not normal approximation — valid near 0 and 1)
- **%reduction**: `(baseline − decoded) / baseline × 100`
- **baseline**: LER with no correction applied (raw physical errors)
- **FT** (fault tolerance): `corrects` = decoder strictly beats baseline (CI intervals disjoint), `=baseline` = CIs overlap, `WORSE` = decoder strictly hurts

## Configurations

Defined in the `CONFIGS` list near the top of `bench.py`:

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

## Results — 6×6

p_g = 0.0005 (0.05%), p_meas = 0.001 (0.1%), 2000 shots per config, seed=2024.

| Circuit | Baseline LER | Decoded LER |
|---------|-------------|-------------|
| CZ | 1.50% | **0.35%** |
| Phenom | 1.10% | **0.00%** |
| Asymmetric | 13.30% | **4.70%** |
| CNOT | 1.50% | **0.35%** |

All four circuit types achieve fault tolerance under the standard `--decode`
pipeline. CNOT performance matches CZ identically — both circuits measure the
same Z-check syndrome type; only the gate set (CZ vs CNOT) and its associated
DEPOLARIZE2 noise profile differ.

## Per-Shot Limits

The decoder is invoked via subprocess for each shot with a 180-second
timeout. The syndrome is piped via stdin (`n` bytes), and the correction is
read from stdout (`n` bytes). For spacetime decoding, all shots are batched
into a single call with a binary header.

## Requirements

- Python 3 with `numpy` and `stim`
- Compiled `plane_warp` binary in the same directory

```bash
pip install numpy stim
```

## See Also

- `README.md` — project overview and decoder architecture
- `README_plane_warp.md` — detailed algorithm documentation
- `plane_warp.c` — decoder source (single-file, gcc)
