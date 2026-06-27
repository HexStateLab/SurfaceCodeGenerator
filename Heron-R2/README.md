# Two-Logical-Qubit Entanglement on the (1+x²)(1+y²) Toric BB Code

Hardware demonstration of two entangled logical qubits encoded in the
(1+x²)(1+y²) bivariate bicycle code on IBM Quantum's Heron processors.

## Results

| Experiment | Z-basis fidelity | X-basis fidelity | Witness |
|-----------|-----------------|-----------------|---------|
| |00⟩ preservation (1 round) | 64.3% | — | — |
| Bell state (0 rounds) | **73.1%** | **66.3%** | **~1.2** |
| Bell state (1 round) | 50.8% | 48.4% | ~0.0 |

**Entanglement confirmed at 0 rounds** — both subspaces |Φ⁺⟩ and |Φ⁻⟩
show conditional correlation well above the separable threshold of 1.
At 1 QEC round the 192 syndrome CX gates (206 total) introduce enough
noise to destroy the correlation; the code's distance needs cleaner
hardware (~1% CX) to show net protection.

## Logical Operators

For the 6×8 (r×s) grid with plus-shaped stabilizers:

| Operator | Support | Weight | Code |
|---------|---------|--------|------|
| Z_L1 | Row 0 | s | Parity of Z on all qubits in row 0 |
| X_L1 | Column 0 | r | Parity of X on all qubits in column 0 |
| Z_L2 | Column 0 | r | Parity of Z on all qubits in column 0 |
| X_L2 | Row 0 | s | Parity of X on all qubits in row 0 |

Row and column loop operators are independent logical cycles of the toric
code. The single-qubit intersection at (0,0) gives X_L1↔Z_L1 and
X_L2↔Z_L2 anticommutation.

## Bell State Preparation

```
|0⟩_anc − H −⊕−⊕−⊕−⊕−⊕−⊕−⊕−⊕−⊕−⊕−⊕−⊕−⊕− H − M
          | | | | | | | | | | | | | |
          c0 c1 c2 c3 c4 c5 r0 r1 r2 r3 r4 r5 r6 r7
```

Controlled-X on **both** column 0 (X_L1, 6 CX) and row 0 (X_L2, 8 CX)
with overlap at (0,0) cancelling (X·X = I). Ancilla outcome |0⟩ → |Φ⁺⟩,
|1⟩ → |Φ⁻⟩.

## Decoder

Two decoders used, both implementing the persistent-basis W-axis algorithm
with `min_weight_kernel` global minimum-weight search:

- **tesseract** — ctypes wrapper of C library (`libplane_warp.so`)
- **waxis** — pure Python implementation (`waxis_decode.py`)

Both converge to near-identical results, validating the Python port.

## Key Files

| File | Purpose |
|------|---------|
| `test_entanglement.py` | Build, submit, decode, analyze circuits |
| `retrieve_results.py` | Offline re-decoding from cached raw data |
| `waxis_decode.py` | Python W-axis decoder with kernel enumeration |
| `pw_qiskit.py` | Qiskit integration, layout, ctypes wrapper |
| `deploy_heron.py` | Job submission helpers |

## Usage

```bash
# Note
--share-pairs can be used as a command too.. I suggest it, actually as it cuts down resources needed.

# Baseline |00⟩ preservation
python3 test_entanglement.py --state 00 --rounds 1 --shots 2000

# Bell state entanglement test
python3 test_entanglement.py --state bell --rounds 0 --shots 2000
python3 test_entanglement.py --state bell --rounds 0 --measure-x --shots 2000

# 1 QEC round (tests decoder)
python3 test_entanglement.py --state bell --rounds 1 --shots 2000

# Re-decode from cache without IBM contact
python3 retrieve_results.py --from-cache
```

## Asymmetric Error Rates

Z_L1 (row 0, weight 8) and X_L1 (column 0, weight 6) have different
physical weights, causing asymmetric logical error rates:

| Observable | Weight | Flip rate (0 rounds) |
|-----------|--------|---------------------|
| Z_L1 | 8 | ~23% |
| Z_L2 | 6 | ~7% |
| X_L1 | 6 | ~23% |
| X_L2 | 8 | ~7% |

The asymmetry swaps between Z and X bases: the lighter logical operator
in each basis is more robust. An 8×8 grid would balance the weights.

## Hardware

| Backend | CX error | Qubits | Used |
|---------|----------|--------|------|
| ibm_fez | ~1.5-2% | 156 | 145 (Bell) / 144 (|00⟩) |
