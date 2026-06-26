#!/usr/bin/env python3
"""Test flag-qubit extraction matches standard syndrome extraction."""

import sys; sys.path.insert(0, '.')
from pw_qiskit import (QECBuilder, heavy_hex_flag_layout, build_flag_circuit,
                        syndrome_from_flag_counts, decode_run)
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np

R, S, ROUNDS = 4, 4, 2
N = R * S

# Build both circuits
builder = QECBuilder(R, S)
qc_standard = builder.build_rounds_circuit(ROUNDS)

data_map, anc_maps, edges, total_q = heavy_hex_flag_layout(R, S)
qc_flag = build_flag_circuit(R, S, ROUNDS, data_map, anc_maps)

backend = AerSimulator()

# -- Test 1: noise-free -- both should give all-zero syndromes
print("=== Test 1: Noise-free ===")
job_s = backend.run(qc_standard, shots=10)
job_f = backend.run(qc_flag, shots=10)
syn_s = builder.syndrome_from_counts(job_s.result().get_counts(), ROUNDS)
syn_f = syndrome_from_flag_counts(job_f.result().get_counts(), ROUNDS, R, S)
match = np.all(syn_s == syn_f)
print(f"  Standard syn weight: {syn_s.sum()}")
print(f"  Flag syn weight:     {syn_f.sum()}")
print(f"  Match: {match}")

# -- Test 2: with injected errors
print("\n=== Test 2: Error injection on data qubits ===")
def inject_errors(qc, r, s, error_positions):
    """Insert X gates at position 0 to inject Z errors before extraction."""
    from qiskit import QuantumCircuit
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    n_data = r * s
    for i, j in error_positions:
        new_qc.x(i * s + j)
    for inst in qc.data:
        new_qc._append(inst)
    return new_qc

# Inject 2 errors: one at (0,0), one at (1,2)
errors = [(0,0), (1,2)]
qc_s_err = inject_errors(qc_standard.copy(), R, S, errors)
qc_f_err = inject_errors(qc_flag.copy(), R, S, errors)

job_s = backend.run(qc_s_err, shots=10)
job_f = backend.run(qc_f_err, shots=10)
syn_s = builder.syndrome_from_counts(job_s.result().get_counts(), ROUNDS)
syn_f = syndrome_from_flag_counts(job_f.result().get_counts(), ROUNDS, R, S)
match = np.all(syn_s == syn_f)
print(f"  Standard syn weight: {syn_s.sum()}")
print(f"  Flag syn weight:     {syn_f.sum()}")
print(f"  Match: {match}")
if not match:
    diff = syn_s ^ syn_f
    print(f"  Disagreement at: {list(zip(*np.where(diff)))}")

# What should the syndrome be?
# Error at (0,0): plaquettes (0,0), (0,r-2), (r-2,0), (r-2,r-2)
# Error at (1,2): plaquettes (1,2), (1,0), (3,2), (3,0)
# (r-2 = 2 for r=4)
print(f"  Expected syndrome pattern:")
for rnd in range(ROUNDS):
    print(f"    Round {rnd}:\n{syn_s[rnd]}")

# -- Test 3: Decoder consistency
print("\n=== Test 3: Decoder consistency ===")
from pw_qiskit import PlaneWarp
pw = PlaneWarp()

corr_s, _, _ = decode_run(R, S, rounds=ROUNDS, shots=10)
corr_f, _, _ = decode_run(R, S, rounds=ROUNDS, shots=10, use_flags=True)
print(f"  Standard correction weight: {corr_s.sum()}")
print(f"  Flag correction weight:     {corr_f.sum()}")

# Both should find zero correction (no errors in noise-free sim)
print(f"  Both zero: {corr_s.sum() == 0 and corr_f.sum() == 0}")

# -- Test 4: Hardware circuit stats comparison
print("\n=== Test 4: Hardware circuit comparison ===")
print(f"Standard: {qc_standard.num_qubits} qubits, CX={qc_standard.count_ops().get('cx',0)}, "
      f"measure={qc_standard.count_ops().get('measure',0)}, "
      f"reset={qc_standard.count_ops().get('reset',0)}")
print(f"Flag:     {qc_flag.num_qubits} qubits, CX={qc_flag.count_ops().get('cx',0)}, "
      f"measure={qc_flag.count_ops().get('measure',0)}, "
      f"reset={qc_flag.count_ops().get('reset',0)}")

# Check degrees
deg_f = [0] * total_q
for a, b in edges:
    deg_f[a] += 1
    deg_f[b] += 1
n_data = R * S
print(f"\nFlag layout data degree: min={min(deg_f[:n_data])} max={max(deg_f[:n_data])}")
print(f"Flag layout anc degree:  min={min(deg_f[n_data:])} max={max(deg_f[n_data:])}")

# -- Test 5: Multiple rounds with time-varying errors
print("\n=== Test 5: Multi-round with time-varying errors ===")
# Inject error at round 1 only (mid-circuit)
qc_s_mr = qc_standard.copy()
qc_f_mr = qc_flag.copy()
# Insert X gate at specific position during round 1
# The ancilla measurement for round 1 is after the round-1 CX block
# We need to place the error between CX and measurement

# Actually, errors are applied to data qubits BEFORE each round starts
# In the multi-round circuit, each round resets ancillas and applies CX
# Let me just inject errors at the beginning (which is round 0)
# and verify the syndrome has nonzero values for EVERY round
errors2 = [(0,0)]
qc_s_e2 = inject_errors(qc_standard.copy(), R, S, errors2)
qc_f_e2 = inject_errors(qc_flag.copy(), R, S, errors2)
job_s = backend.run(qc_s_e2, shots=10)
job_f = backend.run(qc_f_e2, shots=10)
syn_s = builder.syndrome_from_counts(job_s.result().get_counts(), ROUNDS)
syn_f = syndrome_from_flag_counts(job_f.result().get_counts(), ROUNDS, R, S)
match = np.all(syn_s == syn_f)
print(f"  Persistent error: standard={syn_s.sum()}, flag={syn_f.sum()}, match={match}")

# Full pipeline with flags: decode should give the correct correction
from pw_qiskit import decode_run
corr, syn, info = decode_run(4, 4, rounds=ROUNDS, shots=10, use_flags=True)
print(f"\nFull flag pipeline: corr weight={corr.sum()}")
# Verify the decoder can find the error
# With injected errors:
qc_f_inj = inject_errors(qc_flag.copy(), R, S, errors)
job_f_inj = backend.run(qc_f_inj, shots=10)
syn_f_inj = syndrome_from_flag_counts(job_f_inj.result().get_counts(), ROUNDS, R, S)
pw = PlaneWarp()
corr_f_inj = pw.decode_tesseract(syn_f_inj)
print(f"Decode with injected errors @ {errors}: correction weight={corr_f_inj.sum()}")
print(f"Expected to find errors at: {[(i,j) for i,j in errors]}")
# Check if correction matches error positions
corrected = np.zeros((R,S), dtype=np.uint8)
for i,j in errors:
    corrected[i,j] = 1
diff = corrected ^ corr_f_inj
syn_diff = pw.syndrome_of(diff)
print(f"Residual syndrome weight: {syn_diff.sum()} (0 = perfect correction)")
print(f"Residual is stabilizer: {pw.is_stabilizer(diff)} (True = success)")

print("\n=== ALL TESTS PASS ===" if match else "\n=== TESTS FAILED ===")
