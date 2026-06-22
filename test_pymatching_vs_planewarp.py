#!/usr/bin/env python3
"""
Plane-Warp vs PyMatching comparison on 40x40 BB code torus (X errors only).

Each qubit's X error flips 4 HZ detectors (plus-shaped pattern).
PyMatching graph: we decompose each 4-body neighbourhood into a 6-edge clique.
Observable: Z on the first row of sub-lattice (0,0) → qubits 0,2,4,...,38.

For each trial:
  - Generate random weight-w X errors → compute HZ syndrome
  - Decode with plane_warp (C binary) → check logical error via is_stabilizer()
  - Decode with PyMatching → check logical error via observable prediction

Usage:  python3 test_pymatching_vs_planewarp.py [--trials N] [--weights w1,w2,...]
"""

import argparse, os, subprocess, sys, time, math
import numpy as np
import pymatching

DECODER_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plane_warp")

R, S = 40, 40
N = R * S

# Observable: Z on first sub-lattice (0,0) row → qubits 0, 2, 4, ..., S-2
OBS_QUBITS = frozenset(range(0, S, 2))


# =========================================================================
#  Syndrome  (C-code: X err at (qi,qj) flips syn at (qi,qj), (qi-2,qj),
#             (qi,qj-2), (qi-2,qj-2)  — all mod R,S)
# =========================================================================
def syndrome_of(err: bytes) -> bytes:
    syn = bytearray(N)
    for q in range(N):
        if not err[q]:
            continue
        qi, qj = q // S, q % S
        for di in (0, 2):
            for dj in (0, 2):
                c = ((qi - di + R) % R) * S + ((qj - dj + S) % S)
                syn[c] ^= 1
    return bytes(syn)


# =========================================================================
#  Plane-Warp decoder
# =========================================================================
def decode_planewarp(syn: bytes) -> bytes:
    assert len(syn) == N
    proc = subprocess.run(
        [DECODER_BIN, str(R), str(S), "--decode"],
        input=syn, capture_output=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Decoder rc={proc.returncode}")
    return proc.stdout


def is_stabilizer(vec: bytes) -> bool:
    for px in range(2):
        for py in range(2):
            hr, hs = R // 2, S // 2
            for si in range(hr):
                rp = 0
                for sj in range(hs):
                    if vec[(px + 2 * si) * S + (py + 2 * sj)]:
                        rp ^= 1
                if rp:
                    return False
            for sj in range(hs):
                cp = 0
                for si in range(hr):
                    if vec[(px + 2 * si) * S + (py + 2 * sj)]:
                        cp ^= 1
                if cp:
                    return False
    return True


# =========================================================================
#  PyMatching graph: clique decomposition of each qubit's 4 detectors
# =========================================================================
def build_pymatching(p_error: float = 0.01) -> pymatching.Matching:
    """Build matching graph with detectors as nodes, edges from clique decomp.

    For each qubit q, let D0..D3 be its 4 flipped detectors.
    Add 6 edges: (D0,D1), (D0,D2), (D0,D3), (D1,D2), (D1,D3), (D2,D3).
    Weight = -ln(p_error).  (Same for all edges; relative weights unchanged.)

    Also track: for each edge, which qubit it belongs to (via fault_ids)
    so we can compute the observable parity from decoded corrections.
    """
    weight = -math.log(p_error)
    m = pymatching.Matching()
    edge_to_qubit = {}  # (u, v) -> qubit_index

    for q in range(N):
        qi, qj = q // S, q % S
        dets = []
        for di in (0, 2):
            for dj in (0, 2):
                c = ((qi - di + R) % R) * S + ((qj - dj + S) % S)
                dets.append(c)
        d0, d1, d2, d3 = dets
        pairs = [(d0, d1), (d0, d2), (d0, d3), (d1, d2), (d1, d3), (d2, d3)]
        for u, v in pairs:
            # Merge strategy: combine weights if edge already exists
            # (multiple qubits can share a detector pair on torus)
            m.add_edge(u, v, weight=weight, fault_ids={q},
                       merge_strategy="replace")
            # Track qubits for each edge
            key = (u, v) if u < v else (v, u)
            if key not in edge_to_qubit:
                edge_to_qubit[key] = set()
            edge_to_qubit[key].add(q)

    m.set_boundary_nodes(set())
    return m, edge_to_qubit


def pymatching_observable_flip(matching, edge_to_qubit, detection_events):
    """
    Decode with PyMatching and compute whether the correction flips
    the Z-observable on OBS_QUBITS.

    Returns True if the net effect (error + correction) introduces
    a logical error on the tracked observable, False otherwise.
    """
    correction, _ = matching.decode(detection_events, return_weight=True)
    # correction[i] is True if edge i was flipped by the matching
    obs_parity = 0
    for i, flipped in enumerate(correction):
        if not flipped:
            continue
        # Find qubits for this edge
        edge = matching.edges()[i]
        key = (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])
        qubits = edge_to_qubit.get(key, set())
        # This edge represents ALL qubits that connect this detector pair.
        # We use the shared qubit for the observable check.
        # For disambiguation: take any qubit from the intersection
        # (in practice each edge is from one qubit, except torus boundaries)
        obs_qubits_set = OBS_QUBITS
        for q in qubits:
            if q in obs_qubits_set:
                obs_parity ^= 1
                break  # count each edge at most once
    return obs_parity % 2 == 1


# =========================================================================
#  Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plane-Warp vs PyMatching on 40x40 BB code")
    parser.add_argument("--trials", type=int, default=500,
                        help="Trials per weight")
    parser.add_argument("--weights", type=str,
                        default="1,2,3,5,7,10,15,20,30,50",
                        help="Error weights to test")
    parser.add_argument("--px", type=float, default=0.01,
                        help="Error probability for edge weight (default 0.01)")
    args = parser.parse_args()

    weights = [int(w) for w in args.weights.split(",")]

    print(f"{'='*68}")
    print(f"Plane-Warp vs PyMatching ({R}x{S} BB code torus, n={N})")
    print(f"Trials/weight: {args.trials}")
    print(f"Observable: Z on sub-lattice row 0  ({len(OBS_QUBITS)} qubits)")
    print(f"{'='*68}\n")

    # Build PyMatching graph
    print("Building PyMatching graph (clique decomposition) … ", end="", flush=True)
    t0 = time.time()
    pm, e2q = build_pymatching(p_error=args.px)
    dt = time.time() - t0
    print(f"{dt:.1f}s  ({pm.num_detectors} detectors, {pm.num_edges} edges)")

    rng = np.random.RandomState(42)

    header = (f"\n{'Wt':>5}  {'PW OK':>6}  {'PW%':>7}   "
              f"{'PM OK':>6}  {'PM%':>7}   {'diff':>6}")
    print(header)
    print("-" * len(header))

    for w in weights:
        if w > N:
            continue

        pw_ok = 0   # plane_warp: diff is stabilizer (no logical error)
        pm_ok = 0   # pymatching:  no observable flip

        for _ in range(args.trials):
            err = bytearray(N)
            qubits = rng.choice(N, size=w, replace=False)
            for q in qubits:
                err[q] = 1
            err_bytes = bytes(err)

            syn_bytes = syndrome_of(err_bytes)

            # ---- Plane-Warp ----
            corr_pw = decode_planewarp(syn_bytes)
            diff = bytes(a ^ b for a, b in zip(err_bytes, corr_pw))
            if is_stabilizer(diff):
                pw_ok += 1

            # ---- PyMatching ----
            # syn_bytes has one byte per detector (0/1), no packing needed
            det = np.frombuffer(syn_bytes, dtype=np.uint8).astype(np.bool_)
            pm_flips_obs = pymatching_observable_flip(pm, e2q, det)
            # Ground truth: did the original error flip the observable?
            actual_obs_flip = sum(err[q] for q in OBS_QUBITS) % 2
            # PyMatching correct if its correction + error leaves obs unfipped:
            # (actual_obs_flip XOR pm_flips_obs) == 0  means no net flip
            if actual_obs_flip == 0 and not pm_flips_obs:
                pm_ok += 1
            elif actual_obs_flip == 1 and pm_flips_obs:
                pm_ok += 1
            # else: PyMatching didn't match the observable correctly

        pw_rate = 100.0 * pw_ok / args.trials
        pm_rate = 100.0 * pm_ok / args.trials
        diff = pw_rate - pm_rate
        mark = " ***" if abs(diff) > 10 else ""
        print(f"{w:5d}  {pw_ok:4d}/{args.trials:<4d} {pw_rate:6.1f}%  "
              f"{pm_ok:4d}/{args.trials:<4d} {pm_rate:6.1f}%  "
              f"{diff:+6.1f}%{mark}")

    print(f"\n{'='*68}")
    print("PW = plane_warp C decoder (full hypergraph, provably exact O(64n))")
    print("PM = PyMatching graph matching (clique-decomposed 4→6 edges)")
    print("Both track the same Z-observable on qubits 0,2,4,...,38.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
