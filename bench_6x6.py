#!/usr/bin/env python3
"""
Comprehensive 6×6 BB code benchmark: plane_warp vs everything STIM can throw.

Decoders:
  PW-raw   — plane_warp --decode (ML, trusts syndrome)
  PW-pp    — plane_warp --decode-pp (H^T·S=0 preprocess + 4-pass recover)
  PMatch   — pymatching clique-decomposed graph + boundary edges
  BP+OSD   — belief propagation + ordered statistics decoding (if ldpc installed)

Tests:
  1. Exhaustive weight-1 X errors (all 36)
  2. Exhaustive weight-2 X errors (all 630 pairs)
  3. Sampled weight-{3,4,5,6} X errors
  4. Measurement noise sweep (p_flip 0..5%) with weight-2 data errors
  5. Combined data + measurement noise at realistic rates

Grid: 6×6 torus, n=36, distance≈3, 4 sub-lattices 3×3 each.
"""

import os, subprocess, sys, time, math, itertools
import numpy as np
import stim
import pymatching

DECODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plane_warp")
R = S = 6
N = R * S
HR, HS = R // 2, S // 2
OBS_QUBITS = frozenset(range(0, S, 2))  # Z on first sub-lattice row


# =========================================================================
#  Core helpers
# =========================================================================
def syndrome_of(err):
    syn = bytearray(N)
    for q in range(N):
        if not err[q]: continue
        qi, qj = q // S, q % S
        for di in (0, 2):
            for dj in (0, 2):
                syn[((qi - di + R) % R) * S + ((qj - dj + S) % S)] ^= 1
    return bytes(syn)


def is_stabilizer(vec):
    for px in range(2):
        for py in range(2):
            for si in range(HR):
                if sum(vec[(px + 2 * si) * S + (py + 2 * sj)] for sj in range(HS)) & 1:
                    return False
            for sj in range(HS):
                if sum(vec[(px + 2 * si) * S + (py + 2 * sj)] for si in range(HR)) & 1:
                    return False
    return True


def decode_pw(syn, pp=False):
    flag = "--decode-pp" if pp else "--decode"
    p = subprocess.run([DECODER, str(R), str(S), flag],
                       input=syn, capture_output=True, timeout=60)
    return p.stdout


# =========================================================================
#  STIM circuit
# =========================================================================
def build_stim_circuit():
    c = stim.Circuit()
    for qi in range(R):
        for qj in range(S):
            c.append("QUBIT_COORDS", qi * S + qj, [qi, qj])
    # HZ checks
    hz = []
    for a in range(R):
        for b in range(S):
            qs = [(a % R) * S + (b % S), ((a + 2) % R) * S + (b % S),
                  (a % R) * S + ((b + 2) % S), ((a + 2) % R) * S + ((b + 2) % S)]
            hz.append(f"Z{qs[0]}*Z{qs[1]}*Z{qs[2]}*Z{qs[3]}")
    c += stim.Circuit("MPP " + " ".join(hz))
    for i in range(N):
        c.append("DETECTOR", [stim.target_rec(-(N - i))])
    c.append("M", list(range(N)))
    for q in OBS_QUBITS:
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(N - q))], 0)
    return c


# =========================================================================
#  PyMatching
# =========================================================================
def build_pymatching(p_err=0.01, p_flip=0.01):
    wd = -math.log(p_err) if p_err > 0 else 999
    wf = -math.log(p_flip) if p_flip > 0 else 999
    m = pymatching.Matching()
    for q in range(N):
        qi, qj = q // S, q % S
        dt = [((qi - di + R) % R) * S + ((qj - dj + S) % S)
              for di in (0, 2) for dj in (0, 2)]
        for u, v in [(dt[0], dt[1]), (dt[0], dt[2]), (dt[0], dt[3]),
                     (dt[1], dt[2]), (dt[1], dt[3]), (dt[2], dt[3])]:
            m.add_edge(u, v, weight=wd, merge_strategy="replace")
    if p_flip > 0:
        for d in range(N):
            m.add_boundary_edge(d, weight=wf)
    m.set_boundary_nodes(set())
    e2q = {}
    for i, ed in enumerate(m.edges()):
        u, v, _ = ed
        if v is None: continue
        qs = set()
        for q in range(N):
            qi, qj = q // S, q % S
            dt = frozenset(((qi - di + R) % R) * S + ((qj - dj + S) % S)
                           for di in (0, 2) for dj in (0, 2))
            if u in dt and v in dt: qs.add(q)
        if qs: e2q[i] = qs
    return m, e2q


def decode_pm(m, e2q, syn):
    det = np.frombuffer(syn, dtype=np.uint8).astype(np.bool_)
    try:
        corr_edges, _ = m.decode(det, return_weight=True)
    except ValueError:
        return None
    corr = bytearray(N)
    for i, f in enumerate(corr_edges):
        if not f or i not in e2q: continue
        corr[next(iter(e2q[i]))] ^= 1
    return bytes(corr)


# =========================================================================
#  BP+OSD  (ldpc package, if available)
# =========================================================================
try:
    import ldpc
    HAS_LDPC = True
except ImportError:
    HAS_LDPC = False


def build_ldpc_decoder():
    """Build check matrix H (N×N) for the 6×6 plus-shaped code."""
    H = np.zeros((N, N), dtype=np.uint8)
    for q in range(N):
        qi, qj = q // S, q % S
        for di in (0, 2):
            for dj in (0, 2):
                c = ((qi - di + R) % R) * S + ((qj - dj + S) % S)
                H[c, q] ^= 1
    return H


def decode_bposd(H, syn, osd_order=4):
    """Belief propagation + ordered statistics decoding."""
    if not HAS_LDPC:
        return None
    bp = ldpc.bposd_decoder(H, error_rate=0.02, max_iter=50,
                             bp_method="product_sum",
                             osd_method="osd_cs", osd_order=osd_order)
    syndrome = np.frombuffer(syn, dtype=np.uint8).astype(np.int32)
    corr = bp.decode(syndrome)
    if corr is None:
        return None
    return bytes(corr.astype(np.uint8).tobytes())


# =========================================================================
#  Bench
# =========================================================================
def bench_weight_exhaustive(max_w, pw_pp=False):
    """Exhaustive over all weight-w errors, count correct decodes."""
    results = {}
    for w in range(1, max_w + 1):
        tested, ok_pw = 0, 0
        for combo in itertools.combinations(range(N), w):
            err = bytearray(N)
            for q in combo: err[q] = 1
            eb = bytes(err)
            syn = syndrome_of(eb)
            corr = decode_pw(syn, pp=pw_pp)
            if is_stabilizer(bytes(a ^ b for a, b in zip(eb, corr))):
                ok_pw += 1
            tested += 1
        results[w] = (ok_pw, tested)
    return results


def bench_weight_sampled(weights, trials, decoders, pe=0.0, pf=0.0,
                          pm_data=None, ldpc_H=None):
    rng = np.random.RandomState(42)
    results = {name: {w: 0 for w in weights} for name in decoders}
    for w in weights:
        for _ in range(trials):
            err = bytearray(N)
            qs = rng.choice(N, size=w, replace=False) if w <= N else range(N)
            if isinstance(qs, range): qs = list(qs)
            for q in qs: err[q] ^= 1
            eb = bytes(err)
            syn_true = syndrome_of(eb)
            # Add measurement noise
            syn_noisy = bytearray(syn_true)
            for d in range(N):
                if rng.random() < pf: syn_noisy[d] ^= 1
            snb = bytes(syn_noisy)

            for dec_name in decoders:
                if dec_name == "PW-raw":
                    corr = decode_pw(snb, pp=False)
                elif dec_name == "PW-pp":
                    corr = decode_pw(snb, pp=True)
                elif dec_name == "PMatch" and pm_data:
                    corr = decode_pm(*pm_data, snb)
                elif dec_name == "BP+OSD" and ldpc_H is not None:
                    corr = decode_bposd(ldpc_H, snb)
                else:
                    continue
                if corr and is_stabilizer(bytes(a ^ b for a, b in zip(eb, corr))):
                    results[dec_name][w] += 1
    return results


def print_results_table(results, trials, header=""):
    if header:
        print(f"\n{header}")
    dec_names = list(results.keys())
    print(f"  {'weight':>7s}", end="")
    for name in dec_names:
        print(f"  {name:>9s}", end="")
    print()
    print(f"  {'-'*7}", end="")
    for _ in dec_names:
        print(f"  {'-'*9}", end="")
    print()
    weights = sorted(results[dec_names[0]].keys())
    for w in weights:
        print(f"  {w:7d}", end="")
        for name in dec_names:
            ok = results[name][w]
            pct = 100.0 * ok / trials
            print(f"  {ok:4d}/{trials:<4d}", end="")
        print()


# =========================================================================
#  Main
# =========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="6×6 comprehensive bench")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--skip-exhaustive", action="store_true")
    args = parser.parse_args()
    T = args.trials

    print(f"{'='*80}")
    print(f"Comprehensive 6×6 BB Code Benchmark  —  n={N}, d≈3")
    print(f"{'='*80}")
    print(f"\nDecoders under test:")
    print(f"  PW-raw  — plane_warp --decode   (ML, trusts syndrome)")
    print(f"  PW-pp   — plane_warp --decode-pp (H^T·S=0 + 4-pass recover)")
    print(f"  PMatch  — PyMatching clique-decomposed graph + bound edges")
    if HAS_LDPC:
        print(f"  BP+OSD  — belief propagation + OSD (osd_order=4)")
    else:
        print(f"  BP+OSD  — [ldpc not installed]")

    # ---- Tier 1: Exhaustive weight-1 and weight-2 ----
    if not args.skip_exhaustive:
        print(f"\n{'─'*80}")
        print(f"[1] Exhaustive weight-1 & weight-2 (no measurement noise)")
        for pp_flag, label in [(False, "PW-raw"), (True, "PW-pp")]:
            res = bench_weight_exhaustive(2, pw_pp=pp_flag)
            for w in (1, 2):
                ok, total = res[w]
                pct = 100.0 * ok / total
                print(f"  {label:>7s}  weight-{w}: {ok}/{total} ({pct:.1f}%)"
                      f"  {'PASS' if ok==total else 'FAIL: '+str(total-ok)+' logical errors'}")

    # ---- Tier 2: Sampled weights with no measurement noise ----
    print(f"\n{'─'*80}")
    print(f"[2] Sampled weight sweep (no meas noise, T={T})")
    pm, e2q = build_pymatching(p_err=0.02, p_flip=0.0)
    ldpc_H = build_ldpc_decoder() if HAS_LDPC else None
    decoders = ["PW-raw", "PW-pp", "PMatch"]
    if HAS_LDPC: decoders.append("BP+OSD")
    weights = [1, 2, 3, 4, 5, 6]
    res = bench_weight_sampled(weights, T, decoders,
                                pm_data=(pm, e2q), ldpc_H=ldpc_H)
    print_results_table(res, T, header="No measurement noise:")

    # ---- Tier 3: Measurement noise sweep ----
    print(f"\n{'─'*80}")
    print(f"[3] Measurement noise sweep (p_err=0, T={T})")
    for pf in (0.0, 0.01, 0.02, 0.03, 0.05):
        pm2, e2q2 = build_pymatching(p_err=0.001, p_flip=pf)
        res2 = bench_weight_sampled([0], T, decoders, pf=pf,
                                     pm_data=(pm2, e2q2), ldpc_H=ldpc_H)
        vals = "".join(f"  {res2[name][0]:4d}/{T:<4d}" for name in decoders)
        print(f"  p_flip={pf:.2f}{vals}")

    # ---- Tier 4: Combined realistic rates ----
    print(f"\n{'─'*80}")
    print(f"[4] Combined data + measurement noise (realistic, T={T})")
    for pe, pf in [(0.01, 0.01), (0.01, 0.02), (0.02, 0.01), (0.02, 0.02),
                    (0.03, 0.03)]:
        pm3, e2q3 = build_pymatching(p_err=pe, p_flip=pf)
        res3 = bench_weight_sampled([int(N * pe)], T, decoders, pe=pe, pf=pf,
                                     pm_data=(pm3, e2q3), ldpc_H=ldpc_H)
        w = int(N * pe)
        vals = "".join(f"  {res3[name][w]:4d}/{T:<4d}" for name in decoders)
        print(f"  p_err={pe:.2f} p_flip={pf:.2f} (w≈{w}){vals}")

    # ---- Tier 5: STIM cross-validation ----
    print(f"\n{'─'*80}")
    print(f"[5] STIM circuit cross-validation")
    c = build_stim_circuit()
    sampler = c.compile_sampler()
    ok = 0
    for q in range(N):
        err = bytearray(N); err[q] = 1
        ec = stim.Circuit(f"X_ERROR(1) {q}")
        ec += c
        bits = ec.compile_sampler().sample(shots=1)[0].astype(np.uint8)
        syn_stim = bits[:N]
        syn_c = np.array(list(syndrome_of(bytes(err))), dtype=np.uint8)
        if np.array_equal(syn_stim, syn_c): ok += 1
    print(f"  STIM ↔ plane_warp syndrome match: {ok}/{N} "
          f"({'PASS' if ok==N else 'FAIL'})")
    print(f"\n  STIM circuit ({N} qubits, {N} HZ checks):")
    print(f"  {c}")

    print(f"\n{'='*80}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
