#!/usr/bin/env python3
"""6×6 NISQ — comprehensive plane_warp bench. 20 logical observables, 36 data + 36 ancilla qubits."""

import sys, os, subprocess, struct, time
import numpy as np, stim

DECODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plane_warp")
R = S = 6
N = R * S
HR, HS = R // 2, S // 2

# 20 logical observables
OBS = []
for px in range(2):
    for py in range(2):
        for si in range(HR):
            OBS.append([(px + 2 * si) * S + (py + 2 * sj) for sj in range(HS)])
        for sj in range(HS - 1):
            OBS.append([(px + 2 * si) * S + (py + 2 * sj) for si in range(HR)])


def build_circuit(p_gate=0.01, rounds=5):
    c = stim.Circuit()
    ND = NA = N
    for _ in range(rounds):
        c.append("R", range(ND, ND + NA))
        c.append("H", range(ND, ND + NA))
        for a in range(R):
            for b in range(S):
                anc = ND + a * S + b
                qs = [(a % R) * S + (b % S), ((a + 2) % R) * S + (b % S),
                      (a % R) * S + ((b + 2) % S), ((a + 2) % R) * S + ((b + 2) % S)]
                for q in qs:
                    c.append("CZ", [anc, q])
                    c.append("DEPOLARIZE2", [anc, q], p_gate)
        c.append("H", range(ND, ND + NA))
        c.append("M", range(ND, ND + NA))
    c.append("M", range(ND))
    return c


def decode(fn, syn):
    return subprocess.run([DECODER, str(R), str(S), fn],
                          input=syn, capture_output=True, timeout=10).stdout


def decode_mr(rounds_syn):
    buf = struct.pack("<i", len(rounds_syn))
    for s in rounds_syn: buf += s
    return subprocess.run([DECODER, str(R), str(S), "--decode-mr"],
                          input=buf, capture_output=True, timeout=10).stdout


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_obs = len(OBS)

    # Sweep over gate error rates
    configs = [
        (0.001, 5, "p_g=0.1% (5r)"),
        (0.002, 5, "p_g=0.2% (5r)"),
        (0.005, 5, "p_g=0.5% (5r)"),
        (0.010, 5, "p_g=1.0% (5r)"),
        (0.020, 5, "p_g=2.0% (5r)"),
        (0.001, 3, "p_g=0.1% (3r)"),
        (0.010, 3, "p_g=1.0% (3r)"),
    ]

    print(f"{'='*85}")
    print(f"6×6 NISQ Comprehensive — {n_obs} observables, T={T}")
    print(f"36 data + 36 ancilla qubits, {144} CZ gates/round")
    print(f"{'='*85}")
    print(f"  {'config':>16s}  {'base':>7s}  {'raw':>7s}  {'pp':>7s}  "
          f"{'3d':>7s}  {'mr':>7s}  {'best':>6s}")
    print(f"  {'─'*16}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}")

    results = []
    for pg, n_rd, label in configs:
        c = build_circuit(pg, n_rd)
        shots = c.compile_sampler().sample(shots=T).astype(np.uint8)
        assert shots.shape[1] == (n_rd + 1) * N

        n_base = n_raw = n_pp = n_3d = n_mr = total = 0
        t0 = time.time()

        for t in range(T):
            shot = shots[t]
            syn = bytes(shot[(n_rd - 1) * N : n_rd * N])
            dm = shot[n_rd * N : (n_rd + 1) * N]
            rounds_syn = [bytes(shot[r * N : (r + 1) * N]) for r in range(n_rd)]

            cr = decode("--decode", syn)
            cp = decode("--decode-pp", syn)
            c3 = decode("--decode-3d", syn)
            cm = decode_mr(rounds_syn)

            for qs in OBS:
                ov = int(sum(dm[q] for q in qs) % 2)
                total += 1
                if ov == 1:
                    n_base += 1
                if (ov ^ sum(cr[q] for q in qs) % 2) == 1:
                    n_raw += 1
                if (ov ^ sum(cp[q] for q in qs) % 2) == 1:
                    n_pp += 1
                if (ov ^ sum(c3[q] for q in qs) % 2) == 1:
                    n_3d += 1
                if (ov ^ sum(cm[q] for q in qs) % 2) == 1:
                    n_mr += 1

        best = min(n_raw, n_pp, n_3d, n_mr)
        names = {"raw": n_raw, "pp": n_pp, "3d": n_3d, "mr": n_mr}
        winner = [k for k, v in names.items() if v == best][0]

        def mv(v):
            return f"*{100*v/total:.2f}%" if v == best else f"{100*v/total:.2f}%"

        results.append((label, n_base, n_raw, n_pp, n_3d, n_mr, total, winner))
        print(f"  {label:>16s}  {100*n_base/total:6.2f}%  {mv(n_raw)}  "
              f"{mv(n_pp)}  {mv(n_3d)}  {mv(n_mr)}  {winner:>6s}")

    print(f"{'='*85}")
    print(f"base = do-nothing LER. raw = --decode, pp = --decode-pp,"
          f" 3d = --decode-3d, mr = --decode-mr")
    print()
    print("Winner tally:")
    from collections import Counter
    winners = Counter(r[-1] for r in results)
    for w, c in winners.most_common():
        print(f"  {w}: {c}/{len(results)} configs")
    print(f"{'='*85}")

    # Summary: best LER reduction
    print(f"\n  {'config':>16s}  {'base':>7s}  {'best-LER':>9s}  {'reduction':>10s}")
    for label, n_base, n_raw, n_pp, n_3d, n_mr, total, win in results:
        best_ler = min(n_raw, n_pp, n_3d, n_mr) / total * 100
        base_ler = n_base / total * 100
        reduction = (base_ler - best_ler) / base_ler * 100 if base_ler > 0 else 0
        print(f"  {label:>16s}  {base_ler:6.2f}%  {best_ler:8.2f}%  {reduction:9.1f}%")

    print(f"\n{'='*85}")
    print("Done.")


if __name__ == "__main__":
    main()
