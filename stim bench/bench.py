#!/usr/bin/env python3
"""Benchmark --decode vs --decode-persist across grid sizes at pg=0.1."""
import stim, subprocess, struct, numpy as np, sys, time, argparse

PG = 0.1
PM = 0.01
ROUNDS = 5
SEED = 2024

def make_circuit(R, S, rounds):
    N = R * S
    c = stim.Circuit()
    for rnd in range(rounds):
        c.append('DEPOLARIZE1', list(range(N)), PG / 10)
        c.append('R', range(N, 2 * N))
        c.append('H', range(N, 2 * N))
        for a in range(R):
            for b in range(S):
                anc = N + a * S + b
                qs = [(a % R) * S + (b % S),
                      ((a + 2) % R) * S + (b % S),
                      (a % R) * S + ((b + 2) % S),
                      ((a + 2) % R) * S + ((b + 2) % S)]
                for q in qs:
                    c.append('CZ', [anc, q])
        c.append('H', range(N, 2 * N))
        c.append('X_ERROR', range(N, 2 * N), PM)
        c.append('M', range(N, 2 * N))
    c.append('M', range(N))
    return c

parser = argparse.ArgumentParser()
parser.add_argument('--grid', type=int, default=14, help='grid side length')
parser.add_argument('--shots', type=int, default=4000)
parser.add_argument('--pg', type=float, default=0.1, help='two-qubit gate error rate')
parser.add_argument('--pm', type=float, default=0.01, help='measurement error rate')
parser.add_argument('--range', type=int, nargs=3, metavar=('FROM', 'TO', 'STEP'),
                    help='sweep grid sizes: from to step')
opts = parser.parse_args()
PG = opts.pg
PM = opts.pm

if opts.range:
    grids = list(range(opts.range[0], opts.range[1] + 1, opts.range[2]))
else:
    grids = [opts.grid]

print(f"{'grid':>8}  {'base':>6}  {'plain':>7}  {'persist':>8}  {'shots/s':>7}")
print("-" * 47)

for size in grids:
    R = S = size
    N = R * S
    sh = opts.shots

    c = make_circuit(R, S, ROUNDS)
    s = c.compile_sampler(seed=SEED)
    shots = s.sample(shots=sh).astype(np.uint8)

    obs = list(range(0, S, 2))
    ep = epr = eb = 0
    t0 = time.time()

    for t in range(sh):
        shot = shots[t]
        ov = int(sum(int(shot[ROUNDS * N + q]) for q in obs) % 2)
        eb += ov  # baseline: no correction

        syn = shot[(ROUNDS - 1) * N:ROUNDS * N]
        r = subprocess.run(
            ['./plane_warp', str(R), str(S), '--decode'],
            input=syn.tobytes(), capture_output=True)
        cr = np.frombuffer(r.stdout, dtype=np.uint8)
        if len(cr) >= N:
            ep += ov ^ int(sum(int(cr[q]) for q in obs) % 2)

        buf = struct.pack('<I', ROUNDS)
        for rnd in range(ROUNDS):
            buf += bytes(shot[rnd * N:(rnd + 1) * N])
        r = subprocess.run(
            ['./plane_warp', str(R), str(S), '--decode-persist'],
            input=buf, capture_output=True)
        cr = np.frombuffer(r.stdout, dtype=np.uint8)
        if len(cr) >= N:
            epr += ov ^ int(sum(int(cr[q]) for q in obs) % 2)

    dt = time.time() - t0
    print(f"{R}x{S:<4}  {eb/sh*100:>6.1f}%  {ep/sh*100:>6.2f}%  {epr/sh*100:>7.2f}%  {sh/dt:>6.0f}")
