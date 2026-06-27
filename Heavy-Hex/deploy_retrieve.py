#!/usr/bin/env python3
"""
Retrieve a completed Heron r2 job and decode all shots via consistency
projection + linear basis decoder.

Usage:
  python3 deploy_retrieve.py <job_id> [--learn-basis]

Job parameters read from ~/.planewarp_jobs.json (created by deploy_heron.py).
"""

import json, sys, os, time
from pathlib import Path

import numpy as np

SAVE_FILE = Path.home() / ".planewarp_jobs.json"
IBM_TOKEN_ENV = "IBM_QUANTUM_TOKEN"
CLEAN_DIR = Path.home() / ".planewarp_clean"


def get_token():
    token = os.environ.get(IBM_TOKEN_ENV)
    if token:
        return token
    from getpass import getpass
    token = getpass("IBM Quantum API token: ")
    if token:
        return token
    print("No token.", file=sys.stderr)
    sys.exit(1)


def all_syndromes(pub_result, rounds, r, s):
    n_stab = r * s
    first = getattr(pub_result.data, "syn_0")
    shots = first.num_shots
    syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
    bits0 = getattr(pub_result.data, "syn_0").to_bool_array(order='little')
    shared = (bits0.shape[1] == n_stab)
    for c in range(rounds):
        bits = getattr(pub_result.data, f"syn_{c}").to_bool_array(order='little')
        if shared:
            m = bits[:, :n_stab].reshape(shots, r, s)
            syn[:, c] = m ^ np.roll(m, shift=-2, axis=2)
        else:
            pair = bits[:, :2 * n_stab].reshape(shots, n_stab, 2)
            syn[:, c] = (pair[:, :, 0] ^ pair[:, :, 1]).reshape(shots, r, s)
    return syn


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Retrieve and decode Heron r2 job")
    ap.add_argument('job_id', nargs='?', help='job ID to retrieve')
    ap.add_argument('--learn-basis', action='store_true',
                    help='build empirical basis from readout data and save to ~/.planewarp_clean/basis_hw.npz')
    opts = ap.parse_args()

    job_id = opts.job_id
    if not job_id:
        print("Usage: python3 deploy_retrieve.py <job_id>", file=sys.stderr)
        sys.exit(1)

    token = get_token()

    from qiskit_ibm_runtime import QiskitRuntimeService
    from pw_qiskit import PlaneWarp

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    job = service.job(job_id)
    status = str(job.status())
    print(f"Job: {job_id}")
    print(f"Status: {status}")

    if status in ("ERROR", "CANCELLED"):
        print(f"Job ended with {status}.  Cannot retrieve results.", file=sys.stderr)
        sys.exit(1)
    if status != "DONE":
        print("Job not yet completed.  Try again later.", file=sys.stderr)
        sys.exit(1)

    # Load saved params
    params = {}
    if SAVE_FILE.exists():
        saved = json.loads(SAVE_FILE.read_text())
        params = saved.get(job_id, {})

    r = params.get("r", 6)
    s = params.get("s", 8)
    rounds = params.get("rounds", 4)

    result = job.result()
    pub_result = result[0]

    all_syn = all_syndromes(pub_result, rounds, r, s)
    n_shots = all_syn.shape[0]

    # Data readout
    data_raw = None
    if hasattr(pub_result.data, "data"):
        dbits = getattr(pub_result.data, "data").to_bool_array(order='little')
        data_raw = dbits.astype(np.uint8).reshape(n_shots, r, s)

    pw = PlaneWarp()

    # Load 24-dim basis
    # Load basis (prefer HW-learned; supplement with synthetic if <24 dims)
    basis_syn, basis_corr = [], []
    bp = CLEAN_DIR / 'basis_hw.npz'
    if bp.exists():
        bd = np.load(bp)
        n_dims = int(bd.get('n_dims', len(bd['syn'])))
        basis_syn = [np.asarray(bd['syn'][i], dtype=np.uint8).reshape(r, s) for i in range(n_dims)]
        basis_corr = [np.asarray(bd['corr'][i], dtype=np.uint8).reshape(r, s) for i in range(n_dims)]
        print(f"  HW basis loaded: {len(basis_syn)} dims (from {bp})")
        if n_dims < 24:
            syn_syn, syn_corr = pw.build_synthetic_basis(r, s)
            used = set(tuple(s.ravel()) for s in basis_syn)
            for i in range(len(syn_syn)):
                if len(basis_syn) >= 24:
                    break
                t = tuple(syn_syn[i].ravel())
                if t not in used:
                    used.add(t)
                    basis_syn.append(syn_syn[i])
                    basis_corr.append(syn_corr[i])
            print(f"  Supplemented with synthetic: {len(basis_syn)} dims")
    else:
        print(f"  No learned basis at {bp} — building synthetic basis from H pivot columns")
        basis_syn, basis_corr = pw.build_synthetic_basis(r, s)
        print(f"  Synthetic basis: {len(basis_syn)} dims")

    # Consistency projection + basis decode
    decode_ok = 0
    decode_corr = np.zeros((n_shots, r, s), dtype=np.uint8)
    decoded_mask = np.zeros(n_shots, dtype=bool)
    for idx in range(n_shots):
        raw = np.stack([all_syn[idx, c] for c in range(rounds)])
        C, ok = pw.project_decode(basis_syn, basis_corr, raw)
        if ok:
            C = pw.refine_min_weight(C)
            decode_ok += 1
            decode_corr[idx] = C
            decoded_mask[idx] = True

    # Single-observable LER from data readout
    proj_perf = 0
    proj_det = 0
    proj_log = 0
    if data_raw is not None:
        for idx in range(n_shots):
            if not decoded_mask[idx]:
                continue
            E = data_raw[idx]
            res = (E ^ decode_corr[idx]).astype(np.uint8)
            sr = pw.syndrome_of(res)
            if sr.sum() == 0:
                if pw.is_stabilizer(res):
                    proj_perf += 1
                else:
                    proj_log += 1
            else:
                proj_det += 1

    # Build empirical basis from readout data
    if opts.learn_basis:
        if data_raw is None:
            print("\n  --learn-basis requires --readout data. Re-submit the job with --readout.")
        elif decode_ok == 0:
            print("\n  No decoded shots — cannot learn basis.")
        else:
            n_stab = r * s
            stab256 = pw._build_stab_group(r, s)
            pairs = []
            for idx in range(n_shots):
                if not decoded_mask[idx]:
                    continue
                C = decode_corr[idx]
                E = data_raw[idx]
                S_proj = pw.syndrome_of(C).ravel()
                e_flat = E.ravel()
                best_w = 48
                best_s = np.zeros(n_stab, dtype=np.uint8)
                for k in range(256):
                    cand = e_flat ^ stab256[k].ravel()
                    w = int(cand.sum())
                    if w < best_w:
                        best_w = w
                        best_s = stab256[k].ravel().copy()
                C_ideal = (e_flat ^ best_s).astype(np.uint8)
                if np.array_equal(pw.syndrome_of(C_ideal.reshape(r, s)).ravel(), S_proj):
                    pairs.append((S_proj.copy(), C_ideal.copy()))
            if pairs:
                B = np.array([p[0] for p in pairs], dtype=np.uint8)
                A = B.copy()
                n_pivots = min(24, len(A))
                pi_list = []
                for col in range(n_stab):
                    if len(pi_list) >= n_pivots:
                        break
                    if len(pi_list) >= len(A):
                        break
                    nz = np.where(A[:, col])[0]
                    if len(nz) == 0:
                        continue
                    pv = nz[0]
                    if pv != len(pi_list):
                        A[[len(pi_list), pv]] = A[[pv, len(pi_list)]]
                        pairs[len(pi_list)], pairs[pv] = pairs[pv], pairs[len(pi_list)]
                    pi_list.append(col)
                    pr = len(pi_list) - 1
                    for r2 in range(len(A)):
                        if r2 != pr and A[r2, col]:
                            A[r2] ^= A[pr]
                n_found = len(pi_list)
                syn_arr = np.array([pairs[i][0] for i in range(n_found)], dtype=np.uint8)
                corr_arr = np.array([pairs[i][1] for i in range(n_found)], dtype=np.uint8)
                hw_path = CLEAN_DIR / 'basis_hw.npz'
                np.savez_compressed(hw_path, syn=syn_arr, corr=corr_arr, r=r, s=s,
                                    source_job=job_id, sources=len(pairs), n_shots=n_shots,
                                    n_dims=n_found)
                msg = f"  Hardware basis saved: {hw_path}  ({n_found}/24 dims from {len(pairs)} valid pairs)"
                if n_found >= 24:
                    msg += "\n  -> Future runs will auto-use this basis."
                else:
                    msg += f"\n  -> Partial basis ({n_found}/24). Future runs will use this as a starting point."
                print(msg)
            else:
                print("\n  No valid pairs for basis learning (syndrome match required).")

    # Per-round syndrome diagnostics
    for c in range(rounds):
        wt = all_syn[:, c, :, :].sum() / n_shots
        print(f"  Round {c} syn wt: {wt:.1f} / {r*s} ({100*wt/(r*s):.1f}%)")

    print(f"\n=== Heron r2 results ===")
    print(f"  Grid:     {r}×{s}")
    print(f"  Rounds:   {rounds}")
    print(f"  Shots:    {n_shots}")
    print(f"  Method:   consistency projection → Col(H) → basis decode")
    print()
    print(f"  Projection decode:  {decode_ok}/{n_shots} ({100*decode_ok/n_shots:.1f}%)")
    print()
    if data_raw is not None:
        print(f"  Projection outcomes ({decode_ok} decoded):")
        print(f"    Perfect (E⊕C=0):         {proj_perf} ({100*proj_perf/max(1,decode_ok):.1f}%)")
        print(f"    Detectable (syn≠0):       {proj_det} ({100*proj_det/max(1,decode_ok):.1f}%)")
        print(f"    Logical (syn=0, not stab): {proj_log} ({100*proj_log/max(1,decode_ok):.1f}%)")
        print(f"  True LER (undetectable):    {100*proj_log/n_shots:.2f}%")
    else:
        print(f"  No data readout (rerun with --readout for LER).")

    # Update save file
    if SAVE_FILE.exists():
        saved = json.loads(SAVE_FILE.read_text())
        if job_id in saved:
            saved[job_id].update({
                "retrieved": True,
                "shots_completed": n_shots,
                "proj_decode_pct": round(100 * decode_ok / n_shots, 1),
                "proj_perf": int(proj_perf),
                "proj_det": int(proj_det),
                "proj_log": int(proj_log),
                "true_ler_pct": round(100 * proj_log / n_shots, 2) if data_raw is not None else None,
            })
            SAVE_FILE.write_text(json.dumps(saved, indent=2))


if __name__ == "__main__":
    main()
