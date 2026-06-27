"""
waxis_decode.py — project-decode port: basis injection + consistency projection.

Verbatim from plane_warp.c's `--project-decode`:
  stdin: [n:u32] [n*(syn+corr) pairs] [rnd0][rnd1][rnd2][rnd3]
  stdout: [correction]

1. Load n injected (syn, corr) basis pairs.
2. Read 4 rounds of syndrome.
3. Build H matrix column-by-column.
4. Find consistent bits (all 4 rounds agree).
5. RREF solve H[consistent]·E = syn[consistent].
6. Project S' = H·E (guaranteed in Im(H)).
7. decode_linear_basis(S') → correction.
"""

import numpy as np
import os, ctypes as _ct


class WaxisDecoder:
    def __init__(self, r, s):
        self.r = r; self.s = s; self.n = r * s
        self._H = None
        self._basis_syn = None   # RREF'd working copy of basis syndromes
        self._basis_corr = None  # RREF'd working copy of basis corrections
        self._pivots = None      # pivot positions per basis row
        self._basis_rank = 0
        self._load_c_lib()

    def _load_c_lib(self):
        _lib_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(_lib_dir, "libplane_warp.so")
        self._lib = _ct.CDLL(lib_path)
        self._lib.solve_plane.argtypes = [_ct.c_int, _ct.c_int,
            _ct.POINTER(_ct.c_uint8), _ct.POINTER(_ct.c_uint8)]
        self._lib.solve_plane.restype = _ct.c_int
        self._lib.preprocess_syndrome.argtypes = [_ct.c_int, _ct.c_int,
            _ct.POINTER(_ct.c_uint8)]
        self._lib.preprocess_syndrome.restype = None
        self._lib.syndrome_of.argtypes = [_ct.c_int, _ct.c_int,
            _ct.POINTER(_ct.c_uint8), _ct.POINTER(_ct.c_uint8)]
        self._lib.syndrome_of.restype = None

    def _syn_of(self, corr):
        syn = np.zeros((self.r, self.s), dtype=np.uint8)
        self._lib.syndrome_of(self.r, self.s,
            corr.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
            syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
        return syn.reshape(-1)

    def _preprocess(self, syn):
        self._lib.preprocess_syndrome(self.r, self.s,
            syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))

    def _solve(self, syn):
        out = np.zeros((self.r, self.s), dtype=np.uint8)
        self._lib.solve_plane(self.r, self.s,
            syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
            out.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
        return out

    # ---- Verbafom from plane_warp.c: decode_linear_basis ----

    def _inject_basis(self, syns, corrs):
        """Inject n (syn,corr) basis pairs.  Forward-eliminate to RREF.
        Verbatim from decode_linear_basis lines 1544-1573."""
        n = syns.shape[0]
        nn = self.n
        ws = syns.astype(np.uint8).copy()
        wc = corrs.astype(np.uint8).copy()
        pivot = [-1] * n
        for b in range(n):
            for q in range(nn):
                if ws[b, q]:
                    pivot[b] = q
                    break
            if pivot[b] < 0:
                continue
            # Eliminate pivot from all later basis vectors
            for b2 in range(b + 1, n):
                if ws[b2, pivot[b]]:
                    ws[b2] ^= ws[b]
                    wc[b2] ^= wc[b]
        self._basis_syn = ws
        self._basis_corr = wc
        self._pivots = pivot
        self._basis_rank = sum(1 for p in pivot if p >= 0)

    def _basis_solve(self, syn):
        """Decompose syn into basis.  Verbatim from decode_linear_basis lines 1575-1602.
        Returns correction (numpy array) if syn in span, else None."""
        nn = self.n
        n = len(self._pivots)
        temp = syn.copy()
        coeffs = np.zeros(n, dtype=np.uint8)
        for b in range(n):
            p = self._pivots[b]
            if p >= 0 and temp[p]:
                temp ^= self._basis_syn[b]
                coeffs[b] = 1
        if temp.any():
            return None
        out = np.zeros(nn, dtype=np.uint8)
        for b in range(n):
            if coeffs[b]:
                out ^= self._basis_corr[b]
        return out

    # ---- Verbatim from plane_warp.c: --project-decode ----

    def _build_H(self):
        if self._H is not None:
            return
        nn, r, s = self.n, self.r, self.s
        H = np.zeros((nn, nn), dtype=np.uint8)
        err = np.zeros(nn, dtype=np.uint8)
        syn = np.zeros(nn, dtype=np.uint8)
        for col in range(nn):
            err.fill(0); err[col] = 1
            self._lib.syndrome_of(r, s,
                err.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
                syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
            H[:, col] = syn
        self._H = H

    def _build_basis_default(self):
        """Build basis from weight-1 error syndromes if none injected."""
        if self._pivots is not None:
            return
        self._build_H()
        nn = self.n
        syns = self._H.T.copy()  # each row = syndrome of weight-1 error at that qubit
        corrs = np.eye(nn, dtype=np.uint8)
        self._inject_basis(syns, corrs)

    def decode(self, syndromes):
        """Verbatim --project-decode pipeline."""
        self._build_H()
        self._build_basis_default()

        rr, r, s, nn = syndromes.shape[0], self.r, self.s, self.n
        if rr == 1:
            syn = syndromes[0].copy(); self._preprocess(syn)
            return self._solve(syn)

        # Read 4 rounds (verbafom lines 2035-2039)
        rounds_data = syndromes.reshape(rr, nn).astype(np.uint8)

        # Find consistent bits: same value across all rounds (lines 2049-2058)
        clean_idx = []
        for q in range(nn):
            val = rounds_data[0, q]
            all_same = 1
            for rd in range(1, rr):
                if rounds_data[rd, q] != val:
                    all_same = 0; break
            if all_same:
                clean_idx.append(q)

        n_clean = len(clean_idx)
        if n_clean < 24:
            # Fallback: majority → preprocess → solve
            majority = (rounds_data.sum(axis=0) > rr // 2).astype(np.uint8)
            syn_maj = majority.reshape(r, s).copy()
            self._preprocess(syn_maj)
            return self._solve(syn_maj)

        # Augmented matrix: n_clean rows × (nn+1) cols (lines 2061-2068)
        A = np.zeros((n_clean, nn + 1), dtype=np.uint8)
        for i, q in enumerate(clean_idx):
            A[i, :nn] = self._H[q]
            A[i, nn] = rounds_data[0, q]

        # RREF (lines 2069-2086)
        rank = 0
        pivot_col = [-1] * n_clean
        for col in range(nn):
            pv = -1
            for i in range(rank, n_clean):
                if A[i, col]:
                    pv = i; break
            if pv < 0:
                continue
            if pv != rank:
                A[[rank, pv]] = A[[pv, rank]]
            pivot_col[rank] = col
            for i in range(n_clean):
                if i != rank and A[i, col]:
                    A[i] ^= A[rank]
            rank += 1

        # Consistency check (lines 2088-2093)
        for i in range(rank, n_clean):
            if A[i, nn] and A[i, :nn].sum() == 0:
                syn = syndromes[0].copy(); self._preprocess(syn)
                return self._solve(syn)

        # Extract E (lines 2095-2101)
        E = np.zeros(nn, dtype=np.uint8)
        for i in range(rank):
            c = pivot_col[i]
            if c >= 0:
                E[c] = A[i, nn]

        # Compute S' = H·E (lines 2102-2104)
        proj_syn = np.zeros(nn, dtype=np.uint8)
        self._lib.syndrome_of(r, s,
            E.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
            proj_syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))

        # Decode using basis (line 2106)
        corr = self._basis_solve(proj_syn)
        if corr is None:
            syn = syndromes[0].copy(); self._preprocess(syn)
            return self._solve(syn)

        return corr.reshape(r, s)
