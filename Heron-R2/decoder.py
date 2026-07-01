"""
decoder.py — Core decoder for the (1+x²)(1+y²) code.

Provides:
  - tesseract_decode_ffinal(syndromes, r, s)  — ffinal decoder (no AND-vote)
  - tesseract_decode_rot(syndromes, r, s)    — ffinal + best single rotation
  - tesseract_decode_rot4(syndromes, r, s)   — all 4 rotations stacked
  - decode_np(syn, r, s)     — subprocess: plane_warp --decode-np
  - prep(syn, r, s)          — C library preprocess_syndrome wrapper
  - solve(syn, r, s)         — C library solve_plane + min-weight kernel
  - S_of(E, r, s)            — compute syndrome from error pattern
  - check_logical(corr, r, s) — logical Z values from correction

All functions take explicit (r, s) grid dimensions — no global side effects.
"""
import numpy as np
import ctypes as _ct
import os as _os

_lib_dir = _os.path.dirname(_os.path.abspath(__file__))
_lib = _ct.CDLL(_os.path.join(_lib_dir, "libplane_warp.so"))
_lib.preprocess_syndrome.argtypes = [_ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_uint8)]
_lib.preprocess_syndrome.restype = None
_lib.solve_plane.argtypes = [_ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_uint8), _ct.POINTER(_ct.c_uint8)]
_lib.solve_plane.restype = _ct.c_int
_lib.syndrome_of.argtypes = [_ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_uint8), _ct.POINTER(_ct.c_uint8)]
_lib.syndrome_of.restype = None

# ---- 4D rotation helpers ----
ZW = [(1,0,0,1), (0,1,1,0), (1,1,0,1), (1,0,1,1), (0,1,1,1), (1,1,1,0)]

def rotate_syn(syn, dx, dy, mi):
    """Forward 4D rotation: syndrome → rotated syndrome."""
    r, s = syn.shape
    out = np.zeros((r, s), dtype=np.uint8)
    m00,m01,m10,m11 = ZW[mi]
    for i in range(r):
        for j in range(s):
            it = (i + dx) % r
            jt = (j + dy) % s
            si2 = (m00 * (it & 1) ^ m01 * (jt & 1)) & 1
            sj2 = (m10 * (it & 1) ^ m11 * (jt & 1)) & 1
            out[(it & ~1) + si2, (jt & ~1) + sj2] = syn[i, j]
    return out

def unrotate_corr(corr, dx, dy, mi):
    """Inverse 4D rotation: correction → unrotated correction."""
    r, s = corr.shape
    out = np.zeros((r, s), dtype=np.uint8)
    m00,m01,m10,m11 = ZW[mi]
    for i in range(r):
        for j in range(s):
            it = (i + dx) % r
            jt = (j + dy) % s
            si2 = (m00 * (it & 1) ^ m01 * (jt & 1)) & 1
            sj2 = (m10 * (it & 1) ^ m11 * (jt & 1)) & 1
            out[i, j] = corr[(it & ~1) + si2, (jt & ~1) + sj2]
    return out

# Best fixed rotations found at weight 3000 on 100×100
ROTS = [(0,0,0), (50,50,1), (0,50,2), (33,33,3)]

# Cache for min-weight kernel LUT per grid size
_lut_cache = {}

def _get_lut(r, s):
    key = (r, s)
    if key in _lut_cache:
        return _lut_cache[key]
    hr, hs = r // 2, s // 2
    lut = np.zeros((1 << (hr * hs), hr, hs), dtype=np.uint8)
    for idx in range(1 << (hr * hs)):
        sl = np.zeros((hr, hs), dtype=np.uint8)
        for b in range(hr * hs):
            if idx & (1 << b):
                sl[b // hs, b % hs] = 1
        best = sl.copy()
        best_wt = sl.sum()
        for rmask in range(1 << hr):
            for cmask in range(1 << hs):
                temp = sl.copy()
                for ri in range(hr):
                    if rmask & (1 << ri):
                        temp[ri, :] ^= 1
                for ci in range(hs):
                    if cmask & (1 << ci):
                        temp[:, ci] ^= 1
                wt = temp.sum()
                if wt < best_wt:
                    best_wt = wt
                    best = temp.copy()
        lut[idx] = best
    _lut_cache[key] = lut
    return lut


def min_weight_kernel_fast(corr, r, s):
    hr, hs = r // 2, s // 2
    lut = _get_lut(r, s)
    best = corr.copy()
    best_wt = best.sum()
    for target_z1 in (0, 1):
        for target_z2 in (0, 1):
            cur = corr.copy()
            if cur[0, :].sum() % 2 != target_z1:
                cur[0, :] ^= 1
            if cur[:, 0].sum() % 2 != target_z2:
                cur[:, 0] ^= 1
            for px in range(2):
                for py in range(2):
                    sl = cur[px::2, py::2]
                    idx = 0
                    for ri in range(hr):
                        for ci in range(hs):
                            if sl[ri, ci]:
                                idx |= 1 << (ri * hs + ci)
                    cur[px::2, py::2] = lut[idx]
            wt = cur.sum()
            if wt < best_wt:
                best_wt = wt
                best = cur.copy()
    return best


def prep(syn, r, s):
    _lib.preprocess_syndrome(r, s,
        syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))


def solve(syn, r, s):
    out = np.zeros((r, s), dtype=np.uint8)
    _lib.solve_plane(r, s,
        syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
        out.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
    return min_weight_kernel_fast(out, r, s)


def decode_np(syn, r, s, timeout=30):
    """Subprocess-based decode using ./plane_warp --decode-np.
    This calls solve_plane with internal restarts, giving ~72% at w=3000.
    Slower than ctypes but matches run_80pct.py exactly.
    """
    import subprocess as _sp
    bin_path = _os.path.join(_lib_dir, 'plane_warp')
    proc = _sp.run([bin_path, str(r), str(s), '--decode-np'],
                   input=syn.tobytes(), capture_output=True, timeout=timeout)
    return np.frombuffer(proc.stdout, np.uint8).reshape(r, s)


def S_of(E, r, s):
    out = np.zeros((r, s), dtype=np.uint8)
    _lib.syndrome_of(r, s,
        E.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
        out.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
    return out


def check_logical(corr, r, s):
    return corr[0, :].sum() % 2, corr[:, 0].sum() % 2


def tesseract_decode_ffinal(syndromes, r, s):
    """Decode ffinal: use LAST ROUND syndrome directly (skip AND-vote)."""
    syn = syndromes[-1].copy().astype(np.uint8)
    prep(syn, r, s)
    return solve(syn, r, s)


def tesseract_decode_rot(syndromes, r, s, mi=3, dx=33, dy=33):
    """ffinal decoder with best fixed 4D rotation (dx=33,dy=33,mi=3).
    Rotates syndrome before decode, unrotates correction.
    At weight 3000 on 100×100: ~76% vs ~72% identity (—rot).
    """
    syn = syndromes[-1].copy().astype(np.uint8)
    syn_r = rotate_syn(syn, dx, dy, mi)
    prep(syn_r, r, s)
    corr_r = solve(syn_r, r, s)
    return unrotate_corr(corr_r, dx, dy, mi)


def tesseract_decode_rot4(syndromes, r, s):
    """Run all 4 best rotations, return corrections stacked (4,r,s).
    For external union-rate evaluation; not for production single-shot.
    """
    syn = syndromes[-1].copy().astype(np.uint8)
    out = []
    for dx, dy, mi in ROTS:
        syn_r = rotate_syn(syn, dx, dy, mi)
        prep(syn_r, r, s)
        corr_r = solve(syn_r, r, s)
        out.append(unrotate_corr(corr_r, dx, dy, mi))
    return np.stack(out)


def tesseract_decode_np(syndromes, r, s, timeout=30):
    """Subprocess --decode-np: matches run_80pct.py exactly (~72% baseline).
    Takes last-round syndrome, decodes via subprocess (no prep).
    """
    syn = syndromes[-1].copy().astype(np.uint8)
    return decode_np(syn, r, s, timeout)


def tesseract_decode_np_rot4(syndromes, r, s, timeout=30):
    """4-rotation ANY-of-N pipeline using subprocess --decode-np.
    Returns stacked (4,r,s) corrections identical to run_80pct.py.
    Union rate ~84% at weight 3000 on 100×100.
    """
    syn = syndromes[-1].copy().astype(np.uint8)
    out = []
    for dx, dy, mi in ROTS:
        syn_r = rotate_syn(syn, dx, dy, mi)
        corr_r = decode_np(syn_r, r, s, timeout)
        out.append(unrotate_corr(corr_r, dx, dy, mi))
    return np.stack(out)


def tesseract_decode(syndromes, r, s):
    """AND-vote + viability + solve decoder."""
    rr, hr, hs = syndromes.shape[0], r // 2, s // 2
    syn_and = np.ones((r, s), dtype=np.uint8)
    for t in range(rr):
        syn_and &= syndromes[t]

    viable = 1
    for px in range(2):
        for py in range(2):
            for si in range(hr):
                rp = 0
                for sj in range(hs):
                    rp ^= syn_and[(px + 2 * si) % r, (py + 2 * sj) % s]
                if rp:
                    viable = 0
                    break
            if not viable:
                break
        if not viable:
            break

    if viable:
        syn = syn_and.copy()
    else:
        syn = syndromes[-1].copy()

    prep(syn, r, s)
    return solve(syn, r, s)
