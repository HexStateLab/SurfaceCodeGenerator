"""
pw_qiskit.py — Plane-Warp decoder integration with Qiskit.

Provides:
  PlaneWarp   — ctypes wrapper for libplane_warp.so
  QECBuilder  — builds (1+x²)(1+y²) code circuits for Qiskit
  decode_run  — single-call: build circuit → run on backend → decode → return

Usage (simulator):
  from pw_qiskit import decode_run
  correction = decode_run(6, 6, rounds=5, shots=1, backend=None)

Usage (IBM hardware, needs IBM Quantum account):
  from qiskit_ibm_runtime import QiskitRuntimeService
  service = QiskitRuntimeService()
  backend = service.backend("ibm_brisbane")
  correction = decode_run(6, 6, rounds=5, shots=1, backend=backend)
"""

import ctypes
import os
import struct
import numpy as np

# ---------------------------------------------------------------------------
# Load the shared library
# ---------------------------------------------------------------------------
_lib_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_lib_dir, "libplane_warp.so")
_lib = ctypes.CDLL(_lib_path)

# --- set argument / return types for each exported C function ---

# void preprocess_syndrome(int r, int s, uint8_t *syn)
_lib.preprocess_syndrome.argtypes = [ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_uint8)]
_lib.preprocess_syndrome.restype = None

# void syndrome_of(int r, int s, uint8_t *err, uint8_t *syn)
_lib.syndrome_of.argtypes = [ctypes.c_int, ctypes.c_int,
                             ctypes.POINTER(ctypes.c_uint8),
                             ctypes.POINTER(ctypes.c_uint8)]
_lib.syndrome_of.restype = None

# int solve_plane(int r, int s, uint8_t *syn, uint8_t *out)
_lib.solve_plane.argtypes = [ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_uint8),
                              ctypes.POINTER(ctypes.c_uint8)]
_lib.solve_plane.restype = ctypes.c_int

# int solve_plane_layered(int r, int s, uint8_t *syn, uint8_t *out)
_lib.solve_plane_layered.argtypes = [ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_uint8),
                                      ctypes.POINTER(ctypes.c_uint8)]
_lib.solve_plane_layered.restype = ctypes.c_int

# void canonicalize(int r, int s, uint8_t *corr)
_lib.canonicalize.argtypes = [ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_uint8)]
_lib.canonicalize.restype = None

# int is_stabilizer(int r, int s, uint8_t *diff)
_lib.is_stabilizer.argtypes = [ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_uint8)]
_lib.is_stabilizer.restype = ctypes.c_int

# int decode_Z(int r, int s, uint8_t *err_z, uint8_t *dec_z)
_lib.decode_Z.argtypes = [ctypes.c_int, ctypes.c_int,
                          ctypes.POINTER(ctypes.c_uint8),
                          ctypes.POINTER(ctypes.c_uint8)]
_lib.decode_Z.restype = ctypes.c_int

# int solve_plane_fast(int r, int s, uint8_t *syn, uint8_t *out)
_lib.solve_plane_fast.argtypes = [ctypes.c_int, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_uint8),
                                   ctypes.POINTER(ctypes.c_uint8)]
_lib.solve_plane_fast.restype = ctypes.c_int


class PlaneWarp:
    """ctypes wrapper for the plane_warp decoder."""

    def __init__(self, fast=False, singleshot=True, escape=True,
                 weight_cap=0, cap_auto_rate=0.0):
        self.fast = fast
        # Set global knobs via ctypes (they're extern int/double in the .so)
        try:
            g_fast = ctypes.c_int.in_dll(_lib, "g_fast")
            g_fast.value = 1 if fast else 0
        except ValueError:
            pass
        try:
            g_ss = ctypes.c_int.in_dll(_lib, "g_singleshot")
            g_ss.value = 1 if singleshot else 0
        except ValueError:
            pass
        try:
            g_esc = ctypes.c_int.in_dll(_lib, "g_escape_enabled")
            g_esc.value = 1 if escape else 0
        except ValueError:
            pass
        try:
            g_wc = ctypes.c_int.in_dll(_lib, "g_weight_cap")
            g_wc.value = weight_cap
        except ValueError:
            pass
        try:
            g_car = ctypes.c_double.in_dll(_lib, "g_cap_auto_rate")
            g_car.value = cap_auto_rate
        except ValueError:
            pass

    @staticmethod
    def _to_ptr(arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    def syndrome_of(self, error):
        r, s = error.shape
        syn = np.zeros((r, s), dtype=np.uint8)
        _lib.syndrome_of(r, s, self._to_ptr(error), self._to_ptr(syn))
        return syn

    def preprocess(self, syndrome):
        r, s = syndrome.shape
        _lib.preprocess_syndrome(r, s, self._to_ptr(syndrome))

    def decode(self, syndrome):
        r, s = syndrome.shape
        out = np.zeros((r, s), dtype=np.uint8)
        ok = _lib.solve_plane(r, s, self._to_ptr(syndrome), self._to_ptr(out))
        return out, bool(ok)

    def decode_layered(self, syndrome):
        r, s = syndrome.shape
        out = np.zeros((r, s), dtype=np.uint8)
        ok = _lib.solve_plane_layered(r, s, self._to_ptr(syndrome),
                                      self._to_ptr(out))
        return out, bool(ok)

    def decode_fast(self, syndrome):
        r, s = syndrome.shape
        out = np.zeros((r, s), dtype=np.uint8)
        ok = _lib.solve_plane_fast(r, s, self._to_ptr(syndrome),
                                   self._to_ptr(out))
        return out, bool(ok)

    def canonicalize(self, correction):
        r, s = correction.shape
        _lib.canonicalize(r, s, self._to_ptr(correction))

    def is_stabilizer(self, diff):
        r, s = diff.shape
        return bool(_lib.is_stabilizer(r, s, self._to_ptr(diff)))

    def decode_tesseract(self, syndromes):
        """Multi-round tesseract decode.

        syndromes : ndarray (rounds, r, s) of uint8
        returns   : ndarray (r, s) of uint8 — the consensus correction
        """
        rounds, r, s = syndromes.shape
        n = r * s
        buf = struct.pack('<I', rounds)
        buf += syndromes.tobytes()
        # Write to pipe to ./plane_warp --decode-tesseract
        import subprocess
        exe = os.path.join(_lib_dir, "plane_warp")
        if not os.path.exists(exe):
            # build it if needed
            pass
        flags = ["--fast"] if self.fast else []
        p = subprocess.run([exe, str(r), str(s)] + flags +
                           ["--decode-tesseract"],
                           input=buf, capture_output=True)
        corr = np.frombuffer(p.stdout, dtype=np.uint8).reshape(r, s)
        return corr


# =========================================================================
# Qiskit circuit builder for the (1+x²)(1+y²) code
# =========================================================================

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.transpiler import CouplingMap
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False


class QECBuilder:
    """Builds syndrome-extraction circuits for the (1+x²)(1+y²) toric code.

    Layout (r×s data qubits + r×s ancilla qubits):
      data[i,j]   = qubit at index  i*s + j          (0 … N-1)
      anc[i,j]    = qubit at index  N + i*s + j       (N … 2N-1)
    where N = r*s.

    The stabiliser at (i,j) measures X or Z parity of the four data qubits:
      (i,j), (i+2,j), (i,j+2), (i+2,j+2)   (all mod r, s).
    """

    def __init__(self, r, s, code="z"):
        assert r >= 4 and s >= 4, "minimum 4×4 grid"
        assert r % 2 == 0 and s % 2 == 0, "even dimensions required"
        self.r = r
        self.s = s
        self.n = r * s
        # Z-type stabiliser: uses H-CNOT-H → Z-basis measurement
        self.code = code  # 'z' or 'x'

    @property
    def num_qubits(self):
        return 2 * self.n

    def _di(self, i):
        return (i + 2) % self.r if i + 2 < self.r else i + 2 - self.r

    def _dj(self, j):
        return (j + 2) % self.s if j + 2 < self.s else j + 2 - self.s

    def _qdata(self, i, j):
        return i * self.s + j

    def _qanc(self, i, j):
        return self.n + i * self.s + j

    def build_round_circuit(self, barrier=True):
        """Return a QuantumCircuit that performs one syndrome-extraction round.

        Measures Z⊗Z⊗Z⊗Z stabilisers (detects X errors):

          anc: |0⟩ -- * -- * -- * -- * -- M
                       |    |    |    |
          data:       d0   d1   d2   d3

        CNOT: data → ancilla (control = data, target = anc).
        """
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(self.n, "syn")
        qc = QuantumCircuit(qr, cr)
        for i in range(self.r):
            for j in range(self.s):
                a = self._qanc(i, j)
                qc.cx(self._qdata(i, j), a)
                qc.cx(self._qdata(self._di(i), j), a)
                qc.cx(self._qdata(i, self._dj(j)), a)
                qc.cx(self._qdata(self._di(i), self._dj(j)), a)
        if barrier:
            qc.barrier()
        for i in range(self.r):
            for j in range(self.s):
                qc.measure(self._qanc(i, j), cr[i * self.s + j])
        return qc

    def build_rounds_circuit(self, rounds, barrier=True):
        """Return a QuantumCircuit that performs *rounds* syndrome-extraction
        rounds, measuring all ancillas at the end of each round.

        The classical registers are labelled syn_c for round c.
        """
        qr = QuantumRegister(self.num_qubits, "q")
        crs = [ClassicalRegister(self.n, f"syn_{c}") for c in range(rounds)]
        qc = QuantumCircuit(qr, *crs)
        for rnd in range(rounds):
            for i in range(self.r):
                for j in range(self.s):
                    a = self._qanc(i, j)
                    qc.cx(self._qdata(i, j), a)
                    qc.cx(self._qdata(self._di(i), j), a)
                    qc.cx(self._qdata(i, self._dj(j)), a)
                    qc.cx(self._qdata(self._di(i), self._dj(j)), a)
                    qc.measure(a, crs[rnd][i * self.s + j])
                    qc.reset(a)
            if barrier:
                qc.barrier()

        return qc

    def syndrome_from_counts(self, counts, rounds):
        """Convert Qiskit measurement counts into a (rounds, r, s) uint8 array.

        counts : dict mapping bitstring → count from backend.run()
                 Qiskit returns space-separated per-register blocks:
                   'syn_0_bits syn_1_bits ... syn_{rounds-1}_bits'
                 Each block has N bits, rightmost = creg bit 0.
        """
        n = self.n
        syndromes = np.zeros((rounds, self.r, self.s), dtype=np.uint8)
        for bitstring, cnt in counts.items():
            blocks = bitstring.split(' ')  # one block per round
            for c in range(rounds):
                blk = blocks[c]  # syn_c block
                for q in range(n):
                    # Within a block, bitstring[-(q+1)] = creg bit q
                    if blk[-(q + 1)] == '1':
                        syndromes[c, q // self.s, q % self.s] = 1
            # Use first bitstring only (or could weight by count)
            break
        return syndromes


# =========================================================================
# High-level integration
# =========================================================================

def decode_run(r, s, rounds=5, shots=1, backend=None,
               pw_kwargs=None, builder_kwargs=None):
    """One-call: build circuits, run, decode, return correction.

    Parameters
    ----------
    r, s : int — grid dimensions (must be even, ≥4)
    rounds : int — number of QEC rounds
    shots : int — number of measurement shots (default 1)
    backend : Qiskit backend or None (uses AerSimulator)
    pw_kwargs : dict — passed to PlaneWarp()
    builder_kwargs : dict — passed to QECBuilder()

    Returns
    -------
    correction : ndarray (r, s) of uint8 — the decoded data-qubit correction
    raw_syndromes : ndarray (rounds, r, s) of uint8 — the measured syndromes
    """
    if pw_kwargs is None:
        pw_kwargs = {}
    if builder_kwargs is None:
        builder_kwargs = {}

    builder = QECBuilder(r, s, **builder_kwargs)
    pw = PlaneWarp(**pw_kwargs)

    # Build circuit
    qc = builder.build_rounds_circuit(rounds)

    # Run
    if backend is None:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Convert counts → syndrome array
    syn = builder.syndrome_from_counts(counts, rounds)

    # Decode
    correction = pw.decode_tesseract(syn)

    return correction, syn
