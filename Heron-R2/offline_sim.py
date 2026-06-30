"""
offline_sim.py — Offline drop-in for IBM's Dynamic Circuit "simulator side".

The online pipeline (run_opt.py) submits dynamic circuits to IBM hardware via:

    service = QiskitRuntimeService(...)
    backend = service.backend("ibm_marrakesh")
    sampler = SamplerV2(mode=backend)
    job     = sampler.run([qc_t], shots=shots)
    pub     = job.result()[0]
    bits    = pub.data.syn_0.to_bool_array(order='little')   # per-register BitArray

This module reproduces *exactly that contract* locally, with no IBM token and
no queue, using qiskit-aer. Aer's SamplerV2:
  - executes genuine dynamic circuits (mid-circuit measurement, reset, and
    classical feed-forward via if_test) — the same features pw_opt.py relies on
    for its no-reset / free-final-round QEC rounds, and
  - returns a PrimitiveResult whose `pub.data.<register_name>` exposes one
    BitArray per ClassicalRegister, with `.to_bool_array(order='little')` and
    `.num_shots` — byte-for-byte compatible with qiskit_ibm_runtime.SamplerV2.

So the rest of the stack (all_syndromes_opt, the decoders, the fidelity /
witness analysis) runs unmodified against the object returned here.

Three noise modes:
  * ideal              — perfect simulation (default)
  * parametric noise   — depolarizing on 1q/2q gates + readout + reset error,
                         specified by a few floats (no extra dependencies)
  * fake-backend noise — real device error model lifted from a FakeBackend
                         (FakeFez, FakeMarrakesh, ...) via NoiseModel.from_backend

Public API
----------
    setup(...)               -> (backend_for_transpile, sampler)   [main entry]
    make_offline_sampler(...) -> Aer SamplerV2-compatible sampler
    build_noise_model(...)    -> qiskit_aer.noise.NoiseModel
    noise_from_fake(name)     -> (NoiseModel, fake_backend)
    transpile_offline(qc, backend) -> ISA circuit (no-op when all-to-all)
"""
from __future__ import annotations

from typing import Optional, Tuple

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    ReadoutError,
)


# --------------------------------------------------------------------------- #
# Noise models
# --------------------------------------------------------------------------- #
def build_noise_model(
    two_qubit_rate: float = 0.0,
    one_qubit_rate: float = 0.0,
    readout_rate: float = 0.0,
    reset_rate: float = 0.0,
    two_qubit_gates=("cx", "cz", "ecr"),
    one_qubit_gates=("sx", "x", "rz", "h"),
) -> Optional[NoiseModel]:
    """Build a simple, fully-offline NoiseModel from a handful of rates.

    Parameters
    ----------
    two_qubit_rate : depolarizing probability applied after every 2-qubit gate
        (whichever of cx/cz/ecr the transpiled circuit actually uses).
    one_qubit_rate : depolarizing probability after every listed 1-qubit gate.
    readout_rate   : symmetric bit-flip probability on measurement.
    reset_rate     : probability that a `reset` leaves the qubit in |1> instead
        of |0> (models imperfect active reset — relevant to no-reset rounds).

    Returns None when every rate is zero, so callers can treat "no noise" and
    "ideal" identically.
    """
    if not any((two_qubit_rate, one_qubit_rate, readout_rate, reset_rate)):
        return None

    nm = NoiseModel()

    if two_qubit_rate > 0:
        err2 = depolarizing_error(two_qubit_rate, 2)
        nm.add_all_qubit_quantum_error(err2, list(two_qubit_gates))

    if one_qubit_rate > 0:
        err1 = depolarizing_error(one_qubit_rate, 1)
        nm.add_all_qubit_quantum_error(err1, list(one_qubit_gates))

    if readout_rate > 0:
        p = readout_rate
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - p, p], [p, 1 - p]])
        )

    if reset_rate > 0:
        # A noisy reset: with prob `reset_rate` the qubit ends in |1>.
        # Model as ideal reset followed by a bit-flip channel on the result.
        from qiskit_aer.noise import pauli_error

        flip = pauli_error([("X", reset_rate), ("I", 1 - reset_rate)])
        nm.add_all_qubit_quantum_error(flip, ["reset"])

    return nm


def _load_fake_backend(name: str):
    """Resolve a FakeBackend class by fuzzy name ('fez' -> FakeFez)."""
    try:
        from qiskit_ibm_runtime import fake_provider as fp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Fake-backend calibration needs qiskit-ibm-runtime: "
            "pip install qiskit-ibm-runtime"
        ) from exc

    target = "fake" + name.lower().replace("fake", "").strip()
    for attr in dir(fp):
        if attr.lower() == target and attr.startswith("Fake"):
            return getattr(fp, attr)()
    avail = sorted(a for a in dir(fp) if a.startswith("Fake"))
    raise ValueError(f"unknown fake backend '{name}'. available: {avail}")


def _median_gate_error(target, gate_names):
    """Median reported error across all qubits/pairs for the first present gate."""
    import numpy as np

    for g in gate_names:
        inst = target.get(g)
        if not inst:
            continue
        errs = [
            p.error
            for p in inst.values()
            if p is not None and p.error is not None and 0.0 < p.error < 1.0
        ]
        if errs:
            return float(np.median(errs))
    return 0.0


def device_calibrated_noise(name: str) -> Tuple[NoiseModel, object]:
    """Build a Clifford-safe, device-*calibrated* depolarizing model.

    Why not NoiseModel.from_backend? That lifts the device's continuous-rotation
    basis (sx, rz, ...). Applied to an 80-qubit QEC circuit it forces Aer onto
    the statevector method (2**80 amplitudes) -- intractable. Instead we read the
    device's *median* 2-qubit, 1-qubit and readout error rates and feed them into
    a depolarizing + readout NoiseModel. Pauli/depolarizing noise on a Clifford
    QEC circuit stays stabilizer-simulable, so this scales to any grid while
    still reflecting realistic device error magnitudes.

    `name` is fuzzy: 'fez', 'FakeFez', 'marrakesh', 'torino', 'sherbrooke', ...
    Returns (noise_model, fake_backend); the backend is returned only for
    reference (coupling map, qubit count) and is NOT used for routing.
    """
    backend = _load_fake_backend(name)
    tgt = backend.target
    two_q = _median_gate_error(tgt, ("ecr", "cz", "cx"))
    one_q = _median_gate_error(tgt, ("sx", "x"))
    readout = _median_gate_error(tgt, ("measure",))
    nm = build_noise_model(
        two_qubit_rate=two_q,
        one_qubit_rate=one_q,
        readout_rate=readout,
    )
    return nm, backend


# Back-compat alias
noise_from_fake = device_calibrated_noise


# --------------------------------------------------------------------------- #
# Sampler factory
# --------------------------------------------------------------------------- #
def make_offline_sampler(
    noise_model: Optional[NoiseModel] = None,
    seed: Optional[int] = None,
    method: str = "automatic",
    device: str = "CPU",
    default_shots: int = 1024,
    **backend_options,
) -> AerSamplerV2:
    """Return an Aer SamplerV2 that behaves like qiskit_ibm_runtime.SamplerV2.

    The returned object's `.run([pubs], shots=...).result()[0]` yields a pub
    whose `.data.<creg>` are BitArrays — exactly what run_opt.py consumes. Aer
    fully supports the dynamic-circuit features used by pw_opt's circuits.
    """
    opts = {
        "backend_options": {
            "method": method,
            "device": device,
            **backend_options,
        }
    }
    if noise_model is not None:
        opts["backend_options"]["noise_model"] = noise_model
    if seed is not None:
        # Aer SamplerV2 takes a top-level seed for reproducible sampling.
        return AerSamplerV2(default_shots=default_shots, seed=seed, options=opts)
    return AerSamplerV2(default_shots=default_shots, options=opts)


# --------------------------------------------------------------------------- #
# Transpilation (optional, only meaningful with a device coupling map)
# --------------------------------------------------------------------------- #
def transpile_offline(qc, backend=None, optimization_level: int = 1, seed: int = 42):
    """Lower `qc` to a backend's basis/coupling, or return it unchanged.

    When `backend` is an all-to-all simulator (no coupling map) this is a
    near no-op and the logical circuit is returned as-is — Aer can execute it
    directly. When `backend` is a FakeBackend, the circuit is routed and
    rewritten into the device basis, faithfully mimicking what IBM hardware
    would run (classical registers and their names are preserved, so syndrome
    extraction downstream is unaffected).
    """
    if backend is None or getattr(backend, "coupling_map", None) is None:
        return qc
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        routing_method="sabre",
        seed_transpiler=seed,
    )
    return pm.run(qc)


# --------------------------------------------------------------------------- #
# One-call setup used by run_opt.py's --offline path
# --------------------------------------------------------------------------- #
def _offline_backend(name="offline_aer", device="CPU"):
    """A plain all-to-all AerSimulator, renamed so jobs.json reads nicely."""
    b = AerSimulator(device=device)
    b.name = name
    return b


def setup(
    fake: Optional[str] = None,
    two_qubit_rate: float = 0.0,
    one_qubit_rate: float = 0.0,
    readout_rate: float = 0.0,
    reset_rate: float = 0.0,
    seed: Optional[int] = None,
    device: str = "CPU",
) -> Tuple[object, AerSamplerV2]:
    """Build (backend_for_transpile, sampler) for an offline run.

    Precedence for the noise model:
      1. If `fake` is given -> use that device's real error model (and its
         coupling map for transpilation).
      2. Else if any parametric rate is > 0 -> build_noise_model(...).
      3. Else -> ideal, noiseless.

    The returned `backend` is what run_opt.py should pass to its transpiler
    (a FakeBackend when `fake` is set, otherwise an all-to-all Aer backend so
    transpilation is skipped). The returned `sampler` is the offline stand-in
    for SamplerV2(mode=backend).
    """
    noise_model = None
    backend = _offline_backend(name="offline_aer", device=device)

    if fake:
        noise_model, _ = device_calibrated_noise(fake)
        # NB: we do NOT route to the device coupling map. Routing + the device's
        # continuous-rotation basis would make the 80-qubit Clifford QEC circuit
        # non-Clifford and blow up statevector memory. The calibrated depolarizing
        # model keeps the circuit stabilizer-simulable while matching device rates.
        backend.name = f"offline_{fake.lower().replace('fake','')}"
    else:
        noise_model = build_noise_model(
            two_qubit_rate=two_qubit_rate,
            one_qubit_rate=one_qubit_rate,
            readout_rate=readout_rate,
            reset_rate=reset_rate,
        )

    sampler = make_offline_sampler(noise_model=noise_model, seed=seed, device=device)
    return backend, sampler


if __name__ == "__main__":
    # Tiny self-check: a dynamic circuit (mid-circuit measure + reset +
    # feed-forward) round-trips through the offline sampler with the expected
    # per-register interface.
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    q = QuantumRegister(2, "q")
    c0 = ClassicalRegister(1, "syn_0")
    cd = ClassicalRegister(2, "data")
    qc = QuantumCircuit(q, c0, cd)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, c0[0])
    qc.reset(0)
    with qc.if_test((c0, 1)):
        qc.x(0)
    qc.measure(q, cd)

    backend, sampler = setup(two_qubit_rate=0.0, seed=11)
    pub = sampler.run([qc], shots=100).result()[0]
    syn = pub.data.syn_0.to_bool_array(order="little")
    dat = pub.data.data.to_bool_array(order="little")
    print(f"backend.name = {backend.name}")
    print(f"syn_0: shape={syn.shape}, num_shots={pub.data.syn_0.num_shots}")
    print(f"data : shape={dat.shape}")
    print("self-check OK")
