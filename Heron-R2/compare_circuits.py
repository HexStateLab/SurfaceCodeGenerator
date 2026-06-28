"""
Per-sector share-pair layout with path embedding — transpile and compare.
"""
import json, os, sys
import numpy as np

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEMO_DIR)

def load_coupling():
    path = os.path.expanduser(
        "~/.local/lib/python3.12/site-packages/qiskit_ibm_runtime/"
        "fake_provider/backends/fez/conf_fez.json"
    )
    with open(path) as f:
        conf = json.load(f)
    for gate in conf["gates"]:
        if gate["name"] == "cz":
            cm = [(min(a,b), max(a,b)) for a,b in gate["coupling_map"] if a != b]
            return sorted(set(cm))
    raise ValueError("no cz gate found")

cm = load_coupling()
adj = {i: set() for i in range(156)}
for a,b in cm:
    adj[a].add(b)
    adj[b].add(a)

deg3 = set(q for q in range(156) if len(adj[q]) == 3)
deg2 = set(q for q in range(156) if len(adj[q]) == 2)

# All possible column paths
all_paths = []
for d0 in deg3:
    for a0 in adj[d0] & deg2:
        for d1 in adj[a0] & deg3:
            if d1 == d0: continue
            for a1 in adj[d1] & deg2:
                if a1 == a0: continue
                for d2 in adj[a1] & deg3:
                    if d2 in (d0, d1): continue
                    all_paths.append((d0, a0, d1, a1, d2))

print(f"Total paths: {len(all_paths)}")

# Greedy find ncols disjoint paths
def find_sector(used_nodes, ncols=4):
    cand = [i for i, p in enumerate(all_paths) 
            if not (set(p) & used_nodes)]
    if len(cand) < ncols:
        return []
    cand.sort(key=lambda i: len([j for j in cand 
                                  if set(all_paths[i]) & set(all_paths[j])]))
    def search(selected, used, remaining, depth):
        if depth == ncols:
            return selected
        remaining.sort(key=lambda i: len([j for j in remaining 
                                          if set(all_paths[i]) & set(all_paths[j])]))
        for i, idx in enumerate(remaining):
            path_set = set(all_paths[idx])
            if path_set & used:
                continue
            result = search(selected + [idx], used | path_set, 
                          remaining[:i] + remaining[i+1:], depth+1)
            if result:
                return result
        return None
    return search([], set(), cand, 0)

# Build sectors
used_global = set()
sectors = []
for ncols in [4, 4, 4, 2]:
    sector = find_sector(used_global, ncols)
    if sector and len(sector) == ncols:
        sector_nodes = set().union(*[set(all_paths[i]) for i in sector])
        used_global |= sector_nodes
        sectors.append([all_paths[i] for i in sector])
        print(f"Sector {len(sectors)} ({ncols} cols): {len(sector_nodes)} qubits")
        for p in sectors[-1]:
            print(f"  {p}")

print(f"\nTotal data deg3: {len([q for q in used_global if q in deg3])}/48")
print(f"Total anc deg2: {len([q for q in used_global if q in deg2])}/48")
print(f"Total qubits: {len(used_global)}/156")

# Build circuit with found sectors
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.transpiler import CouplingMap, Layout
from qiskit_ibm_runtime.fake_provider.backends.fez.fake_fez import FakeFez

backend = FakeFez()
cm_full = CouplingMap(couplinglist=[(a,b) for a,b in cm])

r, s = 6, 8
hr, hs = r // 2, s // 2

# Count ancillas: 2 per column per sector
n_anc = sum(2 * len(s) for s in sectors)
n_data = r * s  # 48
total = n_data + n_anc

print(f"\nCircuit: {n_data} data + {n_anc} anc = {total} qubits")

qr = QuantumRegister(total, "q")
cr_syn = ClassicalRegister(n_anc, "syn")
qc = QuantumCircuit(qr, cr_syn)

# Assign physical qubits to virtual indices
# Data: virtual 0..47 → physical deg3 nodes
# Anc: virtual 48..48+n_anc-1 → physical deg2 nodes
layout_dict = {}

# Map sector coords to virtual data indices
used_deg3_list = sorted([q for q in used_global if q in deg3])
used_deg2_list = sorted([q for q in used_global if q in deg2])

# Assign data qubits: first 48 virtual slots = data
# Use the found deg3 nodes, fill gaps with remaining deg3
all_deg3 = sorted(deg3)
remaining_deg3 = [q for q in all_deg3 if q not in used_deg3_list]

data_to_phys = {}
phys_to_data = {}
for vi in range(48):
    if vi < len(used_deg3_list):
        phys = used_deg3_list[vi]
    else:
        phys = remaining_deg3[vi - len(used_deg3_list)]
    data_to_phys[vi] = phys
    layout_dict[qr[vi]] = phys

# Assign ancillas
anc_virtual = 48
for si, sector in enumerate(sectors):
    for d0, a0, d1, a1, d2 in sector:
        layout_dict[qr[anc_virtual]] = a0
        anc_virtual += 1
        layout_dict[qr[anc_virtual]] = a1
        anc_virtual += 1

# Build circuit CX operations
# For each sector column:
#   anc[0] between data0 and data1: CX(d0, anc0), CX(d1, anc0)
#   anc[1] between data1 and data2: CX(d1, anc1), CX(d2, anc1)
anc_virtual = 48
for si, sector in enumerate(sectors):
    px = si // 2
    py = si % 2
    for qi, (d0_phys, a0_phys, d1_phys, a1_phys, d2_phys) in enumerate(sector):
        # Map physical qubits to virtual indices
        # Find which virtual index maps to each physical qubit
        vd0 = [vi for vi, p in data_to_phys.items() if p == d0_phys][0]
        vd1 = [vi for vi, p in data_to_phys.items() if p == d1_phys][0]
        vd2 = [vi for vi, p in data_to_phys.items() if p == d2_phys][0]
        
        # anc0: reset, CX(d0, anc), CX(d1, anc), measure
        qc.reset(anc_virtual)
        qc.cx(qr[vd0], qr[anc_virtual])
        qc.cx(qr[vd1], qr[anc_virtual])
        qc.measure(qr[anc_virtual], cr_syn[anc_virtual - 48])
        anc_virtual += 1
        
        # anc1: reset, CX(d1, anc), CX(d2, anc), measure
        qc.reset(anc_virtual)
        qc.cx(qr[vd1], qr[anc_virtual])
        qc.cx(qr[vd2], qr[anc_virtual])
        qc.measure(qr[anc_virtual], cr_syn[anc_virtual - 48])
        anc_virtual += 1

ops = qc.count_ops()
print(f"Virtual circuit: CX={ops.get('cx',0)}")

# Transpile with layout
layout = Layout(layout_dict)
qc_t = transpile(
    qc,
    backend=backend,
    coupling_map=cm_full,
    basis_gates=['cx', 'id', 'rz', 'sx', 'x'],
    optimization_level=3,
    initial_layout=layout,
    seed_transpiler=42,
)
ops_t = qc_t.count_ops()
print(f"Path-layout transpiled: phys={qc_t.num_qubits}, "
      f"depth={qc_t.depth()}, CX={ops_t.get('cx',0)}, "
      f"SWAP={ops_t.get('swap',0)}")

# Compare with standard flag circuit
from pw_qiskit import heavy_hex_flag_layout, build_flag_circuit
data_map, anc_maps, edges, total_q = heavy_hex_flag_layout(r, s)
qc_flag = build_flag_circuit(r, s, 1, data_map, anc_maps)
qc_flag_t = transpile(
    qc_flag,
    backend=backend,
    coupling_map=cm_full,
    basis_gates=['cx', 'id', 'rz', 'sx', 'x'],
    optimization_level=3,
    seed_transpiler=42,
)
ops_f = qc_flag_t.count_ops()
print(f"\nFlag circuit (1 round): {qc_flag.num_qubits}q, CX={qc_flag.count_ops().get('cx',0)}")
print(f"Flag transpiled: phys={qc_flag_t.num_qubits}, "
      f"depth={qc_flag_t.depth()}, CX={ops_f.get('cx',0)}, "
      f"SWAP={ops_f.get('swap',0)}")
