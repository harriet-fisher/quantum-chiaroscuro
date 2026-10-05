"""Plan QDrive (qdrive-api-v1) jobs for a calibrated Standing Light scene. Pure planning: nothing here talks to Moth.

Why this exists (docs/qdrive-field-notes.md, docs/qdrive-solution-design.md; paper arXiv 2605.22744, Motte model):
  - One QDrive target is ONE gate on its own qubit group; all words for that group go in one `expvals` mapping, identity-padded.
  - A `null` entry ends a layer. Targets that overlap in the same layer fight (the engine appears to derive every gate of a
    layer from the state at the layer's start), so overlapping groups go in SUCCESSIVE layers, i.e. a `null` between them.
  - Cost grows with target body size (3 is fine, 8 timed out) and with register width (every 19-qubit job failed), so each
    independent component of the coupling graph is its own small job and the circuits are merged locally.
  - Targets are pushed gradually: rounds repeat the sweep, and the closed loop below repeats only the stubborn groups.

Moment providers: TargetMoments (singles + pair ZZ from calib/targets.json) or OracleMoments (exact, any order, from the scene's
oracle state; gives exact 3-body words so a triangle target is consistent by construction).
"""
import itertools
from dataclasses import dataclass, field

import numpy as np

MAX_GROUP = 3
CLIP = 0.98                      # never ask for |value| = 1 from a correction step


# ---------------------------------------------------------------------------------------------------------- moments
class TargetMoments:
    """<Z_q> and <Z_a Z_b> from the calibrated targets (bloch / relationships rows). Higher orders are unknown (None)."""

    def __init__(self, bloch, rel):
        self.s = {int(b["qubit"]): float(b["Z"]) for b in bloch}
        self.p = {tuple(sorted(r["qubits"])): float(r["ZZ"]) for r in rel}

    def pairs(self):
        return sorted(self.p)

    def qubits(self):
        return sorted(self.s)

    def __call__(self, qubits):
        q = tuple(sorted(qubits))
        return self.s.get(q[0]) if len(q) == 1 else self.p.get(q) if len(q) == 2 else None


class OracleMoments:
    """Exact Z-word moments of a probability distribution over n qubits (little-endian: qubit q is bit q)."""

    def __init__(self, psi_or_probs, n, pairs):
        p = np.abs(np.asarray(psi_or_probs)) ** 2 if np.iscomplexobj(psi_or_probs) else np.asarray(psi_or_probs, float)
        self.p = p / p.sum(); self.n = n; self._pairs = sorted(tuple(sorted(x)) for x in pairs)
        self._idx = np.arange(len(self.p))

    def pairs(self):
        return list(self._pairs)

    def qubits(self):
        return list(range(self.n))

    def __call__(self, qubits):
        sign = np.ones(len(self.p))
        for q in qubits:
            sign = sign * (1 - 2 * ((self._idx >> q) & 1))
        return float(self.p @ sign)


# ---------------------------------------------------------------------------------------------------------- structure
def components(qubits, pairs):
    """Connected components of the target graph (each is an independent state: product across components)."""
    parent = {q: q for q in qubits}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    out = {}
    for q in qubits:
        out.setdefault(find(q), []).append(q)
    return sorted((sorted(v) for v in out.values()), key=lambda v: v[0])


def _cliques(comp, pairs, k):
    adj = {q: set() for q in comp}
    for a, b in pairs:
        if a in adj and b in adj:
            adj[a].add(b); adj[b].add(a)
    out = [(q,) for q in comp]
    for size in range(2, k + 1):
        for c in itertools.combinations(comp, size):
            if all(y in adj[x] for x, y in itertools.combinations(c, 2)):
                out.append(c)
    return out


def cover_groups(comp, pairs, max_k=MAX_GROUP):
    """Greedy cover of every target pair by cliques of at most max_k qubits (fewest, biggest groups first). Qubits that appear in
    no pair get a 1-qubit group so their <Z> is still targeted."""
    pairs = {tuple(sorted(p)) for p in pairs if p[0] in comp and p[1] in comp}
    cl = _cliques(comp, pairs, max_k)
    uncovered, groups = set(pairs), []
    while uncovered:
        def gain(c):
            return sum(1 for e in itertools.combinations(c, 2) if e in uncovered)
        best = max(cl, key=lambda c: (gain(c), len(c), tuple(-x for x in c)))
        if gain(best) == 0:
            break
        groups.append(best)
        uncovered -= set(itertools.combinations(best, 2))
    seen = {q for g in groups for q in g}
    groups += [(q,) for q in comp if q not in seen]
    return groups


def order_groups(groups):
    """Growth order, like the GHZ example in the paper: start at the lowest group, then always take the group sharing the most
    already-covered qubits (so every new gate extends what is already built), ties by qubit index."""
    left = sorted(groups)
    done, covered = [], set()
    while left:
        nxt = max(left, key=lambda g: (len(covered & set(g)), -min(g), tuple(-x for x in g)) if covered else (0, -min(g), tuple(-x for x in g)))
        left.remove(nxt); done.append(nxt); covered |= set(nxt)
    return done


def pack_layers(ordered):
    """Consecutive packing: a group joins the current layer only if it is disjoint from every group already in it; otherwise it
    starts the next layer. Result: no layer ever contains two overlapping targets."""
    layers, cur = [], []
    for g in ordered:
        if any(set(g) & set(h) for h in cur):
            layers.append(cur); cur = []
        cur.append(g)
    if cur:
        layers.append(cur)
    return layers


def group_target(group, moment):
    """One target for a qubit group: every Z-word over every non-empty subset of the group whose moment is known,
    identity-padded, letter i acting on group[i]."""
    words = {}
    for size in range(1, len(group) + 1):
        for sub in itertools.combinations(range(len(group)), size):
            v = moment([group[i] for i in sub])
            if v is None:
                continue
            words["".join("Z" if i in sub else "I" for i in range(len(group)))] = round(float(np.clip(v, -CLIP, CLIP)), 4)
    return dict(qubits=list(group), expvals=words)



# ---------------------------------------------------------------------------------------------------------- growth plan
def elimination(comp, pairs):
    """Min-degree elimination of the target graph restricted to comp. Returns (growth_order, parents, fill_edges): growth_order is the
    reverse elimination order; parents[v] are v's neighbours that come EARLIER in the growth order in the filled graph (a clique),
    so group(v) = parents[v] + [v] has size 1 + treewidth at most. fill_edges are edges added by triangulation (not target pairs)."""
    adj = {q: set() for q in comp}
    for a, b in pairs:
        if a in adj and b in adj:
            adj[a].add(b); adj[b].add(a)
    left, elim, nbrs, fill = set(comp), [], {}, set()
    while left:
        v = min(left, key=lambda q: (len(adj[q] & left), q))
        nb = sorted(adj[v] & left)
        for a, b in itertools.combinations(nb, 2):
            if b not in adj[a]:
                adj[a].add(b); adj[b].add(a); fill.add((a, b))
        elim.append(v); nbrs[v] = nb; left.discard(v)
    return elim[::-1], nbrs, sorted(fill)


def growth_groups(comp, pairs):
    """[(group, new_qubit)] in growth order: each group is the new qubit plus its already-built neighbours. A new qubit starts in |0>,
    so the gate for its group only has to set its conditional distribution: exactly reachable, no entanglement mismatch."""
    order, parents, fill = elimination(comp, pairs)
    return [(tuple(sorted(parents[v] + [v])), v) for v in order], fill


def build_growth_job(comp, pairs, moment, rounds=1, seed=7, update_method="spectral", shots=1024):
    """Sequential-growth job: one group per qubit, consecutive groups that overlap are separated by a `null` (a layer boundary).
    Fill edges from triangulation are added to coupling_map (virtual edges on the simulator; they are not target pairs)."""
    gg, fill = growth_groups(comp, pairs)
    groups = [g for g, _ in gg]
    layers = pack_layers(groups)
    edges = sorted({tuple(sorted(p)) for p in pairs if p[0] in comp and p[1] in comp} | set(fill))
    targets = []
    for _ in range(rounds):
        for layer in layers:
            for g in layer:
                targets.append(dict(qubits=[_local(comp, q) for q in g], expvals=group_target(g, moment)["expvals"]))
            targets.append(None)
    params = dict(machine="aer", n_qubits=len(comp), coupling_map=[[_local(comp, a), _local(comp, b)] for a, b in edges],
                  targets=targets, update_method=update_method, shots=shots, tomography=0, sample=False, seed=seed)
    return Job(qubits=list(comp), layers=layers, params=params, rounds=rounds)


def markov_tv(prob, comp, n, growth):
    """Total-variation distance between a distribution over n qubits and the sequential-growth product
    prod_v P(s_v | s_parents(v)) built ONLY from the group marginals. 0 means the group targets of the growth plan determine the
    distribution exactly (the graph's Markov structure is captured)."""
    p = np.asarray(prob, float); p = p / p.sum()
    idx = np.arange(len(p))
    def marg(qs):
        key = np.zeros(len(p), int)
        for i, q in enumerate(qs):
            key |= ((idx >> q) & 1) << i
        out = np.zeros(2 ** len(qs)); np.add.at(out, key, p); return out, key
    # restrict to comp: other qubits are marginalised away
    q = np.ones(len(p))
    for g, v in growth:
        parents = [x for x in g if x != v]
        num, knum = marg(list(parents) + [v])
        den, kden = marg(parents) if parents else (np.array([1.0]), np.zeros(len(p), int))
        q = q * np.where(den[kden] > 0, num[knum] / np.where(den[kden] > 0, den[kden], 1), 0.0)
    return 0.5 * float(np.abs(p - q).sum()) if len(comp) == n else None

def prefix_states(comp, growth, prob):
    """The partially built state after each growth step, on the component's local qubits (qubit comp[i] = bit i), amplitudes
    sqrt(prod_{j<=k} P(s_j | s_parents(j))) with not-yet-grown qubits in |0>. `prob` is the distribution over the component's local
    qubits. Because the graph's Markov structure makes the full product exact (markov_tv = 0), the last prefix state is the oracle state.
    Yields (group_local, new_local, psi_k)."""
    p = np.asarray(prob, float); p = p / p.sum(); nloc = len(comp)
    idx = np.arange(len(p))
    def marg(qs):
        key = np.zeros(len(p), int)
        for i, q in enumerate(qs):
            key |= ((idx >> q) & 1) << i
        out = np.zeros(2 ** len(qs)); np.add.at(out, key, p); return out, key
    cond = []                                               # per step: P(s_v | s_parents) as a function of the basis index
    for g, v in growth:
        parents = [x for x in g if x != v]
        num, kn = marg(parents + [v]); den, kd = marg(parents) if parents else (np.array([1.0]), np.zeros(len(p), int))
        cond.append(np.where(den[kd] > 0, num[kn] / np.where(den[kd] > 0, den[kd], 1), 0.0))
    done = []
    q = np.ones(len(p))
    for k, ((g, v), c) in enumerate(zip(growth, cond)):
        q = q * c
        done.append(v)
        mask = np.zeros(len(p), bool)                       # basis states whose not-yet-grown qubits are all 0
        later = [w for _, w in growth[k + 1:]]
        ok = np.ones(len(p), bool)
        for w in later:
            ok &= ((idx >> w) & 1) == 0
        # probability over grown qubits only: sum over later qubits is trivial since they sit in |0>; renormalise on the slice
        psi = np.where(ok, np.sqrt(q), 0.0).astype(complex)
        nrm = np.linalg.norm(psi)
        yield g, v, psi / nrm


def _pauli_product(a, b):
    """Product word of two Pauli words, ignoring the phase (X*Z ~ Y etc.)."""
    mul = {("I", "I"): "I", ("I", "X"): "X", ("I", "Y"): "Y", ("I", "Z"): "Z", ("X", "I"): "X", ("Y", "I"): "Y", ("Z", "I"): "Z",
           ("X", "X"): "I", ("Y", "Y"): "I", ("Z", "Z"): "I", ("X", "Y"): "Z", ("Y", "X"): "Z", ("X", "Z"): "Y", ("Z", "X"): "Y",
           ("Y", "Z"): "X", ("Z", "Y"): "X"}
    return "".join(mul[(x, y)] for x, y in zip(a, b))


def rdm_words(rho, k, tol=1e-3, sparse=True):
    """Pauli words of a k-qubit reduced density matrix as {word: value}, letter i = group[i], values clipped to +/-CLIP.
    Words with |value| < tol are dropped EXCEPT Z-type ones (only I/Z letters), kept even at 0 (the lab's working target T07 stated
    <Z> = 0 explicitly). sparse=True also drops stabiliser-implied words: when two kept words both have |value| = 1 their product word
    is determined (a pure Bell state's YY follows from ZZ and XX), and a target that spells out a whole pure state (a rank-1 RDM) is
    exactly what stalled the engine in live rung R1 (c53afb2a). Lab jobs that completed gave partial word sets (T04: ZZ and XX only)."""
    from src.quantum.motte_mock import pauli
    raw = {}
    for w in itertools.product("IXYZ", repeat=k):
        if set(w) == {"I"}:
            continue
        v = float(np.real(np.trace(pauli("".join(w)) @ rho)))
        if abs(v) >= tol or set(w) <= {"I", "Z"}:
            raw["".join(w)] = v
    if sparse:
        stab = []
        for w, v in sorted(raw.items(), key=lambda kv: (-abs(kv[1]), "Y" in kv[0], kv[0])):   # keep Z/X words, drop the Y word
            if abs(v) < 0.999:
                break
            if any(_pauli_product(a, b) == w for i, a in enumerate(stab) for b in stab[i + 1:]):
                del raw[w]
            else:
                stab.append(w)
    return {w: round(float(np.clip(v, -CLIP, CLIP)), 4) for w, v in raw.items()}


@dataclass
class StatePrepPlan:
    """The layers of a sequential state preparation, before they are cut into jobs. Everything is in the component's LOCAL qubit
    indices (qubit comp[i] is local i). layers[j] is the list of targets of layer j (no two overlap); layer_psi[j] is the pure state
    the component should be in once layer j has been applied (qubits not yet grown are still |0>), so a returned circuit can be
    scored after ANY layer, which is what lets a job be cut into rounds."""
    comp: list
    layers: list
    layer_psi: list
    edges: list                        # local coupling map, target pairs plus triangulation fill edges

    @property
    def n(self):
        return len(self.comp)


def plan_state_prep(comp, pairs, prob, tol=1e-3, keep_singletons=False):
    """Growth plan with full-reduced-density-matrix targets (see build_state_prep_job), kept as layers with the target state after each."""
    from src.quantum.motte_mock import rdm
    gg, fill = growth_groups(comp, pairs)
    loc = [(tuple(_local(comp, q) for q in g), _local(comp, v)) for g, v in gg]
    groups = [g for g, _ in gg]
    by_group, psi_of = {}, {}
    for (g_loc, v_loc, psi_k), (g, _) in zip(prefix_states(comp, loc, prob), gg):
        by_group[g] = dict(qubits=list(g_loc), expvals=rdm_words(rdm(psi_k, len(comp), list(g_loc)), len(g_loc), tol))
        psi_of[g] = psi_k
    # one-qubit targets (the root of each chain) have never completed live with an X word: merge a root singleton into the next step,
    # whose prefix state already contains it, so the pair target builds both qubits from |0>
    if not keep_singletons:
        groups = [g for g in groups if len(g) > 1 or not any(len(h) > 1 and set(g) <= set(h) for h in groups)]
    layers = pack_layers(groups)
    edges = sorted({tuple(sorted(p)) for p in pairs if p[0] in comp and p[1] in comp} | set(fill))
    return StatePrepPlan(list(comp), [[by_group[g] for g in layer] for layer in layers], [psi_of[layer[-1]] for layer in layers],
                         [[_local(comp, a), _local(comp, b)] for a, b in edges])


def state_prep_params(plan, lo=0, hi=None, rounds=1, seed=7, update_method="spectral", shots=1024, chained=False, coupling_map=True):
    """QDrive params for layers lo..hi-1 of a plan. chained=True is a job that continues a returned circuit (`initial_circuit` is passed
    as an input file): n_qubits is then left out because the circuit carries its width (the engine raises n_qubits_mismatch otherwise)."""
    hi = len(plan.layers) if hi is None else hi
    targets = []
    for _ in range(rounds):
        for layer in plan.layers[lo:hi]:
            targets += [dict(t) for t in layer] + [None]
    params = dict(machine="aer", targets=targets, update_method=update_method, shots=shots, tomography=0, sample=False, seed=seed)
    if not chained:
        params["n_qubits"] = plan.n
    if coupling_map:
        params["coupling_map"] = [list(e) for e in plan.edges]
    return params


def build_state_prep_job(comp, pairs, prob, rounds=1, seed=7, update_method="spectral", shots=1024, tol=1e-3, keep_singletons=False):
    """Sequential state preparation: one target per growth step whose words are the FULL reduced density matrix (X, Y and Z words)
    of the partially built state on the step's group, which a gate on that group (plus the fresh qubit) can reach exactly.
    `prob` is the component's probability vector (local qubits). The pure state chosen among all with those Z statistics is the
    positive-amplitude one (the project's oracle state). One job holding every layer; qdrive_rounds cuts the same plan into several."""
    plan = plan_state_prep(comp, pairs, prob, tol, keep_singletons)
    params = state_prep_params(plan, rounds=rounds, seed=seed, update_method=update_method, shots=shots)
    groups = [[tuple(plan.comp[i] for i in t["qubits"]) for t in layer] for layer in plan.layers]
    return Job(qubits=list(comp), layers=groups, params=params, rounds=rounds)


# ---------------------------------------------------------------------------------------------------------- jobs
@dataclass
class Job:
    qubits: list                       # global qubit ids of this component, ascending; local index = position
    layers: list                       # [[group, ...], ...] in GLOBAL ids
    params: dict = field(default_factory=dict)
    rounds: int = 1

    @property
    def n_targets(self):
        return sum(1 for t in self.params["targets"] if t is not None)


def _local(job_qubits, q):
    return job_qubits.index(q)


def build_job(comp, pairs, moment, max_k=MAX_GROUP, rounds=1, seed=7, update_method="spectral", shots=1024, only=None, values=None):
    """One QDrive job for one component. `only` restricts to a set of groups (the closed loop's stubborn ones); `values` maps
    (group, word) -> corrected value overriding the moment provider."""
    groups = order_groups(cover_groups(comp, pairs, max_k))
    if only is not None:
        groups = [g for g in groups if g in only]
    layers = pack_layers(groups)
    cmap = [[_local(comp, a), _local(comp, b)] for a, b in sorted(map(tuple, map(sorted, pairs))) if a in comp and b in comp]
    targets = []
    for _ in range(rounds):
        for layer in layers:
            for g in layer:
                t = group_target(g, moment)
                if values:
                    t["expvals"] = {w: values.get((g, w), v) for w, v in t["expvals"].items()}
                targets.append(dict(qubits=[_local(comp, q) for q in g], expvals=t["expvals"]))
            targets.append(None)
    params = dict(machine="aer", n_qubits=len(comp), coupling_map=cmap, targets=targets, update_method=update_method,
                  shots=shots, tomography=0, sample=False, seed=seed)
    return Job(qubits=list(comp), layers=layers, params=params, rounds=rounds)


def plan_scene(moment, max_k=MAX_GROUP, rounds=1, seed=7, update_method="spectral"):
    """One job per independent component. Returns a list of Job."""
    pairs = moment.pairs()
    return [build_job(c, pairs, moment, max_k, rounds, seed, update_method) for c in components(moment.qubits(), pairs)]


def describe(jobs):
    rows = []
    for j in jobs:
        sizes = sorted({len(g) for layer in j.layers for g in layer})
        rows.append(f"qubits {j.qubits[0]}-{j.qubits[-1]} ({len(j.qubits)}q): {sum(len(l) for l in j.layers)} groups (sizes {sizes}) in "
                    f"{len(j.layers)} layers x {j.rounds} rounds = {j.n_targets} targets + {sum(1 for t in j.params['targets'] if t is None)} updates")
    return rows


def check(job):
    """Structural rules the lab found (returns a list of problems; empty = ok)."""
    out = []
    t = job.params["targets"]
    layer = []
    for e in t + [None]:
        if e is None:
            qs = [q for x in layer for q in x["qubits"]]
            if len(qs) != len(set(qs)):
                out.append("two overlapping targets share a layer")
            layer = []
        else:
            layer.append(e)
            if len(e["qubits"]) > MAX_GROUP:
                out.append(f"group of {len(e['qubits'])} qubits (> {MAX_GROUP}): the lab timed out at 8")
            if any(len(w) != len(e["qubits"]) for w in e["expvals"]):
                out.append("word length differs from the group size")
    if job.params["n_qubits"] > 12:
        out.append(f"{job.params['n_qubits']}-qubit register: every 19-qubit job failed; the width limit is not located")
    return out


# ---------------------------------------------------------------------------------------------------------- merge
def merge_circuits(parts, n_total):
    """parts: [(qasm3_text, global_qubit_list)] of independent components -> one OpenQASM 3 circuit on n_total qubits (local)."""
    from qiskit import QuantumCircuit, qasm3
    big = QuantumCircuit(n_total)
    for text, qubits in parts:
        qc = qasm3.loads(text).remove_final_measurements(inplace=False)
        big.compose(qc, qubits=list(qubits), inplace=True)
    return qasm3.dumps(big)


# ---------------------------------------------------------------------------------------------------------- closed loop
def residuals(job, read, moment):
    """{(group, word): (target, achieved)} for every word the job targets. `read(word, global_qubits)` reads the current state."""
    out = {}
    for layer in job.layers:
        for g in layer:
            for w, v in group_target(g, moment)["expvals"].items():
                qs = [g[i] for i, ch in enumerate(w) if ch == "Z"]
                out[(g, w)] = (v, read("Z" * len(qs), qs))
    return out


def stubborn(res, tol):
    """Groups whose worst word is still more than tol from its target."""
    worst = {}
    for (g, w), (t, a) in res.items():
        worst[g] = max(worst.get(g, 0.0), abs(t - a))
    return {g for g, e in worst.items() if e > tol}, worst


def corrected_values(res, gain):
    """Push past the target by `gain` of the remaining error: t' = clip(t + gain (t - a)). gain 0 repeats the target unchanged."""
    return {k: round(float(np.clip(t + gain * (t - a), -CLIP, CLIP)), 4) for k, (t, a) in res.items()}


def closed_loop(comp, pairs, moment, engine, tol=0.03, gain=0.5, max_jobs=6, rounds_first=1, rounds_next=1, max_k=MAX_GROUP, log=print):
    """Gradual push with feedback. Job 1 runs the full sweep; afterwards each job repeats only the groups still more than `tol`
    off, aimed past the target by `gain`. engine(params, prev) -> (state, read) where read(word, qubits) reads that state;
    prev is the previous state (chained circuit) or None. Returns (state, history)."""
    full = build_job(comp, pairs, moment, max_k, 1)               # defines what "done" means: ALL groups, every iteration
    job = build_job(comp, pairs, moment, max_k, rounds_first)
    state, read = engine(job.params, None)
    hist = []
    for i in range(1, max_jobs + 1):
        res = residuals(full, read, moment)
        bad, worst = stubborn(res, tol)
        rms = float(np.sqrt(np.mean([(t - a) ** 2 for t, a in res.values()])))
        hist.append(dict(job=i, rms=rms, max=max(worst.values()), stubborn=len(bad), groups=len(worst)))
        log(f"  job {i}: rms {rms:.4f}  worst group {max(worst.values()):.3f}  stubborn {len(bad)}/{len(worst)}")
        if not bad or i == max_jobs:
            break
        job = build_job(comp, pairs, moment, max_k, rounds_next, only=bad, values=corrected_values(res, gain))
        state, read = engine(job.params, state)
    return state, hist
