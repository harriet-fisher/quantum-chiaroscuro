"""The depth-consistency parity game: every frame is one round of a Mermin (GHZ-style) game between the lamp and the depth qubits of three panels.

Parties: the lamp (L1) and three leaf domains' polarity qubits (one per panel, the ones the lamp is locked to). A round draws an input S, a uniformly random
EVEN-size subset of the four parties. A party in S reads its qubit along its setting A', a party outside S along A (the frames A, A' come from the state:
numerically optimised Mermin axes, see `game_frames`). Everyone gets a +1/-1 outcome (bit 0 is +1). The parties win the round when

        (product of the four outcomes) * (-1)^(|S|/2) = +1.

Why this is a test of quantumness. The eight inputs are the eight terms of the four-party Mermin operator, so a round's win probability is
(1 + M / 8) / 2 for a state whose Mermin value is M. Any classical strategy (outcomes fixed in advance by the setting, shared randomness allowed) has
M <= 4, so it wins at most 75% of the rounds; the GHZ state wins all of them. `classical_bound` proves the 75% by brute force over all strategies.

How a round becomes a frame. The lamp is read along A or A' (a rotated axis: this is the lamp read in X, no longer a coin flip between two lights) and its
OUTCOME then chooses the light side (feed-forward, so the facets keep no record of the lamp and its coherence with the depth qubits survives); the three
leaves are observed along their own settings (the depth-sphere dot of a leaf sits on its axis); every other domain keeps the frame's ordinary observation. The
frame shows the picture those outcomes produce, and the round is scored on the outcomes. The picture alone cannot certify the game (a single round is one bit);
the win RATE over many rounds can, and `GameStats` keeps it with its distance from the classical bound.

The honest limit: the win rate is (1 + M/8)/2 with M the Mermin value of THIS state, reduced by every dephasing the facets and the neighbouring domains cause.
`predicted` says what it should be; the measured rate must agree with it, and it exceeds 0.75 only if M > 4.
"""
import itertools
import math

import numpy as np

from src.quantum import domain_witness as dw


def bloch_angles(v):
    """(gamma, chi) of a unit Bloch vector: polar angle from +z, azimuth."""
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    return float(np.arccos(np.clip(v[2], -1, 1))), float(np.arctan2(v[1], v[0]))


def even_inputs(m):
    """All 2^(m-1) input vectors of m parties with an even number of ones."""
    return [x for x in itertools.product((0, 1), repeat=m) if sum(x) % 2 == 0]


def term_sign(x):
    return (-1) ** (sum(x) // 2)


def classical_bound(m=4):
    """Best win probability of any deterministic classical strategy (party i answers f_i(setting_i) in {+1, -1}) over the uniform even inputs. Randomised
    strategies are mixtures of deterministic ones, so this is the classical maximum. Brute force over 4^m strategies."""
    xs = even_inputs(m)
    best = 0.0
    for f in itertools.product((1, -1), repeat=2 * m):                       # f[2i + s] = answer of party i to setting s
        wins = sum(1 for x in xs if np.prod([f[2 * i + x[i]] for i in range(m)]) * term_sign(x) == 1)
        best = max(best, wins / len(xs))
    return best


def game_frames(state, leaves=None, restarts=24, seed=0):
    """(M, axes): the largest Mermin value of the lamp plus the leaf domains, and the (A, A') Bloch-vector pair of each party, the lamp first. `leaves`
    defaults to the domains the lamp is locked to (one per panel)."""
    sp = state.spec
    lock = sp.lock_array()
    if lock is None:
        raise ValueError("the parity game needs the lamp locked to the depth qubits (lock > 0)")
    leaves = leaves if leaves is not None else _default_leaves(sp)
    ls = dw.lamp_state(state)
    n = sp.n_sim
    rho = ls.rdm([n] + [sp.F + d for d in leaves])
    val, axes = dw.max_mermin(rho, len(leaves) + 1, restarts=restarts, seed=seed)
    return val, axes, leaves


def _default_leaves(sp):
    """One locked domain per panel (the most visible), at most three, as the witness picks them."""
    lock = sp.lock_array()
    dom_panel = {int(sp.panel_of[i]): int(sp.panel_of[i] if sp.facet_panel is None else sp.facet_panel[i]) for i in range(sp.F)}
    best = {}
    for d in range(sp.P):
        if lock[d]:
            p = dom_panel.get(d, 0)
            v = sp.visibility(d)
            if p not in best or v > best[p][1]:
                best[p] = (d, v)
    return [d for d, _ in best.values()][:3]


class GameStats:
    CLASSICAL = 0.75

    def __init__(self, predicted=None):
        self.rounds = self.wins = 0
        self.by_input = {}
        self.predicted = predicted

    def add(self, x, win):
        self.rounds += 1
        self.wins += int(win)
        r = self.by_input.setdefault(tuple(x), [0, 0])
        r[0] += 1
        r[1] += int(win)

    def rate(self):
        return self.wins / self.rounds if self.rounds else None

    def sigma_above_classical(self):
        if not self.rounds:
            return None
        return (self.wins / self.rounds - self.CLASSICAL) / math.sqrt(self.CLASSICAL * (1 - self.CLASSICAL) / self.rounds)

    def wilson(self, z=1.96):
        n = self.rounds
        if not n:
            return None
        p = self.wins / n
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return (c - h, c + h)

    def summary(self):
        lo_hi = self.wilson()
        return dict(rounds=self.rounds, wins=self.wins, rate=self.rate(), classical_bound=self.CLASSICAL, predicted=self.predicted,
                    sigma_above_classical=self.sigma_above_classical(), interval95=lo_hi,
                    by_input={"".join(map(str, k)): dict(rounds=v[0], wins=v[1]) for k, v in sorted(self.by_input.items())})


class ParityGame:
    """The game on an MPS relief state (the lamp read along a rotated axis needs it; a small spec can use MPSReliefState too)."""

    def __init__(self, state, leaves=None, restarts=24, seed=0):
        if getattr(state, "backend", "exact") != "mps":
            raise ValueError("the parity game runs on the matrix-product backend (MPSReliefState)")
        self.state = state
        self.M, self.axes, self.leaves = game_frames(state, leaves, restarts, seed)
        m = len(self.axes)
        self.parties = ["lamp"] + [f"domain {d}" for d in self.leaves]
        self.predicted = 0.5 * (1 + self.M / 2 ** (m - 1))
        self.inputs = even_inputs(m)
        self.stats = GameStats(self.predicted)

    def settings(self, x):
        """The measurement of every party for input x: the lamp's Bloch axis and {domain: (gamma, chi)} of the leaves."""
        lamp = tuple(self.axes[0][x[0]])
        obs = {d: bloch_angles(self.axes[j + 1][x[j + 1]]) for j, d in enumerate(self.leaves)}
        return lamp, obs

    def play(self, rng, K=48, x=None, g=None):
        """One round and the frame it makes. Returns (draw, round) with round = dict(inputs, outcomes, win, ...); the draw also carries it as `.game`."""
        if x is None:
            x = self.inputs[int(rng.integers(len(self.inputs)))]
        lamp_axis, obs = self.settings(x)
        d = self.state.draw(K, rng, g=g, game_axis=lamp_axis, obs_axes=obs)
        a = [1 - 2 * d.lamp_outcome[0]] + [d.pol[dom] for dom in self.leaves]
        win = bool(np.prod(a) * term_sign(x) == 1)
        self.stats.add(x, win)
        rnd = dict(inputs=tuple(x), outcomes=tuple(int(v) for v in a), win=win, parties=list(self.parties), sign=term_sign(x))
        d.game = rnd
        return d, rnd

    def describe(self):
        s = self.stats.summary()
        return (f"parity game: {s['wins']} of {s['rounds']} rounds won ({100 * (s['rate'] or 0):.1f}%), classical maximum 75%, this state's prediction "
                f"{100 * self.predicted:.1f}% (Mermin value {self.M:.2f}); " + (f"{s['sigma_above_classical']:+.1f} sigma from the classical bound" if s["rounds"] else "no rounds yet"))
