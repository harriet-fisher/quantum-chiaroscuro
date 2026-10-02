"""Payload builders: orderings, pruning, and the QDrive mapping form. Nothing is sent.

    python -m unittest tests.test_payloads -v
"""
import unittest

from src.quantum.moth_client import MothClient
from src.targets.payloads import graph_v1_payload, qdrive_payload

BLOCH = [dict(qubit=q, Z=z) for q, z in enumerate([0.0, 0.3, 0.0, 0.001])]
REL = [dict(qubits=[0, 1], ZZ=0.9), dict(qubits=[1, 2], ZZ=-0.02), dict(qubits=[2, 3], ZZ=0.5), dict(qubits=[0, 3], ZZ=-0.7)]


class GraphV1(unittest.TestCase):
    def rel_order(self, **kw):
        p = graph_v1_payload(BLOCH, REL, 4, **kw)["params"]
        return [tuple(o["qubits"]) for o in p["operations"] if o["type"] == "relationship"], p

    def test_default_order_is_what_the_first_run_sent(self):
        order, p = self.rel_order()
        self.assertEqual(order, [(0, 1), (0, 3), (2, 3), (1, 2)])                    # strongest |ZZ| first
        self.assertEqual([o["type"] for o in p["operations"][:4]], ["bloch"] * 4)    # all <Z> first

    def test_weakest_first_puts_the_strongest_last(self):
        order, _ = self.rel_order(order="weakest_first")
        self.assertEqual(order, [(1, 2), (2, 3), (0, 3), (0, 1)])

    def test_drop_below_removes_small_targets_and_prunes_edges(self):
        order, p = self.rel_order(order="weakest_first", drop_below=0.05)
        self.assertNotIn((1, 2), order)
        self.assertNotIn([1, 2], p["coupling_map"])
        self.assertEqual([o["qubit"] for o in p["operations"] if o["type"] == "bloch"], [1])      # only |Z| >= 0.05 survives
        self.assertTrue(all(list(o["qubits"]) in p["coupling_map"] for o in p["operations"] if o["type"] == "relationship"))

    def test_no_negative_zero_and_valid_for_the_engine(self):
        p = graph_v1_payload([dict(qubit=0, Z=-1e-9)], [dict(qubits=[0, 1], ZZ=0.5)], 2)["params"]
        self.assertEqual(str(p["operations"][0]["paulis"]["Z"]), "0.0")
        self.assertEqual(MothClient().prepare("graph-v1", p).problems(), [])


class QDrive(unittest.TestCase):
    def test_expvals_are_mappings_not_numbers(self):
        """The real engine rejected bare numbers: invalid_target ... 'expvals' must be a mapping."""
        p = qdrive_payload(BLOCH, REL, 4, [[0, 1], [1, 2]], rounds=2)["params"]
        t = p["targets"]
        self.assertEqual(t[0], {"qubits": [0], "expvals": {"Z": 0.0}})
        self.assertTrue(all(isinstance(e["expvals"], dict) for e in t if e is not None))
        self.assertEqual([e is None for e in t].count(True), 2)                       # one update entry per round
        self.assertIsNone(t[len(t) // 2 - 1])
        self.assertEqual(MothClient().prepare("qdrive-api-v1", p).problems(), [])


if __name__ == "__main__":
    unittest.main()
