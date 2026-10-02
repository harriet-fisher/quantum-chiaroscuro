"""Lamp hub: two lamp qubits connected to every patch (handoff §2.5, §9.1)."""
import numpy as np


def hub_fields(cells, panels, kh=1.6, kf=1.3):
    """Coupling of each patch to lamp L1 (left/right light, ~ normal x) and lamp L2 (frontal vs grazing,
    ~ normal z minus its mean over patches)."""
    nx = np.array([panels[c["k"]].normal[0] for c in cells])
    nz = np.array([panels[c["k"]].normal[2] for c in cells])
    return kh * nx, kf * (nz - nz.mean())
