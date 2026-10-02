"""Qubit allocation and budget (handoff §7.3 step 7, §10.1).

Index layout, used by every payload: patches 0..n-1, then lamp L1, lamp L2, then one polarity qubit per panel.
Z convention: Z = +1 means lit (patch), light from the right (L1), frontal light (L2), raised relief (polarity).
"""
from src.graph.patches import build_patches

N_LAMPS = 2
CAP_WARN = 20   # Labyrinth's emulator cap; the graph-v1 / QDrive caps are [UNKNOWN]


def allocate(n_patches, n_panels):
    return dict(
        patches=list(range(n_patches)),
        lamps=[n_patches, n_patches + 1],
        polarity=list(range(n_patches + N_LAMPS, n_patches + N_LAMPS + n_panels)),
        n_qubits=n_patches + N_LAMPS + n_panels,
    )


def budget(n_patches, n_panels, cap=CAP_WARN):
    total = n_patches + N_LAMPS + n_panels
    return dict(patches=n_patches, lamps=N_LAMPS, panels=n_panels, total=total, cap=cap, over_cap=total > cap)


def format_budget(b):
    s = f"patches {b['patches']} + lamps {b['lamps']} + panels {b['panels']} (polarity) = {b['total']} qubits"
    if b["over_cap"]:
        s += f"\nWARNING: {b['total']} qubits exceeds {b['cap']}; the emulator cap is unknown, reduce the grid or merge panels"
    return s


def suggest_grids(scene, cap=CAP_WARN, min_cover=0.30, nx_range=range(3, 9), ny_range=range(2, 7), top=3):
    """Grids whose budget fits under the cap, preferring the finest (most patches)."""
    fits = []
    for nx in nx_range:
        for ny in ny_range:
            n = len(build_patches(scene, nx, ny, min_cover)[0])
            if n + N_LAMPS + scene.n_panels <= cap:
                fits.append((n, nx, ny))
    fits.sort(reverse=True)
    return [dict(nx=nx, ny=ny, patches=n, total=n + N_LAMPS + scene.n_panels) for n, nx, ny in fits[:top]]
