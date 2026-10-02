"""Mask-aware (normalised) Gaussian blur: black never bleeds into panels and light never bleeds into glass."""
import numpy as np
from scipy import ndimage as ndi


def gauss(x, s):
    return ndi.gaussian_filter(x, s, mode="nearest")


def masked_blur(val, frame, sigma):
    fr = frame.astype(float)
    return gauss(val * fr, sigma) / np.maximum(gauss(fr, sigma), 1e-6)
