"""
rmath.signal — Signal Processing and Fourier Transforms.

Highly optimized routines for:
    - FFT and Wavelet transforms.
    - Filtering (Butterworth, Chebyshev).
    - Spectral analysis and convolution.
"""
from .._rmath import signal as _signal

for name in dir(_signal):
    if not name.startswith('_'):
        globals()[name] = getattr(_signal, name)
