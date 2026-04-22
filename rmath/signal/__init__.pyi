"""
rmath.signal — High-performance Signal Processing (FFT and Convolution).
"""

from typing import Tuple, Union, Literal, Sequence
from rmath.vector import Vector, ComplexVector

def fft(signal: Union[Vector, Sequence[float]]) -> ComplexVector:
    """Compute the one-dimensional Fast Fourier Transform (FFT).
    
    Examples:
        >>> from rmath.signal import fft
        >>> res = fft([1, 0, 1, 0])
        >>> res.to_mags()
        Vector([2.0, 0.0, 2.0, 0.0])
    """
    ...

def ifft(cv: ComplexVector) -> Vector:
    """Compute the inverse Fast Fourier Transform (IFFT)."""
    ...

def fft_styled(signal: Union[Vector, Sequence[float]]) -> Tuple[Vector, Vector]:
    """Compute the FFT and return both (Magnitudes, Phases)."""
    ...

def rfft(signal: Union[Vector, Sequence[float]]) -> Vector:
    """Real-input FFT – returns ONLY magnitudes for positive frequencies."""
    ...

def convolve(
    signal: Vector,
    kernel: Vector,
    mode: Literal["full", "same", "valid"] = "full"
) -> Vector:
    """Perform 1D convolution using the FFT method (O(N log N)).
    
    Examples:
        >>> from rmath.signal import convolve
        >>> convolve([1, 2, 3], [0, 1, 0], mode='same')
        Vector([1.0, 2.0, 3.0])
    """
    ...
