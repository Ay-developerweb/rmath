"""
rmath.nn — Neural Network building blocks and activation functions.
"""

from typing import List, Union, Tuple, Sequence
from rmath.array import Array
from rmath.vector import Vector

def relu(a: Array) -> Array:
    """Rectified Linear Unit: max(0, x)."""
    ...

def sigmoid(a: Array) -> Array:
    """Sigmoid activation: 1 / (1 + e^-x)."""
    ...

def leaky_relu(a: Array, alpha: float) -> Array:
    """Leaky ReLU: x if x >= 0 else alpha * x."""
    ...

def gelu(a: Array) -> Array:
    """Gaussian Error Linear Unit (GELU)."""
    ...

def softmax(a: Array) -> Array:
    """Softmax activation along the last dimension."""
    ...

def mse_loss(a: Array, target: Array) -> float:
    """Mean Squared Error loss."""
    ...

def cross_entropy_loss(a: Array, labels: Sequence[int]) -> float:
    """Cross-Entropy loss (expects log-probabilities)."""
    ...

def batch_norm(a: Array, mu: Vector, sigma: Vector, gamma: Vector, beta: Vector) -> Array:
    """Batch normalization (inference mode)."""
    ...

def layer_norm(a: Array, eps: float = 1e-5) -> Array:
    """Layer normalization."""
    ...

def dropout(a: Array, p: float) -> Array:
    """Apply dropout during training."""
    ...

# --- Initializers ---
class initializers:
    """Weight initialization kernels."""
    @staticmethod
    def glorot_uniform(*shape: int) -> Array: ...
    @staticmethod
    def glorot_normal(*shape: int) -> Array: ...
    @staticmethod
    def he_uniform(*shape: int) -> Array: ...
    @staticmethod
    def he_normal(*shape: int) -> Array: ...
