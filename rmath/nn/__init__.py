"""
rmath.nn — Neural Network Layers and Optimizers.

Includes parallelized implementations of:
    - SGD and Adam Optimizers.
    - Linear, Conv2D, and Pooling layers.
    - Loss functions (MSE, CrossEntropy).
"""
from .._rmath import nn as _nn

for name in dir(_nn):
    if not name.startswith('_'):
        globals()[name] = getattr(_nn, name)
