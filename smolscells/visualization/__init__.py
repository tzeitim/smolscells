"""Visualization tools for SmolScells"""

try:
    from .plots import TreeVisualizer
    __all__ = ["TreeVisualizer"]
except ImportError:
    __all__ = []