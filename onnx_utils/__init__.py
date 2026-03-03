"""
ONNX model processing utilities.

This module provides tools for:
- ONNX model health checking
- FP16 conversion (onnx2strongly)
- Topological sorting
- Graph surgeon operations
"""

from .health_check import HealthChecker, SeverityLevel

__all__ = ["HealthChecker", "SeverityLevel"]
