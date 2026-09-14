"""Image-scoped package surface for the design-audit dependency chain.

The full repository package exports unrelated training modules through its
normal ``__init__``. The derived image intentionally carries only the model
primitives imported by the frozen design audit, without copying the repository
or installing a same-named package from PyPI.
"""

from .model import CoreMLP, RMSNorm, SlotMix

__all__ = ["CoreMLP", "RMSNorm", "SlotMix"]
