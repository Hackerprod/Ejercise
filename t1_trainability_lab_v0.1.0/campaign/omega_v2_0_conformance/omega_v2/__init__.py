"""OMEGA-V2-0 isolated architectural-conformance reference."""

from .core import ContractualCoreBlock, SharedRecurrentCore, configure_reference_execution
from .variants import R4Shared, U4Untied

__all__ = [
    "ContractualCoreBlock",
    "SharedRecurrentCore",
    "R4Shared",
    "U4Untied",
    "configure_reference_execution",
]
