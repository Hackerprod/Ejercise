from pathlib import Path
import sys

UNIT_ROOT = Path(__file__).resolve().parents[1]
if str(UNIT_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIT_ROOT))

from omega_v2.core import configure_reference_execution  # noqa: E402

configure_reference_execution()
