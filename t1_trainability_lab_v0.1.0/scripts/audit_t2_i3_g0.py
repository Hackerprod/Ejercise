"""G0 oracle-isolation guard for THINK-0."""
from __future__ import annotations
import inspect
from t2_i3_think0 import Think0

FORBIDDEN = ("constraints", "lower", "forbidden", "floor", "avoid", "tokens", "clauses", "segments", "split")

def main() -> None:
    parameters = inspect.signature(Think0.forward).parameters
    if tuple(parameters) != ("self", "workspace"): raise RuntimeError(f"THINK oracle isolation failed: {tuple(parameters)}")
    if any(name in parameters for name in FORBIDDEN): raise RuntimeError("THINK received forbidden oracle input")
    print('{"status":"passed","task":"T2-I3","gate":"G0","training":false}')

if __name__ == "__main__": main()
