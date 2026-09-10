# CNRL transition benchmark analysis

Rows loaded: **18**.

## Structural audit

**PASS:** chain accounting, reset semantics, affinity, sharding and clipping rates are consistent.

## Results

| D | S | transition | chain | shift | state/output/final | target RMS | median ns/cell | clipping | numeric reading |
|---:|---:|---|---:|---:|---|---:|---:|---:|---|
| 1472 | 1 | fixed-point | 1 | 14 | 1/1/0 | 32 | 0.650 | 0.000% | OK |
| 1472 | 1 | fixed-point | 8 | 14 | 1/1/0 | 32 | 0.333 | 0.000% | OK |
| 1472 | 1 | global-rms | 1 | 12 | 1/1/0 | 32 | 1.053 | 0.000% | OK |
| 1472 | 1 | global-rms | 8 | 12 | 1/1/0 | 32 | 0.883 | 0.000% | OK |
| 1472 | 1 | group-rms | 1 | 12 | 1/1/0 | 32 | 0.783 | 0.000% | OK |
| 1472 | 1 | group-rms | 8 | 12 | 1/1/0 | 32 | 0.539 | 0.000% | OK |
| 1472 | 16 | fixed-point | 1 | 14 | 1/1/0 | 32 | 0.135 | 0.000% | OK |
| 1472 | 16 | fixed-point | 8 | 14 | 1/1/0 | 32 | 0.256 | 0.000% | OK |
| 1472 | 16 | global-rms | 1 | 12 | 1/1/0 | 32 | 0.281 | 0.000% | OK |
| 1472 | 16 | global-rms | 8 | 12 | 1/1/0 | 32 | 0.270 | 0.000% | OK |
| 1472 | 16 | group-rms | 1 | 12 | 1/1/0 | 32 | 0.312 | 0.000% | OK |
| 1472 | 16 | group-rms | 8 | 12 | 1/1/0 | 32 | 0.269 | 0.000% | OK |
| 1472 | 8 | fixed-point | 1 | 14 | 1/1/0 | 32 | 0.249 | 0.000% | OK |
| 1472 | 8 | fixed-point | 8 | 14 | 1/1/0 | 32 | 0.177 | 0.000% | OK |
| 1472 | 8 | global-rms | 1 | 12 | 1/1/0 | 32 | 0.333 | 0.000% | OK |
| 1472 | 8 | global-rms | 8 | 12 | 1/1/0 | 32 | 0.775 | 0.000% | OK |
| 1472 | 8 | group-rms | 1 | 12 | 1/1/0 | 32 | 0.296 | 0.000% | OK |
| 1472 | 8 | group-rms | 8 | 12 | 1/1/0 | 32 | 0.325 | 0.000% | OK |

`chain_length=1` mide coste aislado; una cadena R solo caracteriza deriva bajo el output sintético declarado. La validez autoritativa de una transición sigue siendo su clipping dentro de T0-RM real.
