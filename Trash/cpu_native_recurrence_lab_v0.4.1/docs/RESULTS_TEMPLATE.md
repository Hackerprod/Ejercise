# Plantilla de reporte

## Hardware

```text
CPU:
physical cores / SMT:
L2/core:
L3:
RAM channels / speed:
Windows build:
power mode:
compiler / generator / toolset:
commit or ZIP SHA-256:
```

## Evidencia estructural

```text
self-tests:
source audit:
assembly audit:
topology JSON:
calibration rates:
read-only bandwidth:
```

## T0-R

```text
D/S/R:
rows per worker:
A median/min/max:
Bclone median/min/max:
A/B:
one-pass B GB/s:
checksum equality:
verdict:
```

## T0-M

```text
G4/G8/G16:
Fused/Repeat by S:
slot tile:
constant R×S curve:
verdict:
```

## T0-RM

```text
transition:
D/S/R:
static bridge throughput:
recurrent throughput:
transition profile (separate run):
A/Bclone:
projection_shift:
clipping by transition:
numeric validity:
variance:
checksum equality:
verdict:
```

## Claims explicitly not made

- calidad lingüística;
- entrenamiento estable;
- memoria externa útil;
- equivalencia de parámetros con Transformer;
- escalado a CPUs mayores.

## Transición aislada

```text
D/S:
transition:
rows/core:
cell updates/s:
ns/cell:
clipping:
physical cores únicos:
```

## Auditoría contable

```text
header/row field count:
mac_total recomputed:
one_pass bytes recomputed:
logical loads recomputed:
allocated bytes recomputed:
strict analyzer verdict:
```


## Fixed-point shift sweep

```text
D/S/R:
shifts tested:
clipping median by shift:
A/Bclone by shift:
lowest accepted shift:
verdict:
```
