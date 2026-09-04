# CNRL gate analysis

Rows loaded: **36** from **1** file(s).

## Structural audit

**PASS:** no invalid rows, affinity failures, Bclone invariant failures, or shared/Bclone checksum divergences.

## T0-R: residencia por profundidad

| D | S | R | filas | kernel | A/B mediana | min(A)>max(B) | B one-pass GB/s | lectura |
|---:|---:|---:|---:|---|---:|---|---:|---|

## T0-M: matrixización por slots

| D | filas | R | variante | S | G(S)=fused/fused(S=1) | F(S)=fused/repeat | lectura |
|---:|---:|---:|---|---:|---:|---:|---|

## T0-M: intercambio profundidad/slots a R×S constante

| D | filas | variante | R×S | pares medidos (R,S: mediana ms) | más rápido |
|---:|---:|---|---:|---|---|

## T0-RM: recurrencia real

| D | S | R | transición | shift | A/B mediana | clipping mediano | validez numérica | lectura física |
|---:|---:|---:|---|---:|---:|---:|---|---|
| 1472 | 1 | 8 | fixed-point | 12 | 1.083× | 25.433% | NO VÁLIDO NUMÉRICAMENTE | sin separación material |
| 1472 | 1 | 8 | fixed-point | 13 | 0.895× | 3.252% | ADVERTENCIA | sin separación material |
| 1472 | 1 | 8 | fixed-point | 14 | 1.451× | 0.000% | OK | separación |
| 1472 | 1 | 8 | fixed-point | 15 | 0.931× | 0.000% | OK | sin separación material |
| 1472 | 1 | 8 | global-rms | 12 | 0.854× | 0.000% | OK | sin separación material |
| 1472 | 1 | 8 | group-rms | 12 | 1.031× | 0.000% | OK | sin separación material |
| 1472 | 16 | 8 | fixed-point | 12 | 1.027× | 25.411% | NO VÁLIDO NUMÉRICAMENTE | paridad compatible con alta reutilización por slots |
| 1472 | 16 | 8 | fixed-point | 13 | 0.803× | 3.083% | ADVERTENCIA | paridad compatible con alta reutilización por slots |
| 1472 | 16 | 8 | fixed-point | 14 | 1.095× | 0.002% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 16 | 8 | fixed-point | 15 | 1.319× | 0.000% | OK | separación |
| 1472 | 16 | 8 | global-rms | 12 | 0.878× | 0.004% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 16 | 8 | group-rms | 12 | 0.903× | 0.005% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 8 | 8 | fixed-point | 12 | 1.037× | 25.053% | NO VÁLIDO NUMÉRICAMENTE | paridad compatible con alta reutilización por slots |
| 1472 | 8 | 8 | fixed-point | 13 | 1.005× | 2.900% | ADVERTENCIA | paridad compatible con alta reutilización por slots |
| 1472 | 8 | 8 | fixed-point | 14 | 0.621× | 0.000% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 8 | 8 | fixed-point | 15 | 0.649× | 0.000% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 8 | 8 | global-rms | 12 | 1.028× | 0.003% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 8 | 8 | group-rms | 12 | 0.909× | 0.005% | OK | paridad compatible con alta reutilización por slots |

El clipping del microbenchmark de transición no sustituye esta columna: la validez del gate recurrente se juzga sobre la trayectoria T0-RM de R rondas.

## T0-RM: throughput retenido frente al puente frozen

| D | S | R | variante | transición | recurrente/estático |
|---:|---:|---:|---|---|---:|

## Variabilidad externa

No se detectaron condiciones con al menos tres muestras y CV superior a 10%.

## Metric warning

`mac_per_second / S` equals the one-pass int8 weight-stream rate only for a square/rectangular dot-product that loads each weight once per slot group. The authoritative CSV field is `one_pass_weight_gb_per_second`; do not equate raw GMAC/s with DRAM GB/s when S>1.
