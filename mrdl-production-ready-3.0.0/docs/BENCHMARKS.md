# Benchmarks

## Objetivos

`mrdl_bench` mide piezas de arquitectura, no calidad lingüística:

1. costo FULL-only;
2. costo CLEAN-only;
3. FULL+CLEAN aislado;
4. FULL+CLEAN paralelo con reutilización pura exacta;
5. ratio de operaciones;
6. estado activo y almacenamiento de replay;
7. salud de CLEAN según densidad M1;
8. throughput de aplicación/composición monomial.

## Suites

```bash
mrdl_bench --suite dual
mrdl_bench --suite monomial
mrdl_bench --suite all --json
```

`--quick` reduce iteraciones para smoke tests. `--iterations N` controla repeticiones de la suite dual.

## Matriz dual

La suite ejecuta las configuraciones acordadas:

- `k=2, beam=4, depth=16`;
- `k=8, beam=32, depth=16`;
- M1 solicitado: 0 %, 1 %, 10 % y 50 %.

Los grafos son sintéticos y deterministas. La fracción real se reporta porque el muestreo discreto puede variar ligeramente.

## Interpretación

- `R_runtime_isolated > 1`: impuesto dual sin paralelismo.
- `R_runtime_parallel_reuse`: costo observado con dos carriles concurrentes y cache pura.
- `R_ops`: `(ops_FULL + ops_CLEAN) / ops_FULL`.
- `active_full`, `active_clean`: pico de ramas activas; deben respetar el beam.
- `replay_entries`: cantidad reconstruible, no nodos de un árbol simbólico.
- `clean_health`: mínimo relativo de actividad/operaciones útiles.
- `clean_degenerate`: `clean_health < 0.5`.
- `clean_empty`: no sobrevivió candidato CLEAN.

Las filas degeneradas o vacías se excluyen de promedios de rendimiento. El runtime barato porque CLEAN colapsó es un resultado negativo.

## Reproducibilidad

Conserve:

- commit o hash del paquete;
- compilador y flags;
- CPU (`lscpu`);
- governor y frecuencia;
- número de iteraciones;
- carga paralela del VPS;
- salida JSONL completa.

Ejemplo:

```bash
{
  date -u
  uname -a
  lscpu
  mrdl version
  mrdl_bench --suite all --json
} > benchmark-$(date -u +%Y%m%dT%H%M%SZ).log
```
