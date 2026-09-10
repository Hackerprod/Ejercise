# Primera validación bare-metal — Ryzen AI 5 330

## Estado de la evidencia

Este documento registra una validación ejecutada externamente sobre **CNRL v0.4.0** en Windows 11. Los números fueron reportados después de inspección de ensamblado y recomputación independiente de los CSV por el operador. Los CSV crudos no están incluidos en este paquete; por tanto, los resultados se consideran **evidencia externa reportada**, no una recomputación independiente de esta entrega.

## Hardware y toolchain reportados

- AMD Ryzen AI 5 330: 1 núcleo Zen 5 + 3 núcleos Zen 5c, 8 hilos SMT.
- Logical CPUs físicas seleccionadas: `0,2,4,6`, confirmadas por topología nativa.
- L2: aproximadamente 1 MiB por núcleo físico; L3: 8 MiB compartida.
- DDR5-5600 single-channel.
- Windows 11.
- Build Tools VS18/v180, CMake + Ninja + `vcvarsall.bat x64`.
- Flags verificados en `build.ninja`: `/O2 /arch:AVX2 /W4 /WX`.
- CTest: 3/3 PASS en v0.4.0.

## Mediciones reportadas

### Ancho de banda

- `cnrl_bandwidth --mode read`: **16.513 GB/s**.
- Probe histórico `memcpy`: 32.9295 GB/s contando lectura + escritura; normalizado a lectura: 16.465 GB/s.
- Diferencia entre ambos estimadores de lectura: aproximadamente **0.29%**.

### Calibración por núcleo

| Logical CPU | Clase | GMAC/s |
|---:|---|---:|
| 0 | Zen 5 | 42.9 |
| 2 | Zen 5c | 30.4 |
| 4 | Zen 5c | 33.7 |
| 6 | Zen 5c | 33.7 |

### T0-R aislado

PASS. En `R=16`, la mediana `shared/Bclone` fue reportada como:

| Peso promedio por núcleo | A/Bclone |
|---:|---:|
| 384 KiB | 1.95× |
| 512 KiB | 2.26× |
| 640 KiB | 2.80× |
| 768 KiB | 2.68× |

En varias celdas `R=8/R=16`, `min(shared) > max(Bclone)`.

### T0-M aislado

PASS_STRONG. `G(16)` para Bclone alcanzó **5.94×** en `R=16`, 512 KiB por núcleo. Esta métrica expresa incremento de throughput por MAC al matrixizar slots; no significa que dieciséis slots reduzcan la latencia total 5.94×.

### T0-RM recurrente

PASS reportado. Con `D=1472`, `S=1`, `R=8`, las tres transiciones —fixed-point, group-RMS y global-RMS— mantuvieron una separación `shared/Bclone` de aproximadamente **1.96–2.24×**. Para `S=8/16`, A/Bclone se acercó a 1, coherente con que Bclone aumenta su intensidad por reutilización intra-ronda entre slots y deja de estar dominado por DRAM.

El analizador estructural fue ejecutado tres veces sobre 1274 filas y reportó PASS; 186 condiciones superaron 10% de CV externo. Esa variabilidad impide sobreinterpretar celdas cercanas al umbral, pero no invalida las celdas con separación total `min(shared) > max(Bclone)` ni el patrón coherente de T0-RM.

Este resultado cierra la pregunta física de v0.4.0 en ese hardware y kernel int8:

> La residencia por profundidad y la matrixización por slots sobreviven a una transición recurrente real cuando la geometría y la implementación respetan el contrato del laboratorio.

No demuestra todavía calidad lingüística, entrenabilidad, memoria externa útil ni equivalencia con un Transformer.

## Hallazgos de tooling y numerics

1. La receta de build estaba fijada a VS2022/v143; el operador tuvo que usar VS18 Build Tools + Ninja. v0.4.1 autodetecta esa ruta.
2. La auditoría MSVC confundió saves/restores XMM6–XMM15 del ABI x64 con spills del hot loop. v0.4.1 limita el rechazo a la ventana aritmética.
3. Los scripts standalone T0-R/T0-M estaban fijados a `D=512`; v0.4.1 expone `-D`, `-SquareOutput` y `run_exact_bridges.ps1`.
4. El microbenchmark de transición de v0.4.0 encadenaba 1000 updates sintéticos sin reset y produjo 93–94% de clipping fixed-point. Eso caracteriza deriva bajo forzamiento repetido, no clipping de una trayectoria T0-RM de ocho rondas.
5. Una evaluación determinista de v0.4.1 sobre `D=1472,R=8` mostró que fixed shift 12 sí saturaba materialmente (~25%), shift 13 quedaba alrededor de 3%, y shift 14 eliminaba prácticamente el clipping. Por eso v0.4.1 usa fixed=14 y conserva RMS=12 como defaults de probe; el sweep real sigue siendo autoritativo.

## Revalidación dirigida de v0.4.1

```powershell
./scripts/run_v041_patch_validation.ps1 `
  -Cpus "0,2,4,6" -Rates "42.9,30.4,33.7,33.7"
```

Esta corrida verifica build adaptable, auditoría ABI, puentes exactos `D=1472`, T0-RM `R=8`, cadenas de transición 1/8 y fixed shifts 12–15 sin repetir toda la campaña original.
