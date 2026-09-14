# Optimización del núcleo recurrente OMEGA CORE-LM-0 R1 (CPU y GPU)

Análisis de `t1_trainability_lab_v0.1.0` (commit `afe69f7`): `WorkspaceUpdateBlock`
(`scripts/audit_omega_core_lm_0_design.py`), `OmegaCoreLM0R1Technical`
(`scripts/run_omega_core_lm_0_r1_training_technical_preflight.py`) y el runner
`campaign/omega_core_lm_0_r1_scientific_scoping_a/run_scientific_scoping_a.py`.

## 1. Dónde se va el tiempo hoy (tu propia campaña CPU, `cpu_scoping_report.json`)

Por update (batch 8 × 2 ventanas × 256 tokens = 4 096 tokens), Zen 5, 4 hilos, fp32, eager:

| Fase | K1 | K4 | Comentario |
|---|---|---|---|
| `student_backward` | ~4.8 s | ~5.7 s | ~50 %. Autograd sobre ~18 000 nodos/ventana + backward del vocabulario |
| `teacher_forward` | ~3.1 s | ~3.0 s | 30 %. **100 % cacheable**: corpus fijo, profesor congelado |
| `readout` | 0.85 s | 0.9 s | Se hace 256 veces por ventana, con GEMM [B,128]×[128,50257] cada vez |
| `ce_kl_softmax` | 0.5 s | 0.5 s | 4 tensores [2048, 50257] fp32 (411 MB c/u) materializados |
| `student_recurrent_rounds` | 0.25 s | 0.8 s | **El núcleo en sí es lo más barato** |
| **Total** | **~9.8 s** | **~11.2 s** | proyección SCOPE-A: 24.6 h |

Dos hechos que no son evidentes desde el perfil:

1. **El runner de SCOPE-A no usa batch 8.** `_batch_loss` recorre los 8 documentos con
   `batch=1` y llama a `forward_window` 8 veces (y al profesor 8 veces). Toda la
   arquitectura ya soporta `[B, T]`; con B=8 real el coste por update cae ~3× medido
   (kernel-overhead bound: el coste de un op `[1,8,128]` y uno `[8,8,128]` es casi igual).
2. **El profesor recalcula la ventana 0 dos veces** (`teacher_window_logits` usa contexto
   0:256 y luego 0:512). Con máscara causal, las posiciones 0:256 del contexto 0:512 son
   las mismas. Una llamada y dos cortes.

Conteo de ops por token en el forward original: prelude (6) + K×16 + readout (4)
≈ 74 ops con K=4 → 19 000 ops/ventana en forward, ×2–3 en backward. En CPU cada op
pequeño cuesta 15–40 µs de overhead → de ahí salen los segundos; en GPU cada
lanzamiento cuesta 5–10 µs y la GPU queda >99 % ociosa con B=8.

## 2. Resultados medidos (sandbox Linux, 1 core, T=64, B=8, K=4, V=50257, fwd+bwd)

| Configuración | s/update | tok/s | vs runner |
|---|---|---|---|
| Referencia, modo runner (8 × batch 1) | 6.69 | 77 | 1.0× |
| Referencia, batch 8 | 2.03 | 252 | 3.3× |
| `omega_fast` (núcleo fusionado + readout hoisted + pérdida LSE), batch 8 | 1.28 | 402 | **5.2×** |
| `omega_fast` + `torch.compile` (inductor CPU) | 1.27 | 402 | 5.2× (sin ganancia extra) |

Equivalencia contra el original (`bench_omega_fast.py`): loss idéntica en fp32,
gradientes con error relativo máx. 5.7e-7 (todos los parámetros), pérdida LSE con
error 1e-6. Cumple tu tolerancia 1e-5 de la política T0-TR-C.

Componentes aislados (1 hilo): recurrencia fwd+bwd 0.635 → 0.407 s (1.56×);
readout+pérdida 1.58 → 0.90 s (1.76×). En tu Zen 5 con 8+ hilos la parte de
vocabulario se acelera más que la recurrente (los GEMM grandes escalan; los ops de
`[8,8,128]` no), así que espera que la recurrencia pase a ser la fracción dominante
tras aplicar esto.

## 3. Qué hace `omega_fast.py` (todo exacto, mismo modelo, mismos parámetros)

| # | Cambio | Ops/token que elimina |
|---|---|---|
| 1 | Embedding + parte "token" de `prelude` (`W[:, :D]·e_t + b`) en un GEMM sobre los T tokens | 3 |
| 2 | Readout fuera del bucle: apilar estados `[B,T,S,D]`, un RMSNorm y **un** GEMM con el vocabulario | 4 |
| 3 | Q/K/V fusionados `Linear(D,3D)` | 2 por ronda |
| 4 | `F.scaled_dot_product_attention` sobre los 8 slots (1 kernel vs 3) | 2 por ronda |
| 5 | `F.rms_norm` fusionado (1 kernel vs 5) | 4 por ronda + 8 en prelude/readout |
| 6 | Depth-embedding plegada en el bias de `fc1` (`fc1(x+d) = fc1(x) + W1 d`), precomputada por ventana | 1 por ronda |
| 7 | `torch.addcmul(state, gate, update)` | 1 por ronda |
| 8 | `sigmoid(gate_logits)` y `W_sum` una vez por ventana, no por token | 2 por ronda |
| 9 | Pérdida con `logsumexp`: `CE = lse(z) − z[y]`, `T²KL = T²[Σp_t log p_t − p_t·z/T + lse(z/T)]` | 2–3× menos pasadas sobre `[N,V]`; `Σ p_t log p_t` es constante del profesor → cacheable |

Resultado: ~74 → ~26 ops/token con K=4, y el bucle secuencial ya no toca el vocabulario.

`bench_omega_fast.py` reproduce la verificación: `python bench_omega_fast.py --T 256 --B 8
--threads 8` en tu máquina, y `--device cuda --compile` en GPU.

## 4. Plan CPU (Windows, Zen 5, torch 2.11+cpu)

Ordenado por (ganancia / esfuerzo). Los niveles A son exactos; B cambian bits pero
no la semántica; C cambian el experimento.

**A. Exactos (aplicar ya)**
1. Batch real de 8 en `_batch_loss` (3×, es un bug de rendimiento, no de diseño).
2. Una sola llamada al profesor por documento (contexto 0:512, dos cortes): −50 % profesor.
3. **Cache del profesor** (`build_teacher_cache`): `p_t` en fp16 + `Σ p_t log p_t` en fp32 en
   un `np.memmap`. Corpus WikiText-2 elegible ≈ 400 docs × 512 × 50257 × 2 B ≈ 20 GB en disco;
   lectura por update 411 MB (≈0.15 s en NVMe) frente a ~3 s de forward. Elimina el 30 %
   del tiempo y además el profesor deja de ocupar RAM/CPU durante el entrenamiento.
   Si 20 GB molestan: cachea solo los 8 docs del scoping (411 MB) o guarda los logits en
   fp16 y calcula `teacher_targets` al vuelo (más barato que el forward igualmente).
4. `omega_fast.py` (núcleo fusionado + readout hoisted + pérdida LSE): 1.6× adicional medido con 1 hilo.
5. **Hilos**: `torch.set_num_threads(1)` dentro de `recur_states` y el máximo de cores físicos
   para `project`/`logits`/pérdida. Con tensores `[8,8,128]` el fork/join de OpenMP cuesta
   más que el cálculo; con 4 hilos ya estabas pagando eso. Mide `--threads 1/2/4/8` con el bench.
6. **Paralelismo entre runs**: SCOPE-A son 2 seeds × 2 variantes = 4 procesos independientes.
   Lánzalos a la vez (`start /B` o PowerShell `Start-Process`) con 2–4 hilos cada uno y
   `OMP_NUM_THREADS` acorde. En una CPU de 16 hilos esto es casi 4× de wall-clock gratis.
7. `torch.compile` en CPU: **medido, no aporta nada sobre `omega_fast`** (1.273 s vs
   1.275 s, equivalencia 6.7e-7) y tarda minutos en compilar porque desenrolla el bucle
   de T pasos. La fusión manual ya captura lo que inductor fusionaría. No merece la pena
   instalar MSVC (`cl.exe`) solo para esto; si aun así quieres probarlo: *VS Build Tools
   2022 → C++* y lanzar desde *x64 Native Tools Command Prompt*, compilando solo el paso
   por token (no `recur_states` entero) para que la compilación sea corta.

**B. Cambian bits (con gate de tolerancia como ya haces en T0-TR-C)**
8. `torch.autocast("cpu", dtype=torch.bfloat16)` **solo** alrededor de `project` +
   `logits` + pérdida (Zen 4/5 tienen AVX-512 BF16: oneDNN GEMM 2–4× más rápido).
   Mantén la recurrencia en fp32: 256 tokens × 4 rondas = 1 024 RMSNorm encadenadas
   drifean en bf16.
9. Profesor en bf16 antes de cachearlo (una vez; el error va al cache, no a cada update).

**C. Cambian el experimento (registrar como variante nueva)**
10. TBPTT por chunks de 64 dentro de la ventana (misma ventana de 256, gradiente
    truncado a 64): reduce memoria y permite liberar grafo por chunk; no reduce ops.

Expectativa realista en tu máquina: 9.8 s → ~1.5–2.5 s/update solo con A1–A6
(3× batch, −3 s profesor, 1.6× núcleo, hilos bien puestos), y las 24.6 h proyectadas
de SCOPE-A a **~1.5–3 h** contando los 4 procesos en paralelo.

## 5. Plan GPU

En GPU el problema no es FLOPs (el modelo son ~6.6 M parámetros + 6.4 M de embedding),
es **latencia de lanzamiento**: ~74 kernels/token × 256 × (fwd+bwd) ≈ 40–55 k lanzamientos
por ventana. A 6 µs cada uno son 0.3 s por ventana independientemente de B. La GPU
está vacía y el tiempo es idéntico con B=8 y con B=256.

1. Todo lo de la sección 3 (menos ops = menos lanzamientos). `F.rms_norm` y SDPA
   importan aún más aquí.
2. **CUDA Graphs** sobre `recur_states` con shapes estáticas (B, T fijos, sin
   `valid_mask` de shape variable): `torch.compile(fast.recur_states,
   mode="reduce-overhead", dynamic=False)` o, sin Triton (Windows nativo),
   `cuda_graphed_recurrence(...)` en `omega_fast.py`, que usa
   `torch.cuda.make_graphed_callables` y captura forward y backward. Esto convierte los
   ~50 k lanzamientos en 1 replay. Es la optimización que más importa en GPU: 5–20× en
   la recurrencia.
3. Cache del profesor en fp16 en VRAM o RAM pinned (411 MB por update de 8 docs), o el
   profesor en bf16 con SDPA si no cabe.
4. `torch.backends.cuda.matmul.allow_tf32 = True` / `set_float32_matmul_precision("high")`:
   hoy lo tienes en `False`/`highest`. El GEMM del vocabulario y el backward del
   embedding atado (`[50257,128]` denso) ganan 3–8×. Error ~1e-3 relativo: no pasa tu
   gate 1e-5, así que decidirlo explícitamente.
5. **Amortizar el batch**: como el coste es plano en B, entrena las 2 seeds (misma
   variante) **en el mismo proceso** con `torch.func.stack_module_state` + `vmap`
   sobre los modelos, cada uno con sus datos y su AdamW (`torch.optim` con
   `foreach`/`fused=True`). Matemáticamente idéntico a dos runs separados; solo
   comparten lanzamientos. Con K1 y K4 no se puede vmap (programas distintos), pero
   sí las dos seeds de cada variante → mitad de wall-clock.
6. Pérdida: en GPU la versión LSE ya es suficiente; si B·T crece (≥16 k tokens) usa
   `distillation_loss_chunked(use_checkpoint=True)` para no materializar `[N,50257]` fp32.
7. Linux/WSL2 para el pilot GPU: Triton (y por tanto `torch.compile` con inductor) es
   frágil en Windows; `make_graphed_callables` funciona en ambos.

## 6. Opciones arquitectónicas (paralelizar el tiempo; cambian el modelo → variante nueva)

La recurrencia token→token es totalmente no lineal (`anchor = RMSNorm(s_{t-1} + …)` y
luego K rondas de atención+MLP), así que ningún truco de implementación la paraleliza
en T. Si quieres orden de magnitud adicional, hay que tocar el diseño:

- **Carry lineal + workspace paralelo** (idea minGRU/Mamba-2): que el estado que cruza
  tokens sea `s_t = a_t ⊙ s_{t-1} + (1−a_t) ⊙ u_t`, donde `u_t` y `a_t` salen de las K
  rondas aplicadas al *token* (y a un contexto calculable en paralelo). Las K rondas —
  la parte cara y la que define "profundidad recurrente" — se ejecutan sobre
  `[B·T, S, D]` de golpe (4 lanzamientos grandes en vez de 1 024 pequeños), y lo único
  secuencial es un scan asociativo (log-depth en GPU; bucle trivial en CPU). Conserva
  estado persistente y K rondas compartidas; pierde la dependencia no lineal del
  workspace respecto al estado anterior. Es la variante honesta si el objetivo es
  velocidad en ambos dispositivos.
- **Resolución paralela en el tiempo (DEER / Newton-Picard)**: tratar `s_1..s_T` como
  incógnitas de un punto fijo y resolverlo con iteraciones que evalúan el paso para
  todos los t a la vez (`[B·T, …]`). Converge en 10–30 iteraciones frente a 256 pasos
  secuenciales. Mismo modelo exacto en el punto fijo; solo tiene sentido en GPU (en CPU
  hace más FLOPs que el bucle) y hay que validar convergencia con tus normas encadenadas.
- **Rondas con parada anticipada / gate de profundidad** (K adaptativo): con `gate≈0.1`
  inicial, medir cuántas rondas cambian el estado más que ε y saltar el resto; ahorra en
  inferencia y en el forward, no en el backward eager (sí con graphs si se fija K).

## 7. Prioridad recomendada

1. Batch real + una llamada al profesor + cache del profesor (exactos, 1 tarde, ~4×).
2. `omega_fast` con gate de equivalencia (exacto, medido 1.6× adicional).
3. CPU: 4 procesos en paralelo + hilos bien puestos. GPU: CUDA Graphs.
4. Después, si hace falta más: bf16/TF32 con gate de tolerancia, y solo entonces la
   variante de carry lineal como experimento separado.
