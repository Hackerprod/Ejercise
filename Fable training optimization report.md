# OMEGA Training-Optimization Audit (Fable, hipótesis-libre, 2026-09-24)

Contexto: pase post-cierre del kernel (be37623d, STRONG). Lee Conversacion.md, T1.5_Spec_MIX_O.md, MD/146-263, IDEAS_FUTURAS.md y código real. Sin hipótesis pre-cargadas por mí. 10 hallazgos, ordenados por confianza del propio agente al cierre.

1. **Corpus trunca ~85% del texto ya descargado** (`RETAINED_TOKENS=513` en run_omega_ce_only_baseline.py:67/832). 602 docs revisados ~13.3x en 2000 updates. Overfitting ya medido en la rama equal-cost (MD/188, val NLL empeorando). Fix: retener múltiples chunks de 513 tokens por artículo (manifest ya soporta la clave). Nueva unidad, no parche a baseline congelado.

2. **Continuidad de estado tope en 512 tokens** — el estado recurrente (pieza central de la tesis de OMEGA) nunca se entrena sobre dependencias más largas, consecuencia directa del hallazgo 1. MD/146 quería medir "degradación con longitud" y nunca se hizo.

3. **LR constante 3e-4 sin warmup/decay/WD desde el preflight técnico** (MD/148), nunca tocado en ninguna unidad científica. Sol la marcó como deuda pendiente en el cierre de CE-only (MD/188). Tres síntomas documentados coinciden con falta de decay (degradación tardía CE-only, oscilación del gap K1-K4 en SCOPE-B/C, reversión en ER64). Intervención más barata de la lista.

4. **Scheduled sampling reactivado con evidencia real**: CE-only mostró que NLL y calidad de generación libre se desacoplan (MD/189, distinct-1 0.118→0.396). Ya estaba en la receta original T2 (Conversacion.md) y el instrumento de medición (generation-audit) ya existe congelado.

5. **Término λ_h (alineación de estado oculto al teacher) casi gratis ahora** — la infraestructura de hidden-cache (MD/195-198) ya cachea en RAM lo que este término necesitaría; estaba diferido desde el diseño original por costo, que ya no existe.

6. **Loss schedule CE-only→destilación tardía**: síntesis directa de MD/188 (CE-only aprende rápido y barato, destilación estabiliza tarde). Nunca propuesto como híbrido.

7. **Diagnósticos causales de uso del estado (anchor_reset/state_shuffle, gates post-entrenamiento) diseñados pero nunca ejecutados** — costo cero sobre checkpoints ya congelados. K4 ya arranca con ventaja en update 0 en las 3 seeds (posible artefacto de inicialización, nunca descartado).

8. **Validación primaria (8 docs) más ruidosa que el efecto medido** — oscilaciones ±0.03-0.05 del mismo orden que el gate de 0.05. El set de 60 docs ya existe y está infrautilizado.

9. **Recalcular costos con el kernel nuevo + correr ambos brazos en el mismo backend** para exploración (no para los gates de adopción) — 21.6% más updates/hora gratis, sin tocar los pendientes de trayectoria/calidad.

10. (Menor, más especulativo) Curriculum K2→K4→K6 de la receta original nunca probado; interacción rango×profundidad de ER32/ER64 (rank-32 amplificó la ventaja K4, rank-64 la borró) como única palanca observada que fortalece la señal de profundidad.

**Lo que NO encontró**: nada que justifique nuevas familias de optimizador, cambios de precisión, barridos de batch-size, o reescritura arquitectónica.

Enviado a Sol 2026-09-24 (MD/265: retenido como backlog, sin autorizar ejecución -- ver correcciones abajo).

## Ronda 2 (2026-09-24) -- 4 hallazgos nuevos, verificados contra datos crudos donde fue posible

11. **VERIFICADO contra ledger.json real**: CLIP_NORM=1.0 (MD/148 pidió loguear si intervenía, nunca se reportó el resultado en MD 146-237). Distillation R1 clipea ~14-37% de updates (guardrail ocasional). CE-only Phase A clipea 97.6-99.7%. Continuación equal-cost clipea **100.0%** de 4254 updates (confirmé: mediana pre_clip_grad_norm=1.77, exacto). CE-only entrena bajo renormalización casi constante -- régimen de optimización distinto al que se compara. K4 clipea más que K1 en ambos regímenes.

12. **Confirmado en disco**: brazos equal-cost corrieron 2254-4254 updates ciegos -- 1 solo checkpoint (el final) por carpeta, nada entre update 2000 y el final. 3/4 brazos terminaron peor que su propio valor en 2000. Determinista, recuperable re-corriendo con cadencia densa desde checkpoint_02000.pt existente.

13. La réplica de 2 seeds nunca varía orden de datos -- asignación de documentos por aritmética modular fija, sin shuffle/seed. Varianza seed-a-seed observada podría subestimar varianza real de corrida a corrida.

14. Refinamiento del hallazgo 2: horizonte de gradiente entrenado es 256 tokens (no 513) -- `pair_state.detach()` corta gradiente en cada frontera de ventana (protocolo deliberado, MD/165/174). Arreglar solo el truncamiento de datos NO entrena continuidad de estado por sí solo; necesita cruzar esa frontera o usar λ_h como crédito cruzado. Único experimento de horizonte (TBPTT, horizon=16, más corto) cerró DIRECTIONAL-STOP; horizon=512 (cruzando frontera) nunca se probó.

El propio agente evaluó que el modo de auditoría solo-lectura está cerca de agotado -- más señal requeriría EJECUTAR diagnósticos ya propuestos, no más lectura. No se lanzó ronda 3 de pura lectura. Backlog: 14 items totales, disponible para cuando la campaña retome entrenamiento.

Enviado a Sol 2026-09-24 (MD/265 respondió ronda 1: backlog retenido, sin autorización de ejecución; correcciones: 21.6% menos tiempo/update = 27.6% más updates/hora, no 21.6%; cobertura de corpus y continuidad de estado son palancas distintas; hidden-cache abarata λ_h pero no lo hace gratis; ruido ±0.03-0.05 necesita chequear método antes de aceptarlo).
