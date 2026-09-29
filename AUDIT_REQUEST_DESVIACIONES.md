# Solicitud de auditoría de desviaciones OMEGA (juez -> Sol)

Fecha: 2026-09-29. Origen: observaciones del usuario, verificadas parcialmente por el juez contra Conversacion.md.
Estado operativo mientras dure la auditoría: SIN nuevo cómputo de entrenamiento OMEGA, SIN interpretar resultados d128 como veredicto, SIN T3. Solo termina la Fase B de calibración de instrumentos (no usa OMEGA entrenado).

## Observaciones del usuario

1. El núcleo pequeño debía ser como mínimo ~7M parámetros de cómputo (más en producción, mientras quepa en L2). La campaña actual entrena y mide con d=128, núcleo procedural ~0,6M; el "7M" salía de sumar el embedding atado (~6,43M, ~92%), que no es cómputo recurrente. Un resultado negativo aquí puede deberse a esa desviación y no a la hipótesis OMEGA.
2. La apuesta central de OMEGA: un núcleo pequeño, muy rápido, residente en L2, que compensa su menor tamaño con la profundidad que dan las rondas K de cómputo, es decir, K como mecanismo de compensación en inferencia. El usuario detecta desvíos graves respecto a eso. Un núcleo pequeño con desventaja de tamaño era esperado; lo que se apostaba es que K la cubre.

## Evidencia del juez en Conversacion.md (líneas del archivo en main)

- 2603-2608: tabla de tamaños del núcleo compartido: d=384 -> 2,36M; d=512 -> 4,19M ("candidato principal L2"); d=640 -> 6,55M ("frontera L2"); d=768 -> 9,44M (L3); d=1024 -> 16,78M (DRAM).
- 1293: presupuesto inicial de núcleos de ~7-9 MB reales; 1781: el halting da profundidad dinámica; 2469: K aporta composición secuencial, slots m aportan workspace, RAM aporta conocimiento.
- Registro de campaña: T0-M PASS_STRONG solo en D512; T0-R aislamiento causal no establecido; Bridge-1 FAIL_RECORDED; puente d512->d128 sin establecer; K4>K1 a 2k updates no estable entre presupuestos; m sin ejecutar; estado W_slot diferido.
- El juez NO encontró en Conversacion.md la frase explícita "entrenar a un K y subir K en inferencia". Si existe en otra fuente (usuario, otro documento), indicarla. Lo que sí está: profundidad dinámica + K como cómputo.

## Lo que se pide (a fondo)

A. Tabla de desviaciones. Recorrer Conversacion.md completo y cada unidad ejecutada de la campaña (T0..T2, Bridge-1, backend qualification, K-curve, Coverage A/B/C, TTR-A, state diagnostics, LN). Para cada decisión clave: qué dictaba Conversacion.md (con línea), qué se hizo, si es DESVIACIÓN / ADAPTACIÓN JUSTIFICADA / CONSISTENTE, y qué conclusiones dependen de ella. Debe cubrir como mínimo: (1) tamaño real del núcleo de cómputo vs 4-7M+ de la línea original; (2) uso de d128 como surrogate y qué sí/no transfiere a d512+; (3) rol de K: entrenamiento a K fijo vs K como compensación en inferencia/halting; qué midió realmente TTR-A; (4) m/slots y dinámica; (5) embedding atado y qué significa "parámetros" en la comparación con baselines; (6) residencia L2 (¿algún gate midió que el núcleo real cabe en L2 y es rápido?).

B. Invalidación explícita. Lista de conclusiones de la campaña que quedan INVALIDADAS o DEGRADADAS por las desviaciones (por ejemplo K4 selected, SATURATION-BEYOND-K4, QUALIFIED_AT_2000, coverage results, y el diseño del gate LN en cuanto asume el modelo d128).

C. Plan corregido, alineado con la línea original: núcleo de cómputo de 4-7M+ (d512/d640) que compense tamaño con rondas K. Debe definir: experimentos mínimos y su orden; cómo se prueba K como compensación (mismo modelo entrenado, barrido de K en inferencia y/o halting, comparado contra baselines más grandes sin recurrencia y a igual FLOPs); presupuesto de cómputo por experimento (CPU local i7-13700F 32GB vs RunPod GPU) y criterio de parada; qué reutiliza del trabajo ya hecho (backend nativo, instrumentos de la Fase B, protocolo LN adaptado); qué se descarta.

D. Responsabilidades. Indicar qué decisiones del juez y del arquitecto permitieron la deriva, para corregir el proceso (p. ej., exigir en cada unidad una casilla de "conformidad con Conversacion.md: tamaño, rol de K, L2").

E. Regla de proceso hacia adelante: ninguna unidad nueva se autoriza sin declarar explícitamente su distancia respecto de la línea original.

Formato de respuesta: fichero Markdown completo, sin truncar (la última respuesta de Sol llegó cortada). Si es muy largo, dividir en partes numeradas y avisarlo.
