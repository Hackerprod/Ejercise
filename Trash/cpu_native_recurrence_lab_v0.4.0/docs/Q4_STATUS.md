# Estado de Q4

El camino autoritativo de este repositorio es **int8/AVX2**. Esta decisión es deliberada, no una omisión accidental.

El primer probe Q4 desempaquetaba nibbles, convertía a FP32 y usaba FMA. Su consumo de pesos era demasiado bajo para alcanzar el límite de DRAM, por lo que no podía aislar residencia. Ese código se conserva en `legacy/original_probes/q4_probe.cpp`, excluido del build.

Q4 solo debe volver al camino autoritativo después de que T0-RM cierre con int8. El siguiente kernel Q4 tendrá que cumplir simultáneamente:

1. mismo registro de kernel para frozen y recurrencia real;
2. `shared` y `Bclone` byte-idénticos;
3. dequantización de cada tile una vez y reutilización entre slots;
4. oráculo numérico separado, con tolerancia declarada;
5. bytes físicos del formato — nibbles, escalas, padding y metadata — reportados desde la asignación real;
6. throughput de pesos suficiente para que Bclone pueda alcanzar DRAM;
7. auditoría de ensamblado que demuestre que el hot loop no retrocedió a Q4→FP32 escalarizado.

Hasta cumplirlos, presentar una cifra Q4 como validación del mecanismo volvería a mezclar coste de dequantización con jerarquía de memoria.
