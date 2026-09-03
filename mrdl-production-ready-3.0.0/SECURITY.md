# Seguridad e integridad operativa

MRDL procesa corpus y archivos de modelo locales. No expone servidor de red en esta versión.

## Reporte de fallos

Antes de publicar el repositorio, configure un canal privado de reporte. No publique corpus privados, bases SQLite ni snapshots de replay en un issue público.

## Reglas de operación

- Ejecute el runtime con un usuario sin privilegios.
- No permita escritura de usuarios no confiables en el directorio del modelo o configuración.
- Valide corpus y embeddings externos fuera del runtime si provienen de terceros.
- Mantenga backups fuera del VPS.
- No ignore fallos de checksum, `integrity_check` o estado `UNREPLAYABLE`.
- No ejecute dos escritores sobre el mismo modelo; el lock lo bloquea, pero no sustituye permisos correctos.
- El texto generado es experimental y no debe tratarse como fuente fiable sin evaluación independiente.

## Límites

Los `hash64` y checksums internos detectan corrupción accidental; no autentican archivos frente a un atacante con acceso de escritura. Para distribución o transferencia use permisos, hashes criptográficos externos o firmas.

Los formatos binarios están diseñados para Linux little-endian de 64 bits. No se promete compatibilidad con archivos manipulados, arquitecturas big-endian ni versiones futuras sin migración explícita.
