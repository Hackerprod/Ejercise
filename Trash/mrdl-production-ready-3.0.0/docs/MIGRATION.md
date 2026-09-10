# Migración desde la implementación parcial anterior

El documento base menciona una implementación previa en `/root/mrdl` de stages sintéticos. Este repositorio no presupone su esquema interno ni intenta reinterpretar sus objetos.

## Decisión segura

No existe migración automática binaria o SQLite desde la versión anterior. El nuevo runtime usa:

- índices físicos FULL/CLEAN;
- `RelationRecord` versionado;
- snapshots de replay;
- máquina de estados de promoción integrada;
- tokenizador y embeddings propios;
- metadatos de esquema `MRDL-3.0-production-core`.

Importar objetos desconocidos sin conocer su versión exacta podría violar no-lavado o fabricar una clausura de replay falsa.

## Ruta recomendada

1. Conservar `/root/mrdl` intacto y crear un backup.
2. Clonar/descomprimir este proyecto en otra ruta.
3. Ejecutar sus tests y benchmarks sin tocar el modelo anterior.
4. Preparar un modelo nuevo desde el corpus original.
5. Comparar los benchmarks sintéticos anteriores con `mrdl_bench`.
6. Solo después, entrenar el núcleo real.

## Migrador futuro

Un migrador seguro necesitaría, como mínimo:

- código o esquema exacto de la versión fuente;
- mapping de niveles M0/M1/M2;
- versión de cada relación/controlador;
- capacidad de reconstruir replay o marcar explícitamente datos no promovibles;
- prueba A de equivalencia por registro;
- prueba B/C de no-interferencia;
- rollback y reporte de todos los registros descartados.

Si el origen no contiene replay completo, las relaciones M1 pueden importarse únicamente como observaciones no promovibles o descartarse. Nunca deben elevarse a M2 por conveniencia.
