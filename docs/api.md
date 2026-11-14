# API y acceso

## Conexión

- Base por defecto: `https://api.rawtree.com`; configurable con `RAWTREE_API_URL`.
- Header: `Authorization: Bearer <RAWTREE_API_KEY>`; nunca imprimirlo ni guardarlo en notebooks.
- `TEAM_ID` selecciona la base privada del equipo (`team_NN`). Permiso del reto: leer `shop`; leer/escribir la base propia.
- Selección explícita: parámetro URL `database=team_NN` o header `x-rawtree-database`. Si se usan ambos, deben coincidir. Sin selector se usa el default de la clave.
- MCP alternativo: `https://mcp.rawtree.com/mcp`. CLI alternativa: `rtree`. El notebook usa HTTP.
- Referencia oficial de contrato HTTP: [OpenAPI](https://api.rawtree.com/v1/openapi.json); pendiente de contrastar en directo.

## Consultar

```http
POST /v1/query?database=team_NN
Authorization: Bearer <secreto>
Content-Type: application/json

{"sql":"SELECT request_id, ts FROM shop.requests ORDER BY ts::DateTime64(3) DESC LIMIT 5","format":"JSON","query_id":"<uuid>"}
```

`sql` es obligatorio; `format` y `query_id` son opcionales según la guía local. No hay binding SQL mediante un campo `params` en el body. El `params` de requests para seleccionar `database` corresponde a la URL y es diferente. Validar identificadores y escapar literales, nunca interpolar entrada arbitraria.

La respuesta JSON contiene `data` (lista de filas), `meta` (columnas/tipos), `rows`, `statistics` y, cuando aplica, `hints`. El notebook devuelve `data` como DataFrame y muestra métricas de ejecución. Resultado vacío sin meta usada produce un DataFrame sin columnas; no implica que la tabla carezca de esquema. Otros formatos pueden devolver cuerpos sin JSON.

Comprobar estado HTTP antes de parsear JSON. El mensaje del servidor y hints son parte del diagnóstico: éxito HTTP puede coexistir con variantes Dynamic omitidas. No asumir éxito tras timeout. No reintentar indiscriminadamente consultas costosas. Para problemas de acceso revisar clave, base y permisos; no cambiar automáticamente a otra base.

## Insertar respuestas — contrato, no acción de exploración

```http
POST /v1/tables/submissions?database=team_NN
Authorization: Bearer <secreto>
Content-Type: application/json

[{"request_id":"r_...","items":["p_1","p_2","p_3","p_4","p_5","p_6","p_7","p_8","p_9","p_10"],"model_version":"nexitos-v1"}]
```

Acepta objeto JSON o array; agrupar respuestas reduce overhead. Insertar crea la tabla si no existe según la guía RawTree. Cada petición debe recibir una sola respuesta. Las inserciones son append-like: un timeout puede haber escrito datos y repetir puede duplicarlos. No se ha verificado un mecanismo de deduplicación remoto; no inventar headers de idempotencia. El notebook de consultas no inserta nada.

## SQL: cómo descubrir el esquema de verdad

1. `DESCRIBE TABLE shop.requests SETTINGS describe_include_subcolumns = 1` descubre nombres y rutas.
2. Muestrear columnas explícitas con orden y límite para ver estructuras y valores.
3. `dynamicType(campo)` agrupado con `count()` descubre variantes almacenadas.
4. Contar nulos y validar rangos/cardinalidades antes de elegir casts y joins.

`Dynamic` es un contenedor de tipos, no un tipo de negocio. `is_subcolumn=1` significa campo descubierto dentro del JSON. `context.device` es una ruta anidada; no presupone que un objeto `context` pueda seleccionarse igual que sus hojas. Campo conocido ausente produce NULL; ruta totalmente desconocida puede fallar con `UNKNOWN_IDENTIFIER`, especialmente en streams todavía vacíos.

- Identificadores: normalizar a String antes de agrupar/unir; descartar nulos antes de casts estrictos, pues pueden convertirse a `''` y crear coincidencias falsas.
- Tiempos: el starter usa `ts::DateTime64(3)`; auditar tipo y precisión si el orden entre eventos importa. No inferir unidades si aparecen números epoch.
- Valores sucios: `accurateCastOrNull(campo, 'Float64')`, contando fallos explícitamente; puede reducir pruning de índices.
- Arrays: comprobar variante antes de `CAST(... AS Array(String))`. Índices SQL empiezan en **1**, índices Python en **0**.
- Agregaciones numéricas y `argMax` necesitan intención de tipo explícita. `countIf(campo IS NOT NULL)` mide presencia; `count(campo)` sobre wrappers Dynamic no es una medida fiable de no nulos.
- `LIMIT` limita retorno, no filas leídas. Ventanas temporales, columnas concretas y agregación temprana reducen trabajo. Co-vistas sobre toda la historia puede ser costoso.
- Medir `statistics` y `hints`; `EXPLAIN` ayuda a entender el plan. Comparaciones de rendimiento deben usar rangos fijos, no `now()` cambiante.

## Notebook de requests

La celda maestra instala requests/pandas faltantes en el intérprete del kernel usando uv o pip, lee el primer `.env` encontrado desde cwd hacia sus padres sin reemplazar el entorno, y solicita credenciales ausentes. El lector `.env` soporta KEY=value, comillas y export, pero no expansión de variables. Si se abre desde fuera del proyecto, establecer las variables explícitamente.

Después cada consulta depende solo de esa celda maestra y de sus parámetros locales. `query(sql, timeout=60)` muestra HTTP, tiempo, query_id, estadísticas y hints, y devuelve DataFrame. No contiene cancelación ni retries automáticos. Reiniciar kernel requiere ejecutar de nuevo la maestra.
