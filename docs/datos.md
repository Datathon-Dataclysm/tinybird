# Diccionario de datos y relaciones

Tipos de negocio esperados según el contrato, no tipos físicos garantizados. Las tablas son JSON schemaless con campos Dynamic. IDs opacos: no extraer significado de sus prefijos ni su orden.

## Tablas y granularidad

| Tabla | Una fila representa | Disponibilidad |
|---|---|---|
| `shop.catalog` | Producto en catálogo inicial | Desde el inicio, unos 5.000 productos, 10 categorías y 90 subcategorías según contrato |
| `shop.catalog_updates` | Cambio con producto completo actualizado | Stream append-only |
| `shop.history_events` | Evento orgánico histórico, ajeno al carrusel | 60 días, unos 12 millones de filas según contrato |
| `shop.events` | Evento orgánico en directo | Stream append-only, mismo formato |
| `shop.history_sessions` | Impresión histórica de carrusel, no una sesión completa | 60 días, unos 2 millones de filas según contrato |
| `shop.requests` | Petición de recomendación | Stream append-only, unos 3/s |
| `team_NN.submissions` | Respuesta enviada por el equipo | Escritura propia |
| `team_NN.feedback` | Resultado de una petición | Privado; unos segundos después del deadline |
| `team_NN.scratch` | Objeto arbitrario del equipo | Espacio auxiliar, no dataset de negocio predefinido |

## Catálogo

`product_id` string (clave lógica); `category`, `subcategory`, `brand`, `description` strings; `price_eur` número; `created_at` timestamp; `is_active` booleano.

Actualizaciones añaden `ts` (momento del cambio) y `change`: `new_product`, `price_change`, `out_of_stock`, `back_in_stock`. Contienen el producto completo. Estado actual = catálogo inicial con última actualización de cada producto superpuesta, incluyendo productos nuevos. Usar la versión completa de una misma fila; timestamps empatados necesitan desempate si se observan.

Para entrenamiento/evaluación reconstruir el catálogo **a fecha de la petición**, no con futuros cambios. Precio inicial, precio actual y precio observado en un evento son conceptos distintos. No recomendar inactivos aunque sean populares. Una unión solo con catálogo inicial pierde productos nuevos y usa atributos potencialmente obsoletos.

## Eventos orgánicos: history_events / events

`ts` timestamp; `event_type` string (`view`, `add_to_cart`, `purchase`); `user_id`, `session_id`, `product_id` strings; `price_eur` número del momento; `device`, `region` strings.

Clave de agrupación conservadora: `(user_id, session_id)`. No hay `event_id` en el contrato: eventos con el mismo producto y timestamp pueden tener tipos distintos, como muestran las salidas guardadas. No deduplicar solo por usuario/producto/ts. No está demostrado que eventos idénticos sean duplicados de transporte. Una compra orgánica no equivale a una conversión del carrusel y no proporciona impresiones negativas.

## Impresiones históricas: history_sessions

- `request_id`, `user_id`: strings; `ts`: timestamp.
- `context`: objeto con `device`, `region`, `page` (home/product/category).
- `recent_views`: array de objetos con `product_id` y `ts`.
- `shown`: array ordenado de 10 IDs mostrados por la política anterior.
- `propensity`: array de probabilidades, alineado con shown. Cada elemento es probabilidad condicional de mostrar ese ítem en ese slot dados los anteriores; no es probabilidad de compra.
- `clicked`, `purchased`: ID string o null; `revenue_eur`: número.

El contrato recoge un clic y una compra por impresión; no asumir múltiples resultados. La política anterior era best sellers con aleatoriedad: exposición fuertemente sesgada. Producto no mostrado no es un negativo observado. `recent_views` representa información anterior al request; ordenar por timestamps si el orden real no está garantizado.

**Incidencia observada:** evaluación del usuario encontró propensities inválidas. No se conoce aún si son nulos, arrays vacíos, ceros, NaN u otros valores, ni su frecuencia. El evaluador excluye de IPS/SNIPS el primer slot inválido y reporta cobertura; esto puede sesgar la estimación. No sustituir probabilidades por constantes. Auditar arrays completos antes de usarlos.

## Peticiones: requests

`request_id`, `ts`, `user_id`, `is_new_user` booleano, `context`, `recent_views` como arriba. `session_id` no figura en el contrato: no unir a una sesión concreta inventando una clave. El esquema real guardado confirma `context.device`, `.page`, `.region`, `is_new_user`, `recent_views`, `request_id`, `ts`, `user_id`, todos Dynamic.

Una petición cold según el reto: usuario nuevo **o** alguna vista reciente corresponde a un producto añadido durante el evento. No equivale simplemente a recent_views vacío. Las peticiones no traen candidatos ni producto actual explícito fuera de las vistas; no inventar ese campo.

## Respuestas y feedback

Submissions: `request_id` string, `items` array de exactamente diez IDs distintos y activos, `model_version` string público de nombre neutro. Sin timestamp garantizado; ordenar por request_id no reconstruye cronología fiable.

Feedback: `request_id`, `ts`, `shown`, `served_by` (`team` o `fallback`), `fallback_reason` (vacío o `missing`/`late`/`invalid`), `latency_ms`, `clicked`, `purchased`, `revenue_eur`. Sin resultado, clicked/purchased son `""` (a diferencia de null en historia). Ignorar `request_id='__init__'`. Feedback fallback describe lo mostrado por baseline, no necesariamente lo que envió el equipo.

## Relaciones seguras

- Requests ↔ submissions ↔ feedback: `request_id`; comprobar duplicados antes de unir para evitar multiplicar filas.
- Eventos, shown, items y recent_views ↔ catálogo: `product_id`, con versión temporal adecuada.
- Usuarios ↔ eventos/historia/requests: `user_id`, con corte temporal; no presupone que todos existan en historia.
- Sesiones orgánicas: `(user_id, session_id)`. No hay relación directa documentada entre session_id y request_id.
- Históricos y streams: comprobar sus límites temporales y posible solapamiento antes de concatenar.

El feedback no incluye contexto ni model_version según contrato: obtenerlos mediante request_id. Si hay varias submissions por petición, no atribuir automáticamente el resultado a una versión sin conocer cuál fue aceptada.
