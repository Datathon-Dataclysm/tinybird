# Base para el siguiente análisis y modelo

## Orden de trabajo

1. Auditar acceso, tablas, tipos reales, nulos, fechas mínimas/máximas, claves y cardinalidades. Guardar resultados agregados con fecha de captura, no credenciales.
2. Comprobar integridad de arrays y relaciones, cobertura de catálogo y cambios, duplicados y solapamiento histórico/stream. Diagnosticar propensities inválidas y consulta vacía antes de sacar conclusiones.
3. Analizar conversión por producto, categoría, contexto, recurrencia y antigüedad; separar orgánico, carrusel y fallback. Comprobar volumen e incertidumbre.
4. Construir features exclusivamente con información disponible al instante del request. Validación temporal, sin mezclar futuro en agregaciones ni catálogo.
5. Comparar modelos con mismo candidato, periodo y disponibilidad; medir cobertura, cold start, conversión observada cuando aplique y latencia. Reservar test final y evitar ajustar repetidamente sobre él.
6. Integrar servicio solo después de verificar validez y tiempos. La implementación del modelo y su análisis no se han realizado en esta tarea documental.

## Features candidatas, no resultados demostrados

| Familia | Datos | Feature candidata | Precaución |
|---|---|---|---|
| Popularidad | Eventos anteriores | Vistas, carritos, compras por ventanas; tendencia y decaimiento | No confundir compras orgánicas con tasa de conversión del carrusel |
| Afinidad entre productos | Sesiones orgánicas | Co-vistas/co-compras normalizadas y soporte | Deduplicar dentro de sesión; controlar explosión de pares |
| Intención actual | recent_views | Recencia, repeticiones, afinidad con últimas categorías y marcas | Solo vistas anteriores; array no garantiza orden por sí solo |
| Usuario | Eventos anteriores por user_id | Preferencias de categoría/marca/precio, compras previas | Usuario nuevo y sparsity; nunca agregar eventos futuros |
| Catálogo | Estado temporal | Categoría, marca, precio, texto, antigüedad, activo | Nuevos productos y stock requieren refresh |
| Precio | Catálogo/eventos | Diferencia con precios vistos, rango preferido, variación reciente | No hay descuentos ni margen explícitos |
| Contexto | Requests | Device, región, page, hora/día | Medir soporte; no asumir causalidad |
| Feedback | Feedback unido a requests | Rendimiento previo por ítem/contexto, exposición y tendencia | Retardo y sesgo de política; excluir fallback de atribución propia |
| Lista | Candidatos | Diversidad de categorías, redundancia, distribución de precios | Optimización de lista, no solo score independiente |

No hay ratings, demografía, margen, inventario cuantitativo, embeddings precomputados ni preferencias ocultas documentadas. Description permite generar representaciones, pero debe medirse si aportan valor.

## Evaluación existente y límites

Notebook Nexitos: último día histórico menos siete días como corte; entrenar con hasta 200.000 eventos previos más recientes, test hasta 100.000 impresiones con `ORDER BY rand()`. Puede truncar sesiones y el muestreo no es reproducible. Usa catálogo inicial, no reconstrucción por petición. La política comparadora es popularidad ponderada sobre esa misma muestra, **no el baseline oficial**.

Hit rate@10 condicionado a que haya clic: fácil, sesgado hacia exposición histórica. IPS primer slot usa coincidencia con el primer producto histórico y su propensity; SNIPS normaliza por suma de pesos. Son métricas de clic del primer slot, no una estimación completa de compras de una lista de diez. Pocos matches o pesos extremos dan alta varianza. Ausencia de soporte debe devolver NaN, no cero. Reportar coverage, matches y distribución de pesos junto a resultados. Excluir propensities inválidas cambia la población evaluada.

No hay aún evidencia de mejora real de Nexitos, ni benchmark de memoria/latencia. No desplegar por una subida aislada de hit rate. Para compra, estudiar labels purchased y soporte antes de trasladar un estimador de clic.

## Operación a resolver antes de live

- El starter marca requests como vistos al leerlos, antes de confirmar la escritura; un fallo puede perder respuestas. El estado seen crece sin límite y desaparece al reiniciar.
- Consultar solo las últimas 50 peticiones puede perder tráfico tras interrupciones. Se necesita cursor/ventana con deduplicación y control de deadlines.
- Timeout HTTP del starter (10s) supera el presupuesto de 3s; computación y operaciones remotas deben acotarse y desacoplarse.
- Baseline calcula ranking una vez y solo refresca cambios de stock; no incorpora bien todo el catálogo nuevo ni tendencias.
- Nexitos no consulta RawTree ni aplica updates automáticamente. Fit es batch; precomputar fuera del camino crítico y sustituir snapshots de forma coherente.
- Confirmar inserción por lotes, errores parciales, política de reintentos y persistencia sin publicar dos veces una petición. Sin soporte remoto verificado no prometer exactly-once.

## Decisiones para la siguiente fase

Objetivo por defecto: maximizar compra manteniendo diez productos válidos y entrega a tiempo; CTR es diagnóstico. Empezar con una referencia interpretable y features baratas disponibles online, antes de un modelo más complejo.

Antes de comprometer una decisión ambigua, preguntar de forma interactiva: presupuesto de cómputo/latencia y despliegue, cuánto historial descargar, si se permite explorar en directo, y si hay un objetivo distinto de conversión. Lecturas y experimentos locales pueden avanzar mientras se aclara lo independiente. No publicar respuestas ni lanzar un proceso live como efecto secundario del análisis. Mantener nombres públicos neutros.
