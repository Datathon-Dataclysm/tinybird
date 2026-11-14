# Contexto del proyecto

Documentación de trabajo para explorar datos, diseñar features y construir el siguiente modelo. Leer en orden: [API](api.md), [datos](datos.md), [análisis y modelo](analisis-modelo.md).

## Objetivo y restricciones

Reto de recomendación RawTree, Datathon FME 2026. Por petición hay que elegir **10 productos distintos y activos**, ordenados, y entregar una respuesta en menos de **3 segundos desde `request.ts`**, según el reloj del organizador. Aproximadamente 3 peticiones por segundo. La posición afecta la atención. Respuesta ausente, tardía o inválida implica fallback a best sellers.

El objetivo principal es conversión, no CTR. La puntuación del leaderboard normaliza aproximadamente 71% conversión, 14% cold start y 14% fiabilidad; la conversión se calcula con probabilidades del simulador, no con la compra aleatoria observada. El feedback sí muestra resultados realizados, por lo que necesita volumen. Ingresos por petición desempatan. La nota final incluye 70% competición y 30% presentación (método, aprendizaje online, uso de RawTree y comunicación).

Práctica: sábado 14 a domingo 15 de noviembre de 2026 a las 10:00 CET. Final: domingo 10:00–12:00 CET, con nuevos usuarios, productos y tendencias; leaderboard oculto. Las fechas de datos son UTC. El sistema debe funcionar sin intervención continua.

## Mapa del código

| Archivo | Función / estado |
|---|---|
| `../README.md` | Contrato del reto y fuente principal de semántica |
| `../requests.ipynb` | Consola HTTP de lectura: celda maestra y 24 consultas independientes |
| `../starter/queries.sql` | Ejemplos originales: catálogo, tendencias, co-vistas, feedback, fiabilidad, posiciones |
| `../starter/recommender.py` | Baseline real: pull → predict → push; requiere variables de entorno al importar |
| `../starter/offline_eval.ipynb` | Evaluación original: hit rate e IPS/SNIPS de primer slot |
| `../nexitos/estimator.py` | Borrador local, Python sin dependencias externas; no sirve tráfico por sí solo |
| `../nexitos/offline_eval.ipynb` | Compara Nexitos con popularidad sobre la misma muestra temporal |
| `../.env.example` | Variables esperadas; `.env` contiene secretos y no debe copiarse aquí |
| `../Makefile` | `make main` ejecuta el baseline, no Nexitos |
| `../../request/submissions.json` | Metadatos guardados de submissions; no es un dataset de ejemplos |

Nexitos entrena asociaciones por usuario/sesión con pesos vista=1, carrito=3, compra=5; conserva la interacción más fuerte por producto, limita pares a 100 productos por sesión y guarda 100 vecinos. Usa últimas vistas, similitud de categoría/subcategoría/marca y popularidad; penaliza ítems vistos. Ignora por ahora usuario persistente, contexto, texto, precios, feedback y timestamps de vistas. `fit` reconstruye el estado; no hay aprendizaje incremental ni persistencia de modelo. La memoria inicial de pares puede crecer mucho.

## Evidencia y límites

Revisión: 14-11-2026. Se revisaron código, contrato, notebooks y salidas guardadas. `requests.ipynb` contiene ejecución exitosa de conexión, esquema de requests, muestras de catálogo/eventos/requests, catálogo combinado, actualizaciones, popularidad y tendencias. Una captura del catálogo combinado contiene 5.100 productos; **no es un conteo actual garantizado**. Co-vistas devolvió vacío para un producto y periodo concretos: no demuestra ausencia global de asociaciones.

Todos los campos del diccionario proceden del contrato; solo la muestra descrita se ha observado. El esquema de requests observado coincide con los campos declarados y todos aparecen como Dynamic. No se ha auditado exhaustivamente la distribución de tipos, nulos, claves ni integridad referencial. La descarga del OpenAPI falló por DNS en este entorno; los detalles HTTP se basan en el starter y la guía local de RawTree, no en una verificación remota reciente.

**Privacidad del método:** cualquier `model_version` enviado puede aparecer públicamente. Usar etiquetas neutras como `nexitos-v1`; nunca incluir algoritmo, features, parámetros ni secretos. Mantener la etiqueta actual del usuario.
