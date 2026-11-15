# Plan del domingo 15 (final 10:00–12:00)

Todo se ejecuta desde la raíz del repo, rama `lynch-v9`, en **la misma máquina** (el VPS). **Un solo proceso respondiendo**:
si hay otro (`lynch-f1`, `spielberg`, ...), se para antes.

## Esta noche: A/B de 3 versiones en un solo proceso

```
python lynch/live.py --submit --ab lynch-v8=0,lynch-v9a=3,lynch-v9c=3:0.05
```

| Versión | Qué es |
|---|---|
| lynch-v8 | el modelo tal cual |
| lynch-v9a | + como mucho 3 productos por subcategoría |
| lynch-v9c | + lo anterior y una pequeña inclinación por precio (0,05) |

Cada versión responde a un tercio de las peticiones, a la vez, con los mismos compradores. **Apuntar la hora de arranque**:
____:____

Tarda ~2 min en cargar. Cada minuto escribe `LIVE: answered N (lynch-v8 .. + lynch-v9a .. + lynch-v9c ..) (invalid fixed 0)`.

## Por la mañana

**1. Leer el A/B (~8:30).** Desde la hora de arranque + 5 min:
```
python lynch/ab_report.py HH:MM
```
Da conversión, cold, no-cold y €/petición de cada versión, y cada diferencia frente a lynch-v8 con su margen (± 2 SE).

**2. Decidir con esta regla (fijada antes de ver los datos):**
- **Tope (v9a frente a v8):** si v9a pierde conversión por más que su margen → sin tope. Si no → con tope (empate =
  con tope, porque da más €).
- **Inclinación (v9c frente a v9a):** se queda salvo que v9c pierda conversión frente a v9a por más que su margen. Los
  datos reales dicen que no debería costar (los productos caros se compran más de lo que el modelo predice).
- Lo normal es que gane **v9c**. Diferencias más pequeñas que el margen son ruido: no cambiar de opinión por ellas.

**3. Reentrenar con todos los datos (~8:45, ~10 min):**
```
python lynch/train_v8.py --final
```
Reescribe `lynch/models/ranker_v8.txt`; el proceso en marcha lo carga solo en 30 s (sale `reloaded model` en su log).
Si el entrenamiento falla o imprime algo raro: `git checkout lynch/models/ranker_v8.txt` y seguir con el modelo de ayer.

**4. Dejar solo la versión ganadora (~9:40, nunca después de las 9:55).** Parar el A/B (Ctrl+C) y arrancar UNA:
```
python lynch/live.py --submit --version lynch-v9c --max-per-subcat 3 --price-tilt 0.05   # v9c
python lynch/live.py --submit --version lynch-v9a --max-per-subcat 3                     # v9a
python lynch/live.py --submit --version lynch-v8                                         # v8
```
Esperar a la primera línea `LIVE: answered ... (invalid fixed 0)` (~3 min) y mirar en el leaderboard que la luz está verde.

**5. De 10:00 a 12:00: no tocar nada.** Ni reinicios ni cambios de versión: cada hueco son peticiones sin respuesta.
Todo lo nuevo (productos, precios, agotados, compradores) entra solo: catálogo cada 1 s, eventos cada 2 s.

## Si algo se rompe
- El proceso se cae → volver a lanzar el mismo comando del paso 4 (tarda ~2 min).
- El modelo nuevo da problemas → `git checkout lynch/models/ranker_v8.txt` (se recarga solo en 30 s).
