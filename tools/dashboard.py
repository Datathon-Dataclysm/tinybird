"""Live dashboard: see what flows between the shop, our recommender and the feedback.

    streamlit run tools/dashboard.py

Reads TEAM_ID and RAWTREE_API_KEY from .env (project root). Read-only: it never writes to RawTree.
Times are shown in Barcelona time (CET = UTC+1); the data itself is in UTC.
"""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
CET = timedelta(hours=1)

st.set_page_config(page_title=f"RawTree · {TEAM}", page_icon="🛒", layout="wide")


@st.cache_resource
def session() -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"
    return s


def q(sql: str) -> pd.DataFrame:
    r = session().post(f"{API}/v1/query", json={"sql": sql}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"RawTree {r.status_code}: {r.text[:300]}")
    return pd.DataFrame(r.json()["data"])


def to_cet(col: pd.Series) -> pd.Series:
    return pd.to_datetime(col) + CET


@st.cache_data(ttl=600, show_spinner=False)
def catalog() -> pd.DataFrame:
    """Current catalogue: shop.catalog with the latest catalog_updates record of each product on top."""
    df = q("""
        SELECT product_id,
               argMax(category, v) AS category, argMax(subcategory, v) AS subcategory,
               argMax(brand, v) AS brand, argMax(price_eur, v) AS price_eur
        FROM (
            SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
                   brand::String AS brand, price_eur::Float64 AS price_eur, '' AS v
            FROM shop.catalog
            UNION ALL
            SELECT product_id::String, category::String, subcategory::String, brand::String, price_eur::Float64, toString(ts)
            FROM shop.catalog_updates
        )
        GROUP BY product_id""")
    return df.set_index("product_id")


# ---------------------------------------------------------------- queries

def live_requests(minutes: int, limit: int) -> pd.DataFrame:
    """Each recent request joined with our answer (submissions) and what happened (feedback)."""
    return q(f"""
        SELECT r.request_id AS request_id, r.ts AS ts, r.user_id AS user_id, r.is_new_user AS is_new,
               r.device AS device, r.region AS region, r.page AS page, r.views AS recent_views,
               s.items AS items, s.model_version AS model_version,
               f.request_id != '' AS has_feedback, f.served_by AS served_by, f.fallback_reason AS reason,
               f.latency_ms AS latency_ms, f.shown AS shown,
               f.clicked AS clicked, f.purchased AS purchased, f.revenue_eur AS revenue
        FROM (SELECT request_id::String AS request_id, ts::DateTime64(3) AS ts, user_id::String AS user_id,
                     is_new_user::Bool AS is_new_user, context.device::String AS device,
                     context.region::String AS region, context.page::String AS page,
                     arrayMap(x -> x.product_id::String, recent_views::Array(JSON)) AS views
              FROM shop.requests WHERE ts > now() - INTERVAL {minutes} MINUTE
              ORDER BY ts DESC LIMIT {limit}) AS r
        LEFT JOIN (SELECT request_id::String AS request_id, items::Array(String) AS items,
                          model_version::String AS model_version
                   FROM {TEAM}.submissions
                   WHERE request_id IN (SELECT request_id::String FROM shop.requests WHERE ts > now() - INTERVAL {minutes} MINUTE)
                  ) AS s ON s.request_id = r.request_id
        LEFT JOIN (SELECT request_id::String AS request_id, served_by::String AS served_by,
                          fallback_reason::String AS fallback_reason, latency_ms::Float64 AS latency_ms,
                          shown::Array(String) AS shown, clicked::String AS clicked,
                          purchased::String AS purchased, revenue_eur::Float64 AS revenue_eur
                   FROM {TEAM}.feedback WHERE ts > now() - INTERVAL {minutes + 1} MINUTE
                  ) AS f ON f.request_id = r.request_id
        ORDER BY r.ts DESC""")


def kpis(minutes: int) -> pd.Series:
    df = q(f"""
        SELECT count() AS requests,
               countIf(served_by = 'team') AS on_time,
               countIf(clicked != '') AS clicks,
               countIf(purchased != '') AS purchases,
               sum(revenue_eur::Float64) AS revenue,
               quantileIf(0.5)(latency_ms::Float64, served_by = 'team') AS p50,
               quantileIf(0.95)(latency_ms::Float64, served_by = 'team') AS p95
        FROM {TEAM}.feedback
        WHERE request_id != '__init__' AND ts > now() - INTERVAL {minutes} MINUTE""")
    return df.iloc[0]


def per_minute(minutes: int) -> pd.DataFrame:
    df = q(f"""
        SELECT toStartOfMinute(ts::DateTime64(3)) AS minute,
               count() AS requests,
               100 * countIf(served_by = 'team') / count() AS on_time_pct,
               quantileIf(0.95)(latency_ms::Float64, served_by = 'team') AS p95_ms,
               countIf(fallback_reason = 'missing') AS missing,
               countIf(fallback_reason = 'late') AS late,
               countIf(fallback_reason = 'invalid') AS invalid
        FROM {TEAM}.feedback
        WHERE request_id != '__init__' AND ts > now() - INTERVAL {minutes} MINUTE
        GROUP BY minute ORDER BY minute""")
    if not df.empty:
        df["minute"] = to_cet(df["minute"])
        df = df.set_index("minute")
    return df


def by_version(hours: int) -> pd.DataFrame:
    return q(f"""
        SELECT if(s.model_version = '', '(sin respuesta)', s.model_version) AS model_version,
               count() AS requests,
               round(100 * countIf(f.served_by = 'team') / count(), 1) AS a_tiempo_pct,
               round(100 * countIf(f.clicked != '') / count(), 2) AS ctr_pct,
               round(100 * countIf(f.purchased != '') / count(), 2) AS compras_por_100,
               round(sum(f.revenue_eur) / count(), 3) AS eur_por_peticion,
               min(f.ts) AS desde, max(f.ts) AS hasta
        FROM (SELECT request_id::String AS request_id, ts::DateTime64(3) AS ts, served_by::String AS served_by,
                     clicked::String AS clicked, purchased::String AS purchased, revenue_eur::Float64 AS revenue_eur
              FROM {TEAM}.feedback WHERE request_id != '__init__' AND ts > now() - INTERVAL {hours} HOUR) AS f
        LEFT JOIN (SELECT request_id::String AS request_id, any(model_version::String) AS model_version
                   FROM {TEAM}.submissions GROUP BY request_id) AS s ON s.request_id = f.request_id
        GROUP BY model_version ORDER BY hasta DESC""")


def shop_top_products(minutes: int) -> pd.DataFrame:
    return q(f"""
        SELECT product_id::String AS product_id,
               countIf(event_type = 'view') AS views,
               countIf(event_type = 'add_to_cart') AS carts,
               countIf(event_type = 'purchase') AS purchases
        FROM shop.events WHERE ts > now() - INTERVAL {minutes} MINUTE
        GROUP BY product_id ORDER BY purchases DESC, views DESC LIMIT 15""")


def one_request(request_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rid = request_id.replace("'", "")
    req = q(f"SELECT * FROM shop.requests WHERE request_id = '{rid}'")
    sub = q(f"SELECT * FROM {TEAM}.submissions WHERE request_id = '{rid}'")
    fb = q(f"SELECT * FROM {TEAM}.feedback WHERE request_id = '{rid}'")
    return req, sub, fb


# ---------------------------------------------------------------- helpers

def status(row) -> str:
    if not row.has_feedback:
        return "⏳ esperando" if row["items"] else "⏳ sin respuesta aún"
    if row.served_by == "team":
        return "✅ a tiempo"
    return f"❌ {row.reason or 'fallback'}"


def position(product: str, shown) -> str:
    if not product:
        return ""
    try:
        return f"#{list(shown).index(product) + 1}"
    except ValueError:
        return "?"


def describe(products, cat: pd.DataFrame, clicked: str = "", purchased: str = "") -> pd.DataFrame:
    rows = []
    for i, p in enumerate(products, 1):
        info = cat.loc[p] if p in cat.index else None
        mark = "🛒 compra" if p == purchased else ("👆 clic" if p == clicked else "")
        rows.append({"#": i, "producto": p, "resultado": mark,
                     "categoría": None if info is None else info.category,
                     "subcategoría": None if info is None else info.subcategory,
                     "marca": None if info is None else info.brand,
                     "precio €": None if info is None else info.price_eur})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- page

st.title(f"🛒 Panel en directo · {TEAM}")
st.caption("Petición (shop.requests) → nuestra respuesta (submissions) → lo que hizo el comprador (feedback). "
           "Horas en horario de Barcelona.")

with st.sidebar:
    st.header("Ajustes")
    auto = st.toggle("Refresco automático", value=True)
    every = st.slider("Cada cuántos segundos", 3, 60, 5)
    kpi_minutes = st.select_slider("Ventana de los indicadores (min)", [5, 15, 30, 60, 180], value=5)
    n_rows = st.slider("Peticiones en la tabla", 10, 100, 30)
    version_hours = st.select_slider("Comparar versiones: últimas (h)", [1, 3, 6, 12, 24], value=3)
    st.divider()
    st.markdown("[Leaderboard ↗](https://datathon.rawtree.dev/leaderboard)")


@st.fragment(run_every=every if auto else None)
def live() -> None:
    cat = catalog()

    # --- headline numbers -------------------------------------------------
    k = kpis(kpi_minutes)
    n = int(k.requests or 0)
    on_time = 100 * k.on_time / n if n else 0.0
    light = "🟢 VERDE" if on_time > 90 else "🔴 ROJA"
    st.subheader(f"Últimos {kpi_minutes} min · luz {light}")
    c = st.columns(6)
    c[0].metric("Peticiones con feedback", f"{n:,}")
    c[1].metric("A tiempo y válidas", f"{on_time:.1f} %", help="Luz verde si > 90 % en los últimos 5 min")
    c[2].metric("Latencia p50 / p95", f"{(k.p50 or 0) / 1000:.2f} / {(k.p95 or 0) / 1000:.2f} s",
                help="Desde el ts de la petición hasta que la respuesta está en RawTree. Límite: 3 s")
    c[3].metric("Clics por 100", f"{100 * k.clicks / n:.2f}" if n else "—")
    c[4].metric("Compras por 100", f"{100 * k.purchases / n:.2f}" if n else "—",
                help="Lo que puntúa (en esperanza). Con pocas peticiones es muy ruidoso")
    c[5].metric("€ por petición", f"{k.revenue / n:.3f}" if n else "—")

    # --- live flow table ---------------------------------------------------
    st.subheader("Peticiones en directo")
    df = live_requests(minutes=5, limit=n_rows)
    if df.empty:
        st.info("No hay peticiones en los últimos 5 minutos.")
    else:
        df["hora"] = to_cet(df["ts"]).dt.strftime("%H:%M:%S.%f").str[:-3]
        df["estado"] = df.apply(status, axis=1)
        df["vistas"] = df.recent_views.apply(len)
        df["clic"] = df.apply(lambda r: position(r.clicked, r.shown if len(r.shown) else r["items"]), axis=1)
        df["compra"] = df.apply(lambda r: position(r.purchased, r.shown if len(r.shown) else r["items"]), axis=1)
        df["latencia s"] = (df.latency_ms / 1000).where(df.has_feedback & (df.served_by == "team"))
        df["€"] = df.revenue.where(df.revenue > 0)
        df["nuevo"] = df.is_new
        st.dataframe(
            df[["hora", "request_id", "user_id", "nuevo", "device", "region", "page", "vistas",
                "model_version", "estado", "latencia s", "clic", "compra", "€"]],
            hide_index=True, width="stretch",
            column_config={"latencia s": st.column_config.NumberColumn(format="%.2f"),
                           "€": st.column_config.NumberColumn(format="%.2f"),
                           "vistas": st.column_config.NumberColumn(help="Productos en recent_views")},
        )
        st.caption("clic / compra = posición en el carrusel del producto clicado o comprado. "
                   "El feedback llega unos segundos después de la petición.")

    # --- reliability over time --------------------------------------------
    st.subheader("Fiabilidad, última hora (por minuto)")
    pm = per_minute(60)
    if not pm.empty:
        a, b = st.columns(2)
        with a:
            st.markdown("**% a tiempo y válidas**")
            st.line_chart(pm["on_time_pct"], height=220, y_label="%", x_label="")
        with b:
            st.markdown("**Latencia p95 (ms)** · límite 3000")
            st.line_chart(pm["p95_ms"], height=220, y_label="ms", x_label="")
        fails = pm[["missing", "late", "invalid"]].sum()
        st.caption(f"Fallbacks en la última hora: missing {int(fails.missing)} · late {int(fails.late)} · "
                   f"invalid {int(fails.invalid)}")

    # --- versions & shop ----------------------------------------------------
    a, b = st.columns([3, 2])
    with a:
        st.subheader(f"Versiones del modelo (últimas {version_hours} h)")
        v = by_version(version_hours)
        if not v.empty:
            v["desde"] = to_cet(v["desde"]).dt.strftime("%H:%M")
            v["hasta"] = to_cet(v["hasta"]).dt.strftime("%H:%M")
        st.dataframe(v, hide_index=True, width="stretch")
        st.caption("compras_por_100 es la métrica que puntúa. Compara versiones solo con cientos de peticiones o más.")
    with b:
        st.subheader("Tienda: lo más comprado (60 min)")
        top = shop_top_products(60)
        if not top.empty:
            top = top.join(cat[["category", "subcategory", "price_eur"]], on="product_id")
        st.dataframe(top, hide_index=True, width="stretch")
        st.caption("shop.events: actividad orgánica de toda la tienda, no solo nuestro carrusel.")


try:
    live()
except Exception as e:  # keep the page alive if RawTree hiccups
    st.error(f"Error consultando RawTree: {e}")

# --- inspect one request (outside the auto-refresh so it doesn't jump) ---------
st.divider()
st.subheader("🔍 Inspeccionar una petición")
c1, c2 = st.columns([4, 1])
if c2.button("Última con compra", width="stretch"):
    last = q(f"SELECT request_id FROM {TEAM}.feedback WHERE purchased != '' ORDER BY ts DESC LIMIT 1")
    st.session_state["inspect"] = "" if last.empty else last.request_id[0]
rid = c1.text_input("request_id (cópialo de la tabla)", key="inspect", label_visibility="collapsed",
                    placeholder="r_... — copia un request_id de la tabla o pulsa «Última con compra»")
if rid:
    try:
        cat = catalog()
        req, sub, fb = one_request(rid.strip())
        if req.empty:
            st.warning("No encuentro esa petición en shop.requests.")
        else:
            r = req.iloc[0]
            views = [v["product_id"] for v in (r.get("recent_views") or [])]
            st.markdown(
                f"**{r.request_id}** · {to_cet(pd.Series([r.ts])).iloc[0]:%H:%M:%S} · usuario `{r.user_id}`"
                f"{' (nuevo)' if r.is_new_user else ''} · {r.get('context.device')} · "
                f"{r.get('context.region')} · página *{r.get('context.page')}*")
            f = fb.iloc[0] if not fb.empty else None
            clicked = f.clicked if f is not None else ""
            purchased = f.purchased if f is not None else ""
            a, b = st.columns(2)
            with a:
                st.markdown("**① Lo que había mirado** (`recent_views`, el último abajo)")
                st.dataframe(describe(views, cat), hide_index=True, width="stretch")
            with b:
                shown = list(f.shown) if f is not None else (list(sub.iloc[0]["items"]) if not sub.empty else [])
                who = "" if f is None else (" (lo que vio: nuestra respuesta)" if f.served_by == "team"
                                            else f" (lo que vio: FALLBACK por {f.fallback_reason})")
                st.markdown(f"**② Lo que le mostramos**{who}")
                st.dataframe(describe(shown, cat, clicked, purchased), hide_index=True, width="stretch")
            if f is not None:
                st.markdown(f"**③ Resultado:** latencia {f.latency_ms} ms · "
                            f"clic `{clicked or '—'}` · compra `{purchased or '—'}` · {f.revenue_eur} €")
            else:
                st.info("Todavía no hay feedback para esta petición (llega unos segundos después).")
            with st.expander("JSON en bruto de las tres tablas"):
                st.json({"shop.requests": req.to_dict("records"),
                         f"{TEAM}.submissions": sub.to_dict("records"),
                         f"{TEAM}.feedback": fb.to_dict("records")})
    except Exception as e:
        st.error(f"Error: {e}")
