"""Personalised recommender: candidates -> features -> ranker.

    store = Store.load(until="2026-11-07")    # offline: only data before the cut
    store = Store.load(live=True)             # live: all history + today's shop.events / catalog_updates
    model = Recommender(store, ranker=Ranker.load("models/ranker.txt"))
    model.recommend(request)                  # -> 10 product ids, best first

1. Candidates (~100-200 per request): what the shopper just viewed, what they interacted with before
   (user x item table), products co-viewed with their views, best sellers of the subcategories they
   are browsing / like, and the global best sellers.
2. Features per (request, candidate): session signals, user x item history, user x subcategory taste,
   co-views, popularity and conversion of the product, price, context.
3. Ranker: a LightGBM model trained on shop.history_sessions predicts P(purchase) for each candidate.
   Without a trained model it falls back to a hand-weighted score.

The same `features()` function builds the training rows and the live rows, so offline and live agree.
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rawtree import Record, query_df, query_parts, to_epoch

PAGES = {"home": 0, "product": 1, "category": 2}
DEVICES = {"web": 0, "ios": 1, "android": 2}
POP_DAYS = 7          # popularity window
DAY = 86400.0

FEATURES = [
    "pos",                # slot in the carousel (training only; 0 = first slot at inference)
    # session: what the shopper is doing right now
    "in_views", "view_recency", "view_count", "n_views",
    "same_subcat_views", "same_cat_views", "same_brand_views", "coview",
    # user x item: what this shopper did with this product before
    "ui_views", "ui_carts", "ui_buys", "ui_days_since",
    # user x subcategory taste
    "us_share", "us_count", "user_events",
    # product
    "pop_views", "pop_carts", "pop_buys", "cvr", "cart_rate", "log_price", "price_vs_subcat", "age_days",
    # context
    "page", "device", "is_new_user",
]


def _until(until: str | None, col: str = "ts") -> str:
    return f" AND {col} < '{until}'" if until else ""


class Store:
    """Everything the recommender knows, kept in memory for millisecond lookups."""

    def __init__(self) -> None:
        self.until: str | None = None
        self.unavailable: set[str] = set()          # out of stock / inactive right now
        self.live_ui: dict[str, dict[str, list[float]]] = defaultdict(dict)   # user -> item -> [v, c, b, last]
        self.live_us: dict[str, Counter] = defaultdict(Counter)                # user -> subcategory -> n
        self.bought_from_us: dict[str, set[str]] = defaultdict(set)            # carousel purchases (feedback)
        self.last_event_ts = ""

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, until: str | None = None, live: bool = False, verbose: bool = True) -> "Store":
        s = cls()
        s.until = until
        steps = [("catalog", lambda: s._load_catalog(live)),
                 ("user x item", lambda: s._load_user_items(until)),
                 ("user x subcategory", lambda: s._load_user_subcats(until)),
                 ("popularity", lambda: s._load_popularity(until, live)),
                 ("co-views", lambda: s._load_coviews(until))]
        if live:
            steps.append(("live events", s._load_live_events))
        for name, step in steps:
            t = time.time()
            step()
            if verbose:
                print(f"  loaded {name} in {time.time() - t:.1f}s")
        return s

    def _load_catalog(self, live: bool) -> None:
        if live:
            cat = query_df("""
                SELECT product_id, argMax(category, v) AS category, argMax(subcategory, v) AS subcategory,
                       argMax(brand, v) AS brand, argMax(price_eur, v) AS price_eur, argMax(is_active, v) AS is_active,
                       argMax(created_at, v) AS created_at, argMax(change, v) AS last_change
                FROM (
                    SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
                           brand::String AS brand, price_eur::Float64 AS price_eur, is_active::Bool AS is_active,
                           toString(created_at::DateTime64(3)) AS created_at, '' AS change, '' AS v
                    FROM shop.catalog
                    UNION ALL
                    SELECT product_id::String, category::String, subcategory::String, brand::String, price_eur::Float64,
                           is_active::Bool, toString(created_at::DateTime64(3)), change::String, toString(ts::DateTime64(3))
                    FROM shop.catalog_updates
                ) GROUP BY product_id""")
            stock = query_df("""
                SELECT product_id::String AS product_id FROM shop.catalog_updates
                WHERE change::String IN ('out_of_stock', 'back_in_stock') GROUP BY product_id
                HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'""")
            out = set(stock.product_id) if len(stock) else set()
        else:
            cat = query_df("""
                SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
                       brand::String AS brand, price_eur::Float64 AS price_eur, is_active::Bool AS is_active,
                       toString(created_at::DateTime64(3)) AS created_at
                FROM shop.catalog""")
            out = set()
        self.subcat = dict(zip(cat.product_id, cat.subcategory))
        self.category = dict(zip(cat.product_id, cat.category))
        self.brand = dict(zip(cat.product_id, cat.brand))
        self.price = dict(zip(cat.product_id, cat.price_eur.astype(float)))
        self.created = {p: to_epoch(c) for p, c in zip(cat.product_id, cat.created_at)}
        self.unavailable = set(cat.product_id[~cat.is_active.astype(bool)]) | out
        med = cat.groupby("subcategory").price_eur.median()
        self.subcat_price = med.to_dict()
        self.products_by_subcat: dict[str, list[str]] = cat.groupby("subcategory").product_id.apply(list).to_dict()

    def _load_user_items(self, until: str | None) -> None:
        """user x item table from organic history (views/carts/purchases). Single views are left out to save memory."""
        df = query_parts(f"""
            SELECT user_id::String AS u, product_id::String AS p,
                   toUInt32(countIf(event_type = 'view')) AS v, toUInt32(countIf(event_type = 'add_to_cart')) AS c,
                   toUInt32(countIf(event_type = 'purchase')) AS b, toUnixTimestamp(max(ts::DateTime64(3))) AS last
            FROM shop.history_events WHERE {{part}}{_until(until)}
            GROUP BY u, p HAVING b > 0 OR c > 0 OR v >= 2""", parts=4)
        df = df.sort_values("u", kind="stable").reset_index(drop=True)
        users, starts = np.unique(df.u.to_numpy(), return_index=True)
        ends = np.append(starts[1:], len(df))
        self.ui_index = {u: (s, e) for u, s, e in zip(users, starts, ends)}
        self.ui_p = df.p.to_numpy()
        self.ui_vals = df[["v", "c", "b", "last"]].to_numpy(dtype=np.float64)

    def _load_user_subcats(self, until: str | None) -> None:
        df = query_parts(f"""
            SELECT e.user_id::String AS u, c.subcategory AS sc, toUInt32(count()) AS n
            FROM shop.history_events AS e
            INNER JOIN (SELECT product_id::String AS p, subcategory::String AS subcategory FROM shop.catalog) AS c
                ON c.p = e.product_id::String
            WHERE {{part}}{_until(until, 'e.ts')}
            GROUP BY u, sc""", parts=4, key="e.user_id::String")
        df = df.sort_values("u", kind="stable").reset_index(drop=True)
        users, starts = np.unique(df.u.to_numpy(), return_index=True)
        ends = np.append(starts[1:], len(df))
        self.us_index = {u: (s, e) for u, s, e in zip(users, starts, ends)}
        self.us_sc = df.sc.to_numpy()
        self.us_n = df.n.to_numpy(dtype=np.float64)

    def _load_popularity(self, until: str | None, live: bool = False) -> None:
        """Views / carts / purchases per product over the last POP_DAYS days (history + live events)."""
        end = f"toDateTime64('{until}', 3)" if until else "now64(3)"
        source = "shop.history_events"
        if live:
            source = ("(SELECT product_id, event_type, ts FROM shop.history_events "
                      "UNION ALL SELECT product_id, event_type, ts FROM shop.events)")
        df = query_df(f"""
            SELECT product_id::String AS p, countIf(event_type = 'view') AS v,
                   countIf(event_type = 'add_to_cart') AS c, countIf(event_type = 'purchase') AS b
            FROM {source}
            WHERE ts::DateTime64(3) >= {end} - INTERVAL {POP_DAYS} DAY AND ts::DateTime64(3) < {end}
            GROUP BY p""")
        self.pop = {p: (float(v), float(c), float(b)) for p, v, c, b in zip(df.p, df.v, df.c, df.b)}
        ranked = df.sort_values(["b", "c", "v"], ascending=False)
        self.top_global = ranked.p.tolist()
        ranked = ranked.assign(sc=ranked.p.map(self.subcat))
        self.top_by_subcat = ranked.groupby("sc", sort=False).p.apply(lambda s: s.head(20).tolist()).to_dict()

    def _load_coviews(self, until: str | None) -> None:
        """For every product, the 20 products most viewed in the same sessions."""
        df = query_df(f"""
            SELECT a.product_id AS p, b.product_id AS q, count() AS n
            FROM (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events
                  WHERE event_type = 'view'{_until(until)}) AS a
            INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events
                        WHERE event_type = 'view'{_until(until)}) AS b ON a.session_id = b.session_id
            WHERE a.product_id != b.product_id
            GROUP BY p, q ORDER BY n DESC LIMIT 20 BY p""")
        self.coviews = df.groupby("p").q.apply(list).to_dict()

    def _load_live_events(self) -> None:
        """Today's organic activity (shop.events), aggregated like the history."""
        df = query_parts("""
            SELECT user_id::String AS u, product_id::String AS p, event_type::String AS e,
                   toUInt32(count()) AS n, toUnixTimestamp(max(ts::DateTime64(3))) AS last
            FROM shop.events WHERE {part} GROUP BY u, p, e""", parts=2)
        self.ingest(df.itertuples(index=False))
        last = query_df("SELECT toString(max(ts::DateTime64(3))) AS t FROM shop.events")
        self.last_event_ts = last.t[0] if len(last) else ""

    # ------------------------------------------------------------------ live updates
    def ingest(self, events: Iterable) -> None:
        """Add organic events (u, p, e, n, last) to the live user x item / user x subcategory tables."""
        col = {"view": 0, "add_to_cart": 1, "purchase": 2}
        for u, p, e, n, last in events:
            if e not in col:
                continue
            row = self.live_ui[u].setdefault(p, [0.0, 0.0, 0.0, 0.0])
            row[col[e]] += float(n)
            row[3] = max(row[3], float(last))
            sc = self.subcat.get(p)
            if sc:
                self.live_us[u][sc] += float(n)

    def apply_catalog_update(self, u: Record) -> None:
        """One shop.catalog_updates record: new product, price change, stock-out or back in stock."""
        p = u["product_id"]
        change = u.get("change")
        self.subcat[p] = u.get("subcategory") or self.subcat.get(p)
        self.category[p] = u.get("category") or self.category.get(p)
        self.brand[p] = u.get("brand") or self.brand.get(p)
        if u.get("price_eur") is not None:
            self.price[p] = float(u["price_eur"])
        if change == "new_product":
            self.created[p] = to_epoch(u["created_at"]) if u.get("created_at") else time.time()
            items = self.products_by_subcat.setdefault(self.subcat[p], [])
            if p not in items:
                items.append(p)
        if change == "out_of_stock" or u.get("is_active") is False:
            self.unavailable.add(p)
        elif change in ("back_in_stock", "new_product"):
            self.unavailable.discard(p)

    # ------------------------------------------------------------------ lookups
    def user_items(self, user: str) -> dict[str, np.ndarray]:
        """item -> [views, carts, buys, last_ts] for one user (history + live)."""
        s, e = self.ui_index.get(user, (0, 0))
        out = {p: self.ui_vals[i].copy() for i, p in zip(range(s, e), self.ui_p[s:e])}
        for p, row in self.live_ui.get(user, {}).items():
            if p in out:
                out[p][:3] += row[:3]
                out[p][3] = max(out[p][3], row[3])
            else:
                out[p] = np.array(row, dtype=np.float64)
        return out

    def user_subcats(self, user: str) -> Counter:
        s, e = self.us_index.get(user, (0, 0))
        c = Counter(dict(zip(self.us_sc[s:e], self.us_n[s:e])))
        c.update(self.live_us.get(user, {}))
        return c


class Ranker:
    """LightGBM model predicting P(purchase) from FEATURES."""

    def __init__(self, booster: Any = None) -> None:
        self.booster = booster

    @classmethod
    def load(cls, path: str | Path) -> "Ranker":
        import lightgbm as lgb
        return cls(lgb.Booster(model_file=str(path)))

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.booster is not None:
            return self.booster.predict(X)
        return heuristic_score(X)


def heuristic_score(X: np.ndarray) -> np.ndarray:
    """Hand-made score (log-odds style) used before a model is trained. Weights from the exploration:
    a just-viewed item sells ~15x more, an item the user bought before ~30x, same subcategory ~2x."""
    f = {name: X[:, i] for i, name in enumerate(FEATURES)}
    return (2.7 * f["in_views"] + 0.5 * f["view_recency"] + 0.4 * f["view_count"]
            + 1.2 * np.log1p(f["ui_buys"]) + 0.6 * np.log1p(f["ui_carts"]) + 0.3 * np.log1p(f["ui_views"])
            + 0.6 * np.minimum(f["same_subcat_views"], 2) + 0.15 * f["same_cat_views"] + 0.2 * f["same_brand_views"]
            + 1.5 * f["us_share"] + 0.3 * f["coview"]
            + 0.25 * np.log1p(f["pop_buys"]) + 8 * f["cvr"])


class Recommender:
    def __init__(self, store: Store, ranker: Ranker | None = None, n_candidates: int = 200) -> None:
        self.store = store
        self.ranker = ranker or Ranker()
        self.n_candidates = n_candidates

    # ------------------------------------------------------------------ public
    def recommend(self, req: Record, k: int = 10) -> list[str]:
        """10 distinct, available product ids for one request, best first. Never raises."""
        try:
            ctx = self.context(req)
            cands = self.candidates(ctx)
            if cands:
                scores = self.ranker.score(self.features(ctx, cands))
                order = np.argsort(-scores, kind="stable")
                ranked = [cands[i] for i in order]
            else:
                ranked = []
        except Exception as e:  # never let one bad request break the batch
            print(f"recommend failed for {req.get('request_id')}: {e!r}")
            ranked = []
        return self.fill(ranked, k, exclude=self.store.bought_from_us.get(req.get("user_id", ""), set()))

    def fill(self, ranked: list[str], k: int = 10, exclude: set[str] = frozenset()) -> list[str]:
        """Keep valid items, then pad with best sellers so the answer always has k distinct products."""
        s = self.store
        out: list[str] = []
        seen: set[str] = set()
        for p in list(ranked) + s.top_global:
            if p not in seen and p not in s.unavailable and p not in exclude and p in s.subcat:
                out.append(p)
                seen.add(p)
                if len(out) == k:
                    break
        return out

    # ------------------------------------------------------------------ steps
    def context(self, req: Record) -> dict[str, Any]:
        """Everything about the request that features need, computed once."""
        s = self.store
        views = [v["product_id"] for v in (req.get("recent_views") or []) if isinstance(v, dict)]
        user = req.get("user_id") or ""
        ts = req.get("ts")
        now = to_epoch(ts) if isinstance(ts, str) else time.time()
        recency: dict[str, int] = {}
        for k, p in enumerate(reversed(views)):        # 0 = most recent view
            recency.setdefault(p, k)
        cov: Counter = Counter()
        for k, p in enumerate(reversed(views[-5:])):
            for rank, q in enumerate(s.coviews.get(p, [])):
                cov[q] += 0.7 ** k / (rank + 1)
        us = s.user_subcats(user)
        is_new = req.get("is_new_user")
        if is_new is None:            # history rows don't carry the flag: new = no organic history at all
            is_new = user not in s.us_index
        return {
            "user": user, "now": now, "views": views, "recency": recency, "view_count": Counter(views),
            "v_subcats": Counter(s.subcat.get(p) for p in views), "v_cats": Counter(s.category.get(p) for p in views),
            "v_brands": Counter(s.brand.get(p) for p in views), "coview": cov,
            "ui": s.user_items(user), "us": us, "us_total": float(sum(us.values())),
            "page": PAGES.get(req.get("page") or "", -1), "device": DEVICES.get(req.get("device") or "", -1),
            "is_new_user": float(bool(is_new)),
        }

    def candidates(self, ctx: dict[str, Any]) -> list[str]:
        s = self.store
        out: dict[str, None] = {}

        def add(items: Iterable[str], limit: int) -> None:
            n = 0
            for p in items:
                if n >= limit:
                    break
                if p in s.subcat and p not in s.unavailable:
                    out.setdefault(p)
                    n += 1

        add(reversed(ctx["views"]), 20)                                                   # just viewed
        ui = ctx["ui"]
        add(sorted(ui, key=lambda p: -(5 * ui[p][2] + 2 * ui[p][1] + ui[p][0] + ui[p][3] / 1e9))[:60], 40)  # user history
        add((q for q, _ in ctx["coview"].most_common(40)), 40)                           # co-viewed
        for sc, _ in (ctx["v_subcats"] + Counter()).most_common(3):                       # browsing subcategories
            add(s.top_by_subcat.get(sc, []), 10)
            add(self._newest(sc), 3)                                                      # cold start: new products
        for sc, _ in ctx["us"].most_common(3):                                            # long-term taste
            add(s.top_by_subcat.get(sc, []), 6)
        add(s.top_global, 30)                                                             # best sellers
        return list(out)[: self.n_candidates]

    def _newest(self, subcat: str) -> list[str]:
        s = self.store
        items = s.products_by_subcat.get(subcat, [])
        return sorted(items, key=lambda p: -s.created.get(p, 0))[:3]

    def features(self, ctx: dict[str, Any], cands: list[str], positions: list[int] | None = None) -> np.ndarray:
        s = self.store
        X = np.zeros((len(cands), len(FEATURES)), dtype=np.float64)
        n_views = len(ctx["views"])
        for i, p in enumerate(cands):
            sc, cat, br = s.subcat.get(p), s.category.get(p), s.brand.get(p)
            ui = ctx["ui"].get(p)
            pv, pc, pb = s.pop.get(p, (0.0, 0.0, 0.0))
            price = s.price.get(p, 0.0)
            rec = ctx["recency"].get(p)
            X[i] = (
                positions[i] if positions else 0,
                1.0 if rec is not None else 0.0,
                1.0 / (1 + rec) if rec is not None else 0.0,
                ctx["view_count"].get(p, 0),
                n_views,
                ctx["v_subcats"].get(sc, 0),
                ctx["v_cats"].get(cat, 0),
                ctx["v_brands"].get(br, 0),
                ctx["coview"].get(p, 0.0),
                ui[0] if ui is not None else 0.0,
                ui[1] if ui is not None else 0.0,
                ui[2] if ui is not None else 0.0,
                min((ctx["now"] - ui[3]) / DAY, 90.0) if ui is not None and ui[3] > 0 else 90.0,
                ctx["us"].get(sc, 0.0) / ctx["us_total"] if ctx["us_total"] else 0.0,
                ctx["us"].get(sc, 0.0),
                ctx["us_total"],
                math.log1p(pv), math.log1p(pc), math.log1p(pb),
                (pb + 0.1) / (pv + 20.0),
                (pc + 0.1) / (pv + 20.0),
                math.log1p(price),
                price / s.subcat_price.get(sc, price or 1.0) if price else 1.0,
                min((ctx["now"] - s.created.get(p, 0.0)) / DAY, 999.0),
                ctx["page"], ctx["device"], ctx["is_new_user"],
            )
        return X


def request_from_row(row: Any) -> Record:
    """A shop.history_sessions / shop.requests row (dict or namedtuple) -> the dict recommend() expects."""
    get = row.get if isinstance(row, dict) else lambda k, d=None: getattr(row, k, d)
    return {"request_id": get("request_id"), "ts": get("ts"), "user_id": get("user_id"),
            "is_new_user": get("is_new_user", None), "recent_views": get("recent_views") or [],
            "page": get("page"), "device": get("device"), "region": get("region")}
