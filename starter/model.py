"""Personalised recommender (v2): candidates -> features -> ranker -> live adjustments.

    store = Store.load(until="2026-11-07")    # offline: only data before the cut
    store = Store.load(live=True)             # live: all history + today's shop.events / catalog_updates
    model = Recommender(store, ranker=Ranker.load("models/ranker.txt"))
    model.recommend(request)                  # -> 10 product ids, best first

1. Candidates (~100-200 per request): what the shopper just viewed, what they bought / carted before (organic
   and in the carousel), products co-viewed with their views, best sellers and new products of the subcategories
   they browse or like, and the global best sellers.
2. Features per (request, candidate), see FEATURES. They are *scale-free* (shares and ratios, not raw counts)
   because the live stream is ~30x denser than the history: a count learnt on history would mean something else live.
3. Ranker: LightGBM trained on shop.history_sessions predicts P(purchase) of each candidate.
4. Live adjustments, for effects the history cannot teach (no price changes or trends in it):
   - price drops: P x (base price / current price)^2, clipped to [0.5, 2.5]  (live: a ~19% drop -> ~+60% sales)
   - trending subcategories: P x (purchase share last 2 h / usual share), shrunk to 1 with little data, clipped [0.67, 2]

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
REGIONS = {r: i for i, r in enumerate([
    "Andalucía", "Aragón", "Canarias", "Castilla y León", "Catalunya",
    "Comunitat Valenciana", "Galicia", "Madrid", "Murcia", "País Vasco"])}
POP_DAYS = 7          # popularity window
TREND_HOURS = 2       # trend window (live)
DAY = 86400.0
NEVER = 999.0         # "days since" when it never happened
NEW_PRODUCT_DAYS = 2  # a product created less than this ago counts as new
# Every product description carries exactly 2 of these attributes. Live, a shopper buys ~20x more when the product's
# attributes are the ones they usually interact with -- even outside their favourite categories.
ATTRIBUTES = [
    "a sporty style", "a family-size format", "organic ingredients", "a luxury feel", "a minimalist design",
    "built for professionals", "premium materials", "designed for kids", "bold colours", "wireless connectivity",
    "whisper-quiet operation", "a vintage look", "smart app control", "made from recycled materials",
    "great value for money", "a travel-ready build", "a waterproof shell", "extra durability", "a handmade finish",
    "compact and lightweight"]
ATTR_INDEX = {a: i for i, a in enumerate(ATTRIBUTES)}
ATTR_SQL = "[" + ",".join(f"'{a}'" for a in ATTRIBUTES) + "]"


def parse_attrs(description: str) -> tuple[int, ...]:
    """The attribute indices mentioned in a product description."""
    d = description or ""
    return tuple(i for i, a in enumerate(ATTRIBUTES) if a in d)
NEW_PRODUCT_BOOST = 3.0

FEATURES = [
    "pos",                # slot in the carousel (training only; 0 = first slot at inference)
    # session: what the shopper is doing right now
    "in_views", "view_count", "n_views", "same_subcat_views", "same_cat_views", "same_brand_views", "coview",
    # user x item, organic (shop.history_events + shop.events)
    "ui_buys", "ui_carts", "ui_views", "days_since_buy", "days_since_cart",
    # user x item, carousel (shop.history_sessions + our feedback)
    "cs_clicks", "cs_buys",
    # taste: shares of the user's activity / purchases (scale-free)
    "us_share", "uc_share", "us_buy_share", "ub_buy_share", "has_history",
    # product (scale-free: 1.0 = an average product)
    "pop_views", "pop_carts", "pop_buys", "cvr", "cart_rate", "log_price", "price_vs_subcat", "age_days",
    # live dynamics (learnt from the live replay; constant in the history)
    "is_new_product", "discount", "trend", "times_shown",
    # attribute taste (hidden driver of the simulator): user's affinity with the product's 2 attributes
    "attr_lift_mean", "attr_lift_min", "attr_lift_max", "attr_views_match",
    # context
    "page", "device", "region", "is_new_user",
]
CATEGORICAL = ["page", "device", "region"]


def _until(until: str | None, col: str = "ts") -> str:
    return f" AND {col} < '{until}'" if until else ""


def _index(users: np.ndarray) -> dict[str, tuple[int, int]]:
    """Sorted user column -> {user: (start, end)} row slice."""
    keys, starts = np.unique(users, return_index=True)
    ends = np.append(starts[1:], len(users))
    return {u: (s, e) for u, s, e in zip(keys, starts, ends)}


class Store:
    """Everything the recommender knows, kept in memory for millisecond lookups."""

    def __init__(self) -> None:
        self.until: str | None = None
        self.unavailable: set[str] = set()                                     # out of stock / inactive right now
        self.live_ui: dict[str, dict[str, list[float]]] = defaultdict(dict)    # user -> item -> [v, c, b, last_cart, last_buy]
        self.live_us: dict[str, Counter] = defaultdict(Counter)                 # user -> subcategory -> events
        self.live_us_buy: dict[str, Counter] = defaultdict(Counter)             # user -> subcategory -> purchases
        self.live_ub: dict[str, Counter] = defaultdict(Counter)                 # user -> brand -> purchases
        self.live_cs: dict[str, dict[str, list[float]]] = {}                    # user -> item -> [clicks, buys] (our feedback)
        self.trend_sc: dict[str, float] = {}                                    # subcategory -> trend (live, 1 = usual)
        self.shown: dict[str, Counter] = defaultdict(Counter)                   # user -> item -> times we showed it
        self.live_ua: dict[str, np.ndarray] = {}                                # user -> events per attribute (live)
        self.last_event_ts = ""

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, until: str | None = None, live: bool = False, verbose: bool = True) -> "Store":
        s = cls()
        s.until = until
        steps = [("catalog", lambda: s._load_catalog(live)),
                 ("user x item", lambda: s._load_user_items(until)),
                 ("user x subcategory", lambda: s._load_user_subcats(until)),
                 ("user x brand", lambda: s._load_user_brands(until)),
                 ("user x attribute", lambda: s._load_user_attrs(until)),
                 ("carousel history", lambda: s._load_carousel(until)),
                 ("popularity", lambda: s.load_popularity(until, live)),
                 ("co-views", lambda: s._load_coviews(until))]
        if live:
            steps += [("live events", s._load_live_events), ("trends", s.load_trends)]
        for name, step in steps:
            t = time.time()
            step()
            if verbose:
                print(f"  loaded {name} in {time.time() - t:.1f}s", flush=True)
        return s

    def _load_catalog(self, live: bool) -> None:
        base = query_df("""
            SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
                   brand::String AS brand, price_eur::Float64 AS price_eur, is_active::Bool AS is_active,
                   toString(created_at::DateTime64(3)) AS created_at, description::String AS description
            FROM shop.catalog""")
        self.base_price = dict(zip(base.product_id, base.price_eur.astype(float)))   # price before any live change
        cat, out = base, set()
        if live:
            cat = query_df("""
                SELECT product_id, argMax(category, v) AS category, argMax(subcategory, v) AS subcategory,
                       argMax(brand, v) AS brand, argMax(price_eur, v) AS price_eur, argMax(is_active, v) AS is_active,
                       argMax(created_at, v) AS created_at, argMax(description, v) AS description
                FROM (
                    SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
                           brand::String AS brand, price_eur::Float64 AS price_eur, is_active::Bool AS is_active,
                           toString(created_at::DateTime64(3)) AS created_at, description::String AS description, '' AS v
                    FROM shop.catalog
                    UNION ALL
                    SELECT product_id::String, category::String, subcategory::String, brand::String, price_eur::Float64,
                           is_active::Bool, toString(created_at::DateTime64(3)), description::String, toString(ts::DateTime64(3))
                    FROM shop.catalog_updates
                ) GROUP BY product_id""")
            stock = query_df("""
                SELECT product_id::String AS product_id FROM shop.catalog_updates
                WHERE change::String IN ('out_of_stock', 'back_in_stock') GROUP BY product_id
                HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'""")
            out = set(stock.product_id) if len(stock) else set()
        self.subcat = dict(zip(cat.product_id, cat.subcategory))
        self.category = dict(zip(cat.product_id, cat.category))
        self.brand = dict(zip(cat.product_id, cat.brand))
        self.price = dict(zip(cat.product_id, cat.price_eur.astype(float)))
        self.created = {p: to_epoch(c) for p, c in zip(cat.product_id, cat.created_at)}
        self.attrs = {p: parse_attrs(d) for p, d in zip(cat.product_id, cat.description)}
        self.by_attr_pair: dict[tuple, list[str]] = defaultdict(list)          # (a, b) -> products with both
        for p, at in self.attrs.items():
            if len(at) == 2:
                self.by_attr_pair[at].append(p)
        for p, pr in self.price.items():
            self.base_price.setdefault(p, pr)                                   # new products: first price is the base
        self.unavailable = set(cat.product_id[~cat.is_active.astype(bool)]) | out
        self.cat_of_subcat = dict(zip(cat.subcategory, cat.category))
        self.subcat_price = base.groupby("subcategory").price_eur.median().to_dict()
        self.products_by_subcat: dict[str, list[str]] = cat.groupby("subcategory").product_id.apply(list).to_dict()

    def _load_user_items(self, until: str | None) -> None:
        """user x item from organic history. Only carts / purchases: views alone carry no signal (see analysis)."""
        df = query_parts(f"""
            SELECT user_id::String AS u, product_id::String AS p,
                   toUInt32(countIf(event_type = 'view')) AS v, toUInt32(countIf(event_type = 'add_to_cart')) AS c,
                   toUInt32(countIf(event_type = 'purchase')) AS b,
                   toUnixTimestamp(maxIf(ts::DateTime64(3), event_type = 'add_to_cart')) AS lc,
                   toUnixTimestamp(maxIf(ts::DateTime64(3), event_type = 'purchase')) AS lb
            FROM shop.history_events WHERE {{part}}{_until(until)}
            GROUP BY u, p HAVING b > 0 OR c > 0""", parts=4)
        df = df.sort_values("u", kind="stable").reset_index(drop=True)
        self.ui_index = _index(df.u.to_numpy())
        self.ui_p = df.p.to_numpy()
        self.ui_vals = df[["v", "c", "b", "lc", "lb"]].to_numpy(dtype=np.float64)

    def _load_user_subcats(self, until: str | None) -> None:
        df = query_parts(f"""
            SELECT e.user_id::String AS u, c.subcategory AS sc, toUInt32(count()) AS n,
                   toUInt32(countIf(e.event_type::String = 'purchase')) AS nb
            FROM shop.history_events AS e
            INNER JOIN (SELECT product_id::String AS p, subcategory::String AS subcategory FROM shop.catalog) AS c
                ON c.p = e.product_id::String
            WHERE {{part}}{_until(until, 'e.ts')}
            GROUP BY u, sc""", parts=4, key="e.user_id::String")
        df = df.sort_values("u", kind="stable").reset_index(drop=True)
        self.us_index = _index(df.u.to_numpy())
        self.us_sc = df.sc.to_numpy()
        self.us_n = df.n.to_numpy(dtype=np.float64)
        self.us_nb = df.nb.to_numpy(dtype=np.float64)

    def _load_user_brands(self, until: str | None) -> None:
        df = query_df(f"""
            SELECT e.user_id::String AS u, c.brand AS br, toUInt32(count()) AS nb
            FROM shop.history_events AS e
            INNER JOIN (SELECT product_id::String AS p, brand::String AS brand FROM shop.catalog) AS c
                ON c.p = e.product_id::String
            WHERE e.event_type::String = 'purchase'{_until(until, 'e.ts')}
            GROUP BY u, br""")
        df = df.sort_values("u", kind="stable").reset_index(drop=True)
        self.ub_index = _index(df.u.to_numpy())
        self.ub_br = df.br.to_numpy()
        self.ub_nb = df.nb.to_numpy(dtype=np.float64)

    def _load_carousel(self, until: str | None) -> None:
        """Products each user clicked / bought in the old carousel (shop.history_sessions)."""
        df = query_df(f"""
            SELECT user_id::String AS u, clicked::String AS p, toUInt32(count()) AS c,
                   toUInt32(countIf(purchased::String = clicked::String)) AS b
            FROM shop.history_sessions WHERE clicked::String != ''{_until(until)}
            GROUP BY u, p""")
        self.cs: dict[str, dict[str, list[float]]] = defaultdict(dict)
        for u, p, c, b in zip(df.u, df.p, df.c, df.b):
            self.cs[u][p] = [float(c), float(b)]

    def _load_user_attrs(self, until: str | None) -> None:
        """Events of each user per product attribute (organic history), and the overall share of each attribute."""
        df = query_parts(f"""
            SELECT e.user_id::String AS u, a AS attr, toUInt32(count()) AS n
            FROM shop.history_events AS e
            INNER JOIN (SELECT product_id::String AS p,
                               arrayFilter(x -> position(description::String, x) > 0, {ATTR_SQL}) AS attrs
                        FROM shop.catalog) AS i ON i.p = e.product_id::String
            ARRAY JOIN i.attrs AS a
            WHERE {{part}}{_until(until, 'e.ts')}
            GROUP BY u, attr""", parts=4, key="e.user_id::String")
        df = df.sort_values("u", kind="stable").reset_index(drop=True)
        self.ua_index = _index(df.u.to_numpy())
        self.ua_attr = df.attr.map(ATTR_INDEX).to_numpy(dtype=np.int64)
        self.ua_n = df.n.to_numpy(dtype=np.float64)
        base = np.bincount(self.ua_attr, weights=self.ua_n, minlength=len(ATTRIBUTES))
        self.attr_base = base / max(base.sum(), 1.0)

    def load_popularity(self, until: str | None = None, live: bool = False) -> None:
        """Views / carts / purchases per product over the last POP_DAYS days, as shares (1.0 = average product)."""
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
        self.pop_raw = {p: [float(v), float(c), float(b)] for p, v, c, b in zip(df.p, df.v, df.c, df.b)}
        self.set_popularity(df)

    def set_popularity(self, df) -> None:
        """Counts per product (columns p, v, c, b) -> scale-free popularity features and best-seller lists."""
        n = max(len(self.subcat), 1)
        tv, tc, tb = (max(float(df[k].sum()), 1.0) / n for k in ("v", "c", "b"))
        pop = {p: (v / tv, c / tc, b / tb, (b + 0.1) / (v + 20.0), (c + 0.1) / (v + 20.0))
               for p, v, c, b in zip(df.p, df.v.astype(float), df.c.astype(float), df.b.astype(float))}
        ranked = df.sort_values(["b", "c", "v"], ascending=False)
        top_global = ranked.p.tolist()
        ranked = ranked.assign(sc=ranked.p.map(self.subcat))
        top_by_subcat = ranked.groupby("sc", sort=False).p.apply(lambda s: s.head(20).tolist()).to_dict()
        ranked = ranked.assign(cc=ranked.p.map(self.category))
        top_by_cat = ranked.groupby("cc", sort=False).p.apply(lambda s: s.head(40).tolist()).to_dict()
        self.pop, self.top_global, self.top_by_subcat, self.top_by_cat = pop, top_global, top_by_subcat, top_by_cat

    def load_trends(self) -> None:
        """Live: purchase share of each subcategory in the last TREND_HOURS vs its share in the last week of history."""
        df = query_df(f"""
            SELECT c.subcategory AS sc,
                   countIf(e.src = 'live') AS a, countIf(e.src = 'ref') AS r
            FROM (SELECT product_id::String AS p, 'live' AS src FROM shop.events
                  WHERE event_type::String = 'purchase' AND ts::DateTime64(3) > now64(3) - INTERVAL {TREND_HOURS} HOUR
                  UNION ALL
                  SELECT product_id::String AS p, 'ref' AS src FROM shop.history_events
                  WHERE event_type::String = 'purchase'
                    AND ts::DateTime64(3) > (SELECT max(ts::DateTime64(3)) FROM shop.history_events) - INTERVAL 7 DAY) AS e
            INNER JOIN (SELECT product_id::String AS p, subcategory::String AS subcategory FROM shop.catalog) AS c ON c.p = e.p
            GROUP BY sc""")
        self.set_trends(dict(zip(df.sc, df.a.astype(float))), dict(zip(df.sc, df.r.astype(float))))

    def set_trends(self, live_by_sc: dict, ref_by_sc: dict) -> None:
        """Trend of each subcategory = purchases in the last TREND_HOURS / purchases expected from its usual share
        (shrunk towards 1 when there is little data)."""
        A, R = float(sum(live_by_sc.values())), max(float(sum(ref_by_sc.values())), 1.0)
        trend = {}
        for sc in set(ref_by_sc) | set(live_by_sc):
            expected = A * ref_by_sc.get(sc, 0.0) / R
            trend[sc] = float(np.clip((live_by_sc.get(sc, 0.0) + 20.0) / (expected + 20.0), 0.5, 3.0))
        self.trend_sc = trend

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
            FROM shop.events WHERE {part} GROUP BY u, p, e""", parts=4)
        self.ingest(df.itertuples(index=False))
        last = query_df("SELECT toString(max(ts::DateTime64(3))) AS t FROM shop.events")
        self.last_event_ts = last.t[0] if len(last) else ""

    # ------------------------------------------------------------------ live updates
    def ingest(self, events: Iterable) -> None:
        """Add organic events (user, product, event_type, count, last_ts) to the live tables."""
        attrs = getattr(self, "attrs", {})
        for u, p, e, n, last in events:
            n, last = float(n), float(last)
            sc = self.subcat.get(p)
            if sc:
                self.live_us[u][sc] += n
            at = attrs.get(p)
            if at:
                vec = self.live_ua.get(u)
                if vec is None:
                    vec = self.live_ua[u] = np.zeros(len(ATTRIBUTES))
                for a in at:
                    vec[a] += n
            if e == "view":
                if p in self.live_ui.get(u, {}):
                    self.live_ui[u][p][0] += n
                continue
            row = self.live_ui[u].setdefault(p, [0.0, 0.0, 0.0, 0.0, 0.0])
            if e == "add_to_cart":
                row[1] += n
                row[3] = max(row[3], last)
            elif e == "purchase":
                row[2] += n
                row[4] = max(row[4], last)
                if sc:
                    self.live_us_buy[u][sc] += n
                br = self.brand.get(p)
                if br:
                    self.live_ub[u][br] += n

    def apply_catalog_update(self, u: Record) -> None:
        """One shop.catalog_updates record: new product, price change, stock-out or back in stock."""
        p = u["product_id"]
        change = u.get("change")
        self.subcat[p] = u.get("subcategory") or self.subcat.get(p)
        self.category[p] = u.get("category") or self.category.get(p)
        self.brand[p] = u.get("brand") or self.brand.get(p)
        if u.get("price_eur") is not None:
            self.price[p] = float(u["price_eur"])
            self.base_price.setdefault(p, float(u["price_eur"]))
        if u.get("description") and hasattr(self, "attrs"):
            at = parse_attrs(u["description"])
            if p not in self.attrs and len(at) == 2:
                self.by_attr_pair[at].append(p)
            self.attrs[p] = at
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
        """item -> [views, carts, buys, last_cart_ts, last_buy_ts] for one user (history + live)."""
        s, e = self.ui_index.get(user, (0, 0))
        out = {p: self.ui_vals[i].copy() for i, p in zip(range(s, e), self.ui_p[s:e])}
        for p, row in self.live_ui.get(user, {}).items():
            if p in out:
                out[p][:3] += row[:3]
                out[p][3:] = np.maximum(out[p][3:], row[3:])
            else:
                out[p] = np.array(row, dtype=np.float64)
        return out

    def user_subcats(self, user: str) -> tuple[Counter, Counter]:
        """(events per subcategory, purchases per subcategory) for one user."""
        s, e = self.us_index.get(user, (0, 0))
        n = Counter(dict(zip(self.us_sc[s:e], self.us_n[s:e])))
        nb = Counter({k: v for k, v in zip(self.us_sc[s:e], self.us_nb[s:e]) if v})
        n.update(self.live_us.get(user, {}))
        nb.update(self.live_us_buy.get(user, {}))
        return n, nb

    def user_attrs(self, user: str) -> np.ndarray:
        """Events of one user per attribute (history + live)."""
        out = np.zeros(len(ATTRIBUTES))
        s, e = self.ua_index.get(user, (0, 0))
        if e > s:
            np.add.at(out, self.ua_attr[s:e], self.ua_n[s:e])
        live = self.live_ua.get(user)
        if live is not None:
            out += live
        return out

    def user_brands(self, user: str) -> Counter:
        s, e = self.ub_index.get(user, (0, 0))
        c = Counter(dict(zip(self.ub_br[s:e], self.ub_nb[s:e])))
        c.update(self.live_ub.get(user, {}))
        return c

    def carousel(self, user: str) -> dict[str, list[float]]:
        """item -> [clicks, buys] in the carousel (history + our feedback)."""
        hist, live = self.cs.get(user, {}), self.live_cs.get(user, {})
        if not live:
            return hist
        out = {p: list(v) for p, v in hist.items()}
        for p, (c, b) in live.items():
            row = out.setdefault(p, [0.0, 0.0])
            row[0] += c
            row[1] += b
        return out


class Ranker:
    """LightGBM ranker. One model -> its score. Two models (two-stage) -> P(click) x P(purchase | click):
    the shopper clicks out of curiosity but buys out of affinity, so the two are learnt separately."""

    def __init__(self, booster: Any = None, buy_booster: Any = None) -> None:
        self.booster = booster
        self.buy_booster = buy_booster

    @staticmethod
    def _booster(path: Path):
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(path))
        if booster.feature_name() != FEATURES:   # an old model with other features would silently mis-score
            raise ValueError(f"{path} was trained with other features; retrain it")
        return booster

    @classmethod
    def load(cls, path: str | Path) -> "Ranker":
        """`ranker_x.txt` (+ `ranker_x_buy.txt` if present: two-stage)."""
        path = Path(path)
        buy = path.with_name(path.stem + "_buy" + path.suffix)
        return cls(cls._booster(path), cls._booster(buy) if buy.exists() else None)

    def score(self, X: np.ndarray) -> np.ndarray:
        """A positive score (probability-like)."""
        if self.booster is None:
            return np.exp(heuristic_score(X))
        s = self.booster.predict(X)
        if self.buy_booster is not None:
            s = s * self.buy_booster.predict(X)
        return s


def heuristic_score(X: np.ndarray) -> np.ndarray:
    """Hand-made log-score used before a model is trained, from the multipliers measured in the data analysis."""
    f = {name: X[:, i] for i, name in enumerate(FEATURES)}
    recent_buy = np.where(f["days_since_buy"] < 1, 1.0, np.where(f["days_since_buy"] < 7, 0.8, 0.5))
    return (3.0 * f["in_views"] + 3.0 * np.minimum(f["ui_buys"], 1) * recent_buy + 1.5 * np.minimum(f["ui_carts"], 1)
            + 2.5 * np.minimum(f["cs_buys"], 1) + 1.0 * np.minimum(f["cs_clicks"], 1)
            + 4.0 * f["uc_share"] + 2.0 * f["us_share"] + 1.0 * f["ub_buy_share"] + 0.6 * np.minimum(f["same_subcat_views"], 2)
            + 0.2 * f["coview"] + 0.3 * np.log1p(f["pop_buys"]) + 8 * f["cvr"])


class Recommender:
    def __init__(self, store: Store, ranker: Ranker | None = None, n_candidates: int = 260,
                 adjust: bool = False, wide: bool = False) -> None:
        self.store = store
        self.ranker = ranker or Ranker()
        self.n_candidates = n_candidates
        self.wide = wide                         # also best sellers of the browsed / liked *categories*
        self.adjust = adjust                     # live adjustments for price drops and trends

    # ------------------------------------------------------------------ public
    def recommend(self, req: Record, k: int = 10) -> list[str]:
        """10 distinct, available product ids for one request, best first. Never raises."""
        try:
            ctx = self.context(req)
            cands = self.candidates(ctx)
            if cands:
                scores = self.ranker.score(self.features(ctx, cands))
                if self.adjust:
                    scores = scores * self.adjustments(cands, ctx["now"])
                order = np.argsort(-scores, kind="stable")
                ranked = [cands[i] for i in order]
            else:
                ranked = []
            exclude = self.already_bought(ctx)
        except Exception as e:  # never let one bad request break the batch
            print(f"recommend failed for {req.get('request_id')}: {e!r}")
            ranked, exclude = [], set()
        return self.fill(ranked, k, exclude)

    @staticmethod
    def already_bought(ctx: dict[str, Any]) -> set[str]:
        """Products this shopper already bought (organic or in the carousel). Live they are almost never bought
        again (0.06 per 1,000 impressions vs 0.8 for an unrelated product), so they waste a slot."""
        out = {p for p, row in ctx["ui"].items() if row[2] > 0}
        out.update(p for p, (_, b) in ctx["cs"].items() if b > 0)
        return out

    def fill(self, ranked: list[str], k: int = 10, exclude: set[str] = frozenset()) -> list[str]:
        """Keep valid items, then pad with best sellers so the answer always has k distinct products."""
        s = self.store
        out: list[str] = []
        seen: set[str] = set()
        for p in list(ranked) + s.top_global:
            if p not in seen and p not in exclude and p not in s.unavailable and p in s.subcat:
                out.append(p)
                seen.add(p)
                if len(out) == k:
                    break
        return out

    def adjustments(self, cands: list[str], now: float) -> np.ndarray:
        """Multipliers for what the history cannot teach (1.0 offline):
        - products added during the event sell ~6x more per impression live -> x NEW_PRODUCT_BOOST
        - trending subcategories (purchase share last 2 h vs usual)
        Price drops are NOT boosted: live, discounted products did not sell more in the carousel."""
        s = self.store
        out = np.ones(len(cands))
        for i, p in enumerate(cands):
            if now - s.created.get(p, 0.0) < NEW_PRODUCT_DAYS * DAY:
                out[i] *= NEW_PRODUCT_BOOST
            out[i] *= s.trend_sc.get(s.subcat.get(p), 1.0)
        return out

    # ------------------------------------------------------------------ steps
    def context(self, req: Record) -> dict[str, Any]:
        """Everything about the request that features need, computed once."""
        s = self.store
        views = [v["product_id"] for v in (req.get("recent_views") or []) if isinstance(v, dict)]
        user = req.get("user_id") or ""
        ts = req.get("ts")
        now = to_epoch(ts) if isinstance(ts, str) else time.time()
        cov: Counter = Counter()
        for k, p in enumerate(reversed(views[-5:])):
            for rank, q in enumerate(s.coviews.get(p, [])):
                cov[q] += 0.7 ** k / (rank + 1)
        us, us_buy = s.user_subcats(user)
        uc: Counter = Counter()
        for sc, n in us.items():
            uc[s.cat_of_subcat.get(sc)] += n
        ub = s.user_brands(user)
        ua = s.user_attrs(user)
        ua_total = float(ua.sum())
        lift = (ua / ua_total) / s.attr_base if ua_total >= 4 else np.ones(len(ATTRIBUTES))   # 1 = average shopper
        v_attrs: Counter = Counter(a for p in views for a in s.attrs.get(p, ()))
        is_new = req.get("is_new_user")
        if is_new is None:            # history rows don't carry the flag: new = no organic history at all
            is_new = user not in s.us_index
        return {
            "user": user, "now": now, "views": views, "view_count": Counter(views),
            "v_subcats": Counter(s.subcat.get(p) for p in views), "v_cats": Counter(s.category.get(p) for p in views),
            "v_brands": Counter(s.brand.get(p) for p in views), "coview": cov,
            "ui": s.user_items(user), "cs": s.carousel(user),
            "us": us, "us_total": float(sum(us.values())), "us_buy": us_buy, "us_buy_total": float(sum(us_buy.values())),
            "uc": uc, "ub": ub, "ub_total": float(sum(ub.values())),
            "attr_lift": lift, "attr_total": ua_total, "v_attrs": v_attrs,
            "page": PAGES.get(req.get("page") or "", -1), "device": DEVICES.get(req.get("device") or "", -1),
            "region": REGIONS.get(req.get("region") or "", -1), "is_new_user": float(bool(is_new)),
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

        ui, cs = ctx["ui"], ctx["cs"]
        add(reversed(ctx["views"]), 20)                                                   # just viewed
        pop_b = lambda p: -s.pop.get(p, (0, 0, 0))[2]  # noqa: E731
        if ctx["attr_total"] >= 4:                       # products with the shopper's favourite attribute pairs
            top = [int(a) for a in np.argsort(-ctx["attr_lift"])[:4]]
            for pair in _pairs(top):
                add(sorted(s.by_attr_pair.get(tuple(sorted(pair)), []), key=pop_b), 8)
        for pair in _pairs([a for a, _ in ctx["v_attrs"].most_common(3)]):              # attributes being browsed
            add(sorted(s.by_attr_pair.get(tuple(sorted(pair)), []), key=pop_b), 6)
        carted = [p for p in ui if ui[p][1] > 0 and ui[p][2] == 0]                       # in the cart, not bought
        add(sorted(carted, key=lambda p: -ui[p][3])[:40], 30)
        add(sorted((p for p in cs if cs[p][1] == 0), key=lambda p: -cs[p][0])[:20], 15)  # clicked in the carousel
        new = self._new_products(ctx["now"])                                              # cold start: new products
        if new:
            cats = [c for c, _ in (ctx["v_cats"] + Counter()).most_common(2)] + [c for c, _ in ctx["uc"].most_common(2)]
            for c in dict.fromkeys(cats):
                add((p for p in new if s.category.get(p) == c), 4)
            add(sorted(new, key=lambda p: -s.pop.get(p, (0, 0, 0))[2])[:10], 6)
        add((q for q, _ in ctx["coview"].most_common(40)), 40)                           # co-viewed
        for sc, _ in (ctx["v_subcats"] + Counter()).most_common(3):                       # browsing subcategories
            add(s.top_by_subcat.get(sc, []), 10)
            add(self._newest(sc), 3)                                                      # cold start: new products
        for sc, _ in ctx["us"].most_common(3):                                            # long-term taste
            add(s.top_by_subcat.get(sc, []), 6)
            add(self._newest(sc), 2)
        if self.wide:                     # live, the category (not the subcategory) is what drives purchases
            top_by_cat = getattr(s, "top_by_cat", {})
            for c, _ in (ctx["v_cats"] + Counter()).most_common(2):                       # browsed categories
                add(top_by_cat.get(c, []), 15)
            for c, _ in ctx["uc"].most_common(3):                                         # liked categories
                add(top_by_cat.get(c, []), 10)
        add(s.top_global, 30)                                                             # best sellers
        return list(out)[: self.n_candidates]

    def _new_products(self, now: float) -> list[str]:
        """Products created less than NEW_PRODUCT_DAYS ago, newest first (cached for a minute)."""
        s = self.store
        if now - getattr(self, "_new_at", -1e18) > 60 or abs(now - self._new_at) > 3600:
            cutoff = now - NEW_PRODUCT_DAYS * DAY
            self._new_cache = sorted((p for p, t in s.created.items() if cutoff <= t <= now),
                                     key=lambda p: -s.created[p])
            self._new_at = now
        return self._new_cache

    def _newest(self, subcat: str) -> list[str]:
        s = self.store
        items = s.products_by_subcat.get(subcat, [])
        return sorted(items, key=lambda p: -s.created.get(p, 0))[:3]

    def features(self, ctx: dict[str, Any], cands: list[str], positions: list[int] | None = None) -> np.ndarray:
        s = self.store
        X = np.zeros((len(cands), len(FEATURES)), dtype=np.float64)
        n_views = len(ctx["views"])
        now = ctx["now"]
        us_total, us_buy_total, ub_total = ctx["us_total"], ctx["us_buy_total"], ctx["ub_total"]
        has_history = 1.0 if us_total > 0 else 0.0
        shown = s.shown.get(ctx["user"], {})
        lift, v_attrs = ctx["attr_lift"], ctx["v_attrs"]

        def attr_feats(p: str) -> tuple:
            at = s.attrs.get(p, ())
            if not at:
                return 1.0, 1.0, 1.0, 0.0
            ls = [float(lift[a]) for a in at]
            match = sum(v_attrs.get(a, 0) for a in at) / max(n_views, 1)
            return sum(ls) / len(ls), min(ls), max(ls), match
        for i, p in enumerate(cands):
            sc, cat, br = s.subcat.get(p), s.category.get(p), s.brand.get(p)
            ui = ctx["ui"].get(p)
            cs = ctx["cs"].get(p)
            pv, pc, pb, cvr, cart_rate = s.pop.get(p, (0.0, 0.0, 0.0, 0.005, 0.005))
            price = s.price.get(p, 0.0)
            X[i] = (
                positions[i] if positions else 0,
                1.0 if p in ctx["view_count"] else 0.0,
                ctx["view_count"].get(p, 0),
                n_views,
                ctx["v_subcats"].get(sc, 0),
                ctx["v_cats"].get(cat, 0),
                ctx["v_brands"].get(br, 0),
                ctx["coview"].get(p, 0.0),
                ui[2] if ui is not None else 0.0,
                ui[1] if ui is not None else 0.0,
                ui[0] if ui is not None else 0.0,
                min((now - ui[4]) / DAY, NEVER) if ui is not None and ui[4] > 0 else NEVER,
                min((now - ui[3]) / DAY, NEVER) if ui is not None and ui[3] > 0 else NEVER,
                cs[0] if cs is not None else 0.0,
                cs[1] if cs is not None else 0.0,
                ctx["us"].get(sc, 0.0) / us_total if us_total else 0.0,
                ctx["uc"].get(cat, 0.0) / us_total if us_total else 0.0,
                ctx["us_buy"].get(sc, 0.0) / us_buy_total if us_buy_total else 0.0,
                ctx["ub"].get(br, 0.0) / ub_total if ub_total else 0.0,
                has_history,
                pv, pc, pb, cvr, cart_rate,
                math.log1p(price),
                price / s.subcat_price.get(sc, price or 1.0) if price else 1.0,
                min((now - s.created.get(p, 0.0)) / DAY, 999.0),
                1.0 if now - s.created.get(p, 0.0) < NEW_PRODUCT_DAYS * DAY else 0.0,
                price / s.base_price.get(p, price) if price and s.base_price.get(p) else 1.0,
                s.trend_sc.get(sc, 1.0),
                shown.get(p, 0),
                *attr_feats(p),
                ctx["page"], ctx["device"], ctx["region"], ctx["is_new_user"],
            )
        return X


def _pairs(items: list) -> list[tuple]:
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


def request_from_row(row: Any) -> Record:
    """A shop.history_sessions / shop.requests row (dict or namedtuple) -> the dict recommend() expects."""
    get = row.get if isinstance(row, dict) else lambda k, d=None: getattr(row, k, d)
    return {"request_id": get("request_id"), "ts": get("ts"), "user_id": get("user_id"),
            "is_new_user": get("is_new_user", None), "recent_views": get("recent_views") or [],
            "page": get("page"), "device": get("device"), "region": get("region")}
