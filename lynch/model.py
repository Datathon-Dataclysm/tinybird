"""Lynch (v8) recommender = v7 pipeline (commit e986b41) with three evidence-driven changes (docs/experiments.md):
  1. feature set "spike-s4 + age_days" (+ optional MF): context and trend/fatigue dropped (no measurable value, and
     is_new_user/page would be near-constant in a final round of new shoppers); product age kept for new products.
  2. optional matrix-factorisation features: history SVD (mf_hist) + live fold-in of the shopper's events (mf_foldin).
  3. optional new-product calibration: v7 under-predicted products < 2 days old by about x1.6.
The model file declares its own feature names, so the same code serves v7-style and v8 models.

--- original v7 docstring ---
Personalised recommender (v7): session + affinity retrieval -> features -> LightGBM ranker.

    store = Store.load(live=True)             # live: all history + today's shop.events / catalog_updates
    model = Recommender(store, Ranker.load("models/ranker_v7.txt"))
    model.recommend(request)                  # -> 10 product ids, best first

How the shop simulator decides (found in the data, see starter/offline_live.py):
- every shopper has hidden tastes for categories and for the 20 product *attributes* written in the descriptions
  ("a luxury feel", "organic ingredients", ...); we estimate them from their organic events (history + live)
- what they are browsing right now (recent views) points the same way
- they almost never buy again something they already bought, and get tired of products shown again and again
- popular products sell more; new products attract attention

1. Candidates (~350), two families:
   - session (what they browse NOW): viewed products, best sellers of the browsed subcategories, products co-viewed
     with them, products sharing the browsed attributes, new products of those subcategories
   - taste: carted / carousel-clicked products, and the top of the WHOLE catalogue by affinity (taste + popularity)
   v6 used only the second family and lost the session: category-page and new-user conversion dropped (v7 fixes it).
2. Features (29, see FEATURES): affinity of the user and the session with the product's category / subcategory /
   attributes, co-views with the session, user x product history, popularity, freshness, fatigue, context.
   v6 had dropped the session-subcategory and co-view features; putting them back moves real purchases into the
   top 3 more often (54% -> 59%, offline_live.py e2e).
3. Ranker: LightGBM trained on our live feedback (starter/train_live.py); products bought before are left out.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rawtree import Record, query_df, query_parts, to_epoch

PAGES = {"home": 0, "product": 1, "category": 2}
DEVICES = {"web": 0, "ios": 1, "android": 2}
REGIONS = {r: i for i, r in enumerate([
    "Andalucía", "Aragón", "Canarias", "Castilla y León", "Catalunya",
    "Comunitat Valenciana", "Galicia", "Madrid", "Murcia", "País Vasco"])}
DAY = 86400.0
NEVER = 999.0            # "days since" when it never happened
POP_DAYS = 7             # popularity window
TREND_HOURS = 2          # trend window (live)
NEW_PRODUCT_DAYS = 2     # a product created less than this ago counts as new

# Every product description carries exactly 2 of these 20 attributes.
ATTRIBUTES = [
    "a sporty style", "a family-size format", "organic ingredients", "a luxury feel", "a minimalist design",
    "built for professionals", "premium materials", "designed for kids", "bold colours", "wireless connectivity",
    "whisper-quiet operation", "a vintage look", "smart app control", "made from recycled materials",
    "great value for money", "a travel-ready build", "a waterproof shell", "extra durability", "a handmade finish",
    "compact and lightweight"]
ATTR_INDEX = {a: i for i, a in enumerate(ATTRIBUTES)}
ATTR_SQL = "[" + ",".join(f"'{a}'" for a in ATTRIBUTES) + "]"

AFF_ALPHA = 10.0         # pseudo-events of the average shopper mixed into a user's profile (shrinkage)
SES_ALPHA = 2.0          # same for the session profile (few views, but very informative)
# Retrieval weights, tuned offline to put as many real purchases as possible in the top 150 (offline_live.py tune)
W_SUBCAT, W_ATTR, W_SESSION, W_POP = 0.25, 1.0, 0.5, 0.6
N_RETRIEVE = 150       # top of the whole catalogue by affinity (on top of the session candidates)

FEATURES = [
    "pos",                                                       # slot (training only; 0 at inference)
    "in_views", "ses_cat", "ses_subcat", "ses_attr", "attr_views_match", "coview", "n_views",   # session (now)
    "ui_buys", "ui_carts", "days_since_buy", "days_since_cart",  # user x product, organic
    "cs_clicks",                                                 # user x product, carousel
    "aff_cat", "aff_subcat", "aff_attr", "aff_attr_min", "attr_lift_mean",   # long-term taste
    "pop_views", "pop_buys", "cvr", "cart_rate",                 # product popularity (shares, scale-free)
    "age_days", "trend", "times_shown",                          # freshness, trend, fatigue
    "page", "device", "region", "is_new_user",                   # context
]
CATEGORICAL = ["page", "device", "region"]
MF_FEATURES = ["mf_hist", "mf_foldin"]
FEATURES_ALL = FEATURES + MF_FEATURES          # what features() computes; a model uses the subset it was trained on
CONTEXT = ["page", "device", "region", "is_new_user"]
DYNAMICS = ["age_days", "trend", "times_shown"]
FEATURE_SETS = {                                # "spike-*" names shared with the codex experiments
    "spike-s1": FEATURES,
    "spike-s4-age": [f for f in FEATURES if f not in CONTEXT + DYNAMICS] + ["age_days"],
    "spike-s4-age-mf": [f for f in FEATURES if f not in CONTEXT + DYNAMICS] + ["age_days"] + MF_FEATURES,
}
NEW_DAYS_CAL = 2.0                              # age (days) below which the new-product calibration applies
LIVE_ITEMS = 50                                 # recent live products kept per shopper for the MF fold-in


def parse_attrs(description: str) -> tuple[int, ...]:
    """The attribute indices mentioned in a product description."""
    d = description or ""
    return tuple(i for i, a in enumerate(ATTRIBUTES) if a in d)


def _until(until: str | None, col: str = "ts") -> str:
    return f" AND {col} < '{until}'" if until else ""


def _index(users: np.ndarray) -> dict[str, tuple[int, int]]:
    """Sorted user column -> {user: (start, end)} row slice."""
    keys, starts = np.unique(users, return_index=True)
    ends = np.append(starts[1:], len(users))
    return {u: (s, e) for u, s, e in zip(keys, starts, ends)}


# ====================================================================================================== store
class Store:
    """Everything the recommender knows, in memory for millisecond lookups. History is loaded once; the live
    process keeps it fresh (ingest / apply_catalog_update / load_popularity / load_trends)."""

    def __init__(self) -> None:
        self.unavailable: set[str] = set()                                     # out of stock / inactive right now
        self.live_ui: dict[str, dict[str, list[float]]] = defaultdict(dict)    # user -> item -> [carts, buys, last_cart, last_buy]
        self.live_us: dict[str, Counter] = defaultdict(Counter)                 # user -> subcategory -> events
        self.live_ua: dict[str, np.ndarray] = {}                                # user -> events per attribute
        self.live_cs: dict[str, dict[str, float]] = {}                          # user -> item -> clicks in our carousel
        self.shown: dict[str, Counter] = defaultdict(Counter)                   # user -> item -> times we showed it
        self.trend_sc: dict[str, float] = {}                                    # subcategory -> trend (1 = usual)
        self.catalog_version = 0
        self.last_event_ts = ""
        self.live_items: dict[str, deque] = defaultdict(_recent_items)   # user -> recent products (MF fold-in)
        self.mf_items: dict[str, int] = {}                                      # product -> row of mf_V
        self.mf_users: dict[str, int] = {}                                      # user -> row of mf_U (history)
        self.mf_V = self.mf_U = None

    def load_mf(self, path: str | Path) -> bool:
        """Item and user factors built once from the (immutable) history by lynch/build_mf.py."""
        path = Path(path)
        if not path.exists():
            return False
        z = np.load(path, allow_pickle=True)
        self.mf_items = {p: i for i, p in enumerate(z["items"])}
        self.mf_users = {u: i for i, u in enumerate(z["users"])}
        self.mf_V, self.mf_U = z["V"], z["U"]
        return True

    @classmethod
    def load(cls, until: str | None = None, live: bool = False, verbose: bool = True) -> "Store":
        s = cls()
        steps = [("catalog", lambda: s._load_catalog(live)),
                 ("user x product", lambda: s._load_user_items(until)),
                 ("user x subcategory", lambda: s._load_user_subcats(until)),
                 ("user x attribute", lambda: s._load_user_attrs(until)),
                 ("carousel history", lambda: s._load_carousel(until)),
                 ("co-views", lambda: s._load_coviews(until)),
                 ("popularity", lambda: s.load_popularity(until, live))]
        if live:
            steps += [("live events", s._load_live_events), ("trends", s.load_trends)]
        for name, step in steps:
            t = time.time()
            step()
            if verbose:
                print(f"  loaded {name} in {time.time() - t:.1f}s", flush=True)
        s.compute_priors()
        return s

    # ------------------------------------------------------------------ loading (SQL in RawTree)
    def _load_catalog(self, live: bool) -> None:
        """shop.catalog, plus (live) the latest shop.catalog_updates record of each product and the stock-outs."""
        cols = """product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
                  is_active::Bool AS is_active, toString(created_at::DateTime64(3)) AS created_at,
                  description::String AS description"""
        if live:
            cat = query_df(f"""
                SELECT product_id, argMax(category, v) AS category, argMax(subcategory, v) AS subcategory,
                       argMax(is_active, v) AS is_active, argMax(created_at, v) AS created_at,
                       argMax(description, v) AS description
                FROM (SELECT {cols}, '' AS v FROM shop.catalog
                      UNION ALL SELECT {cols}, toString(ts::DateTime64(3)) AS v FROM shop.catalog_updates)
                GROUP BY product_id""")
            stock = query_df("""
                SELECT product_id::String AS product_id FROM shop.catalog_updates
                WHERE change::String IN ('out_of_stock', 'back_in_stock') GROUP BY product_id
                HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'""")
            out = set(stock.product_id) if len(stock) else set()
        else:
            cat, out = query_df(f"SELECT {cols} FROM shop.catalog"), set()
        self.subcat = dict(zip(cat.product_id, cat.subcategory))
        self.category = dict(zip(cat.product_id, cat.category))
        self.created = {p: to_epoch(c) for p, c in zip(cat.product_id, cat.created_at)}
        self.attrs = {p: parse_attrs(d) for p, d in zip(cat.product_id, cat.description)}
        self.cat_of_subcat = dict(zip(cat.subcategory, cat.category))
        self.by_attr_pair: dict[tuple, list[str]] = defaultdict(list)          # (a, b) -> products with both
        self.by_subcat: dict[str, list[str]] = defaultdict(list)               # subcategory -> products
        for p, at in self.attrs.items():
            self.by_subcat[self.subcat[p]].append(p)
            if len(at) == 2:
                self.by_attr_pair[at].append(p)
        self.unavailable = set(cat.product_id[~cat.is_active.astype(bool)]) | out

    def _load_user_items(self, until: str | None) -> None:
        """user x product: carts and purchases (views alone carry no signal). A purchase means 'do not show again'."""
        df = query_parts(f"""
            SELECT user_id::String AS u, product_id::String AS p,
                   toUInt32(countIf(event_type = 'add_to_cart')) AS c, toUInt32(countIf(event_type = 'purchase')) AS b,
                   toUnixTimestamp(maxIf(ts::DateTime64(3), event_type = 'add_to_cart')) AS lc,
                   toUnixTimestamp(maxIf(ts::DateTime64(3), event_type = 'purchase')) AS lb
            FROM shop.history_events WHERE {{part}}{_until(until)}
            GROUP BY u, p HAVING b > 0 OR c > 0""", parts=4)
        df = df.sort_values("u", kind="stable").reset_index(drop=True)
        self.ui_index = _index(df.u.to_numpy())
        self.ui_p = df.p.to_numpy()
        self.ui_vals = df[["c", "b", "lc", "lb"]].to_numpy(dtype=np.float64)

    def _load_user_subcats(self, until: str | None) -> None:
        """Events of each user per subcategory (categories are derived from it)."""
        df = query_parts(f"""
            SELECT e.user_id::String AS u, c.subcategory AS sc, toUInt32(count()) AS n
            FROM shop.history_events AS e
            INNER JOIN (SELECT product_id::String AS p, subcategory::String AS subcategory FROM shop.catalog) AS c
                ON c.p = e.product_id::String
            WHERE {{part}}{_until(until, 'e.ts')}
            GROUP BY u, sc""", parts=4, key="e.user_id::String")
        df = df.sort_values("u", kind="stable").reset_index(drop=True)
        self.us_index = _index(df.u.to_numpy())
        self.us_sc = df.sc.to_numpy()
        self.us_n = df.n.to_numpy(dtype=np.float64)

    def _load_user_attrs(self, until: str | None) -> None:
        """Events of each user per product attribute, parsed from the descriptions inside RawTree."""
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

    def _load_carousel(self, until: str | None) -> None:
        """Products each user clicked in the old carousel (shop.history_sessions)."""
        df = query_df(f"""
            SELECT user_id::String AS u, clicked::String AS p, toUInt32(count()) AS c
            FROM shop.history_sessions WHERE clicked::String != ''{_until(until)} GROUP BY u, p""")
        self.cs: dict[str, dict[str, float]] = defaultdict(dict)
        for u, p, c in zip(df.u, df.p, df.c):
            self.cs[u][p] = float(c)

    def _load_coviews(self, until: str | None) -> None:
        """For every product, the 20 products most viewed in the same organic sessions."""
        df = query_df(f"""
            SELECT a.product_id AS p, b.product_id AS q, count() AS n
            FROM (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events
                  WHERE event_type = 'view'{_until(until)}) AS a
            INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events
                        WHERE event_type = 'view'{_until(until)}) AS b ON a.session_id = b.session_id
            WHERE a.product_id != b.product_id
            GROUP BY p, q ORDER BY n DESC LIMIT 20 BY p""")
        self.coviews = df.groupby("p").q.apply(list).to_dict()

    def load_popularity(self, until: str | None = None, live: bool = False) -> None:
        """Views / carts / purchases per product over the last POP_DAYS days (history + live)."""
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
        """Counts per product (columns p, v, c, b) -> shares (1.0 = an average product) and best-seller list.
        Shares, not counts: the live stream is ~30x denser than the history."""
        n = max(len(self.subcat), 1)
        tv, tb = (max(float(df[k].sum()), 1.0) / n for k in ("v", "b"))
        self.pop = {p: (v / tv, b / tb, (b + 0.1) / (v + 20.0), (c + 0.1) / (v + 20.0))
                    for p, v, c, b in zip(df.p, df.v.astype(float), df.c.astype(float), df.b.astype(float))}
        ranked = df.sort_values(["b", "c", "v"], ascending=False)
        self.top_global = ranked.p.tolist()
        self.top_by_subcat = ranked.assign(sc=ranked.p.map(self.subcat)).groupby("sc", sort=False).p.apply(
            lambda x: x.head(20).tolist()).to_dict()

    def load_trends(self) -> None:
        """Live: purchases per subcategory in the last TREND_HOURS vs the last week of history."""
        df = query_df(f"""
            SELECT c.subcategory AS sc, countIf(e.src = 'live') AS a, countIf(e.src = 'ref') AS r
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
        """Trend = purchases in the window / purchases expected from the usual share (shrunk towards 1)."""
        A, R = float(sum(live_by_sc.values())), max(float(sum(ref_by_sc.values())), 1.0)
        self.trend_sc = {sc: float(np.clip((live_by_sc.get(sc, 0.0) + 20.0) / (A * ref_by_sc.get(sc, 0.0) / R + 20.0), 0.5, 3.0))
                         for sc in set(ref_by_sc) | set(live_by_sc)}

    def _load_live_events(self) -> None:
        """Today's organic activity (shop.events), aggregated like the history."""
        df = query_parts("""
            SELECT user_id::String AS u, product_id::String AS p, event_type::String AS e,
                   toUInt32(count()) AS n, toUnixTimestamp(max(ts::DateTime64(3))) AS last
            FROM shop.events WHERE {part} GROUP BY u, p, e""", parts=4)
        self.ingest(df.itertuples(index=False))
        last = query_df("SELECT toString(max(ts::DateTime64(3))) AS t FROM shop.events")
        self.last_event_ts = last.t[0] if len(last) else ""

    def compute_priors(self) -> None:
        """The 'average shopper': overall share of events per category / subcategory / attribute."""
        self.cat_vocab = {c: i for i, c in enumerate(sorted({c for c in self.category.values() if c}))}
        self.sc_vocab = {c: i for i, c in enumerate(sorted({c for c in self.subcat.values() if c}))}
        sc_n = np.zeros(len(self.sc_vocab))
        np.add.at(sc_n, [self.sc_vocab.get(x, 0) for x in self.us_sc], self.us_n)
        cat_n = np.zeros(len(self.cat_vocab))
        for sc, i in self.sc_vocab.items():
            cat_n[self.cat_vocab[self.cat_of_subcat[sc]]] += sc_n[i]
        attr_n = np.bincount(self.ua_attr, weights=self.ua_n, minlength=len(ATTRIBUTES))
        self.p_sc = (sc_n + 1) / (sc_n.sum() + len(sc_n))
        self.p_cat = (cat_n + 1) / (cat_n.sum() + len(cat_n))
        self.p_attr = (attr_n + 1) / (attr_n.sum() + len(attr_n))

    # ------------------------------------------------------------------ live updates
    def ingest(self, events: Iterable) -> None:
        """Organic events (user, product, event_type, count, last_ts) -> live user tables."""
        for u, p, e, n, last in events:
            n, last = float(n), float(last)
            self.live_items[u].append(p)
            sc = self.subcat.get(p)
            if sc:
                self.live_us[u][sc] += n
            at = self.attrs.get(p)
            if at:
                vec = self.live_ua.get(u)
                if vec is None:
                    vec = self.live_ua[u] = np.zeros(len(ATTRIBUTES))
                for a in at:
                    vec[a] += n
            if e == "add_to_cart" or e == "purchase":
                row = self.live_ui[u].setdefault(p, [0.0, 0.0, 0.0, 0.0])
                k = 0 if e == "add_to_cart" else 1
                row[k] += n
                row[2 + k] = max(row[2 + k], last)

    def apply_catalog_update(self, u: Record) -> None:
        """One shop.catalog_updates record: new product, price change, stock-out or back in stock."""
        p, change = u["product_id"], u.get("change")
        self.catalog_version += 1
        if u.get("subcategory"):
            self.subcat[p] = u["subcategory"]
            self.category[p] = u.get("category") or self.category.get(p)
        if u.get("description"):
            at = parse_attrs(u["description"])
            if p not in self.attrs and len(at) == 2:
                self.by_attr_pair[at].append(p)
            self.attrs[p] = at
        if change == "new_product" and p not in self.by_subcat.get(self.subcat.get(p), []):
            self.by_subcat[self.subcat.get(p)].append(p)
        if change == "new_product":
            self.created[p] = to_epoch(u["created_at"]) if u.get("created_at") else time.time()
        if change == "out_of_stock" or u.get("is_active") is False:
            self.unavailable.add(p)
        elif change in ("back_in_stock", "new_product"):
            self.unavailable.discard(p)

    # ------------------------------------------------------------------ lookups
    def user_items(self, user: str) -> dict[str, np.ndarray]:
        """product -> [carts, buys, last_cart_ts, last_buy_ts] for one user (history + live)."""
        s, e = self.ui_index.get(user, (0, 0))
        out = {p: self.ui_vals[i].copy() for i, p in zip(range(s, e), self.ui_p[s:e])}
        for p, row in self.live_ui.get(user, {}).items():
            if p in out:
                out[p][:2] += row[:2]
                out[p][2:] = np.maximum(out[p][2:], row[2:])
            else:
                out[p] = np.array(row, dtype=np.float64)
        return out

    def user_subcats(self, user: str) -> Counter:
        s, e = self.us_index.get(user, (0, 0))
        c = Counter(dict(zip(self.us_sc[s:e], self.us_n[s:e])))
        c.update(self.live_us.get(user, {}))
        return c

    def user_attrs(self, user: str) -> np.ndarray:
        out = np.zeros(len(ATTRIBUTES))
        s, e = self.ua_index.get(user, (0, 0))
        if e > s:
            np.add.at(out, self.ua_attr[s:e], self.ua_n[s:e])
        live = self.live_ua.get(user)
        return out + live if live is not None else out

    def carousel(self, user: str) -> dict[str, float]:
        """product -> clicks in the carousel (history + our feedback)."""
        hist, live = self.cs.get(user, {}), self.live_cs.get(user, {})
        if not live:
            return hist
        out = dict(hist)
        for p, c in live.items():
            out[p] = out.get(p, 0.0) + c
        return out

    def item_index(self) -> dict[str, np.ndarray]:
        """The catalogue as arrays, to score every product at once (rebuilt when catalogue or popularity change)."""
        key = (self.catalog_version, id(self.pop))
        if getattr(self, "_ix_key", None) != key:
            items = [p for p, sc in self.subcat.items() if sc in self.sc_vocab and self.category.get(p) in self.cat_vocab]
            at = [self.attrs.get(p, ()) for p in items]
            self._ix = {
                "items": np.array(items, dtype=object),
                "cat": np.array([self.cat_vocab[self.category[p]] for p in items]),
                "sc": np.array([self.sc_vocab[self.subcat[p]] for p in items]),
                "a1": np.array([a[0] if len(a) > 0 else -1 for a in at]),
                "a2": np.array([a[1] if len(a) > 1 else -1 for a in at]),
                "ok": np.array([p not in self.unavailable for p in items]),
                "logpop": np.log(np.array([self.pop.get(p, (0, 0, 0, 0))[1] for p in items]) + 0.05),
            }
            self._ix_key = key
        return self._ix


# ====================================================================================================== ranker
class Ranker:
    """LightGBM model scoring each candidate (trained by starter/train_live.py)."""

    def __init__(self, booster: Any = None, new_boost: float = 1.0) -> None:
        self.booster = booster
        self.new_boost = new_boost
        self.cols = None
        if booster is not None:
            missing = [f for f in booster.feature_name() if f not in FEATURES_ALL]
            if missing:                           # an unknown feature would silently mis-score
                raise ValueError(f"model needs features this code does not compute: {missing}")
            self.cols = [FEATURES_ALL.index(f) for f in booster.feature_name()]

    @classmethod
    def load(cls, path: str | Path) -> "Ranker":
        """Model file + optional sidecar <name>.json {"new_boost": float} written by lynch/train_v8.py."""
        import json
        import lightgbm as lgb
        # Read as text without carriage returns: on Windows, git (core.autocrlf) can check the .txt out with CRLF line
        # endings and LightGBM then fails with "Model format error, expect a tree here" (see /.gitattributes).
        booster = lgb.Booster(model_str=Path(path).read_text(encoding="utf-8").replace("\r", ""))
        side = Path(path).with_suffix(".json")
        boost = float(json.loads(side.read_text()).get("new_boost", 1.0)) if side.exists() else 1.0
        return cls(booster, boost)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.booster is not None:
            p = self.booster.predict(X[:, self.cols])
            if self.new_boost != 1.0:             # calibrated multiplier for products < NEW_DAYS_CAL days old
                p = p * np.where(X[:, FEATURES_ALL.index("age_days")] < NEW_DAYS_CAL, self.new_boost, 1.0)
            return p
        f = {name: X[:, i] for i, name in enumerate(FEATURES)}      # no model yet: a sensible hand-made score
        return (f["aff_cat"] + f["aff_attr"] + 0.5 * f["ses_cat"] + 0.5 * f["ses_attr"] + 1.5 * f["in_views"]
                + 1.0 * np.minimum(f["ui_carts"], 1) - 5.0 * np.minimum(f["ui_buys"], 1) + 0.6 * np.log(f["pop_buys"] + 0.05))


# ====================================================================================================== recommender
class Recommender:
    def __init__(self, store: Store, ranker: Ranker | None = None, n_candidates: int = 400,
                 n_retrieve: int = N_RETRIEVE, max_per_subcat: int | None = None) -> None:
        self.store = store
        self.ranker = ranker or Ranker()
        # At most N products of one subcategory in the 10 (None = off; lynch-v9a uses 3); the rest move to the end.
        # A request ends in at most one purchase, so near-identical products compete for it. Live evidence is mixed
        # and within noise (see README: v9 2.18 -> 2.39, lynch 15 min the other way). A cap per CATEGORY (only 10
        # exist) was worse: it replaced 3 of the 10 products.
        self.max_per_subcat = max_per_subcat
        self.n_candidates = n_candidates
        self.n_retrieve = n_retrieve

    def recommend(self, req: Record, k: int = 10) -> list[str]:
        """10 distinct, available product ids for one request, best first. Never raises."""
        try:
            ctx = self.context(req)
            cands = self.candidates(ctx)
            scores = self.ranker.score(self.features(ctx, cands)) if cands else np.zeros(0)
            ranked = [cands[i] for i in np.argsort(-scores, kind="stable")]
            if self.max_per_subcat:
                head, tail, used = [], [], Counter()
                for p in ranked:
                    sc = self.store.subcat.get(p)
                    (head if used[sc] < self.max_per_subcat else tail).append(p)
                    used[sc] += 1
                ranked = head + tail
            exclude = ctx["bought"]
        except Exception as e:  # never let one bad request break the batch
            print(f"recommend failed for {req.get('request_id')}: {e!r}")
            ranked, exclude = [], set()
        return self.fill(ranked, k, exclude)

    def fill(self, ranked: list[str], k: int = 10, exclude: set[str] = frozenset()) -> list[str]:
        """Keep valid items, then pad with best sellers so the answer always has k distinct products."""
        s = self.store
        out: list[str] = []
        for p in list(ranked) + s.top_global:
            if p not in out and p not in exclude and p not in s.unavailable and p in s.subcat:
                out.append(p)
                if len(out) == k:
                    break
        return out

    # ------------------------------------------------------------------ steps
    def profile(self, sc_counts: Counter, attr_counts: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
        """Counts of a user (or a session) -> shrunk log-lift per category / subcategory / attribute:
        log((n + alpha * p) / (N + alpha)) - log(p). 0 = like the average shopper, > 0 = likes it more.
        With no data every lift is 0: new shoppers fall back to the session and to popularity."""
        s = self.store
        sc_n, cat_n = np.zeros(len(s.sc_vocab)), np.zeros(len(s.cat_vocab))
        for sc, n in sc_counts.items():
            i = s.sc_vocab.get(sc)
            if i is not None:
                sc_n[i] += n
                cat_n[s.cat_vocab[s.cat_of_subcat[sc]]] += n

        def lift(n, prior):
            return np.log((n + alpha * prior) / (n.sum() + alpha)) - np.log(prior)
        return {"cat": lift(cat_n, s.p_cat), "sc": lift(sc_n, s.p_sc), "attr": lift(np.asarray(attr_counts, float), s.p_attr)}

    def context(self, req: Record) -> dict[str, Any]:
        """Everything about the request that candidates and features need, computed once."""
        s = self.store
        views = [v["product_id"] for v in (req.get("recent_views") or []) if isinstance(v, dict)]
        user = req.get("user_id") or ""
        ts = req.get("ts")
        ua = s.user_attrs(user)
        v_attrs = Counter(a for p in views for a in s.attrs.get(p, ()))
        ses_ua = np.zeros(len(ATTRIBUTES))
        for a, n in v_attrs.items():
            ses_ua[a] = n
        ui = s.user_items(user)
        coview: Counter = Counter()                                  # products co-viewed with the last 5 views
        for k, p in enumerate(reversed(views[-5:])):
            for rank, q in enumerate(s.coviews.get(p, [])):
                coview[q] += 0.7 ** k / (rank + 1)
        mf_hist_u = mf_fold_u = None                                 # MF user vectors (history / live fold-in)
        if s.mf_V is not None:
            if user in s.mf_users:
                mf_hist_u = s.mf_U[s.mf_users[user]]
            idx = [s.mf_items[p] for p in list(s.live_items.get(user, ())) + views if p in s.mf_items]
            if idx:
                mf_fold_u = s.mf_V[idx].mean(0)
        is_new = req.get("is_new_user")
        if is_new is None:            # history rows don't carry the flag: new = no organic history at all
            is_new = user not in s.us_index
        return {
            "user": user, "now": to_epoch(ts) if isinstance(ts, str) else time.time(),
            "views": views, "in_views": set(views), "v_attrs": v_attrs, "coview": coview,
            "ui": ui, "bought": {p for p, row in ui.items() if row[1] > 0}, "cs": s.carousel(user),
            "aff": self.profile(s.user_subcats(user), ua, AFF_ALPHA),
            "ses": self.profile(Counter(s.subcat.get(p) for p in views), ses_ua, SES_ALPHA),
            "attr_ratio": (ua / ua.sum()) / s.p_attr if ua.sum() >= 4 else np.ones(len(ATTRIBUTES)),
            "page": PAGES.get(req.get("page") or "", -1), "device": DEVICES.get(req.get("device") or "", -1),
            "region": REGIONS.get(req.get("region") or "", -1), "is_new_user": float(bool(is_new)),
            "mf_hist_u": mf_hist_u, "mf_fold_u": mf_fold_u,
        }

    def retrieve(self, ctx: dict[str, Any], k: int) -> list[str]:
        """Score EVERY available product by taste + session + popularity, keep the top k."""
        ix = self.store.item_index()
        aff, ses = ctx["aff"], ctx["ses"]
        a1, a2 = ix["a1"], ix["a2"]

        def part(p):
            attrs = np.where(a1 >= 0, p["attr"][a1], 0) + np.where(a2 >= 0, p["attr"][a2], 0)
            cat = p["cat"][ix["cat"]]
            return cat + W_SUBCAT * (p["sc"][ix["sc"]] - cat) + W_ATTR * attrs
        score = part(aff) + W_SESSION * part(ses) + W_POP * ix["logpop"]
        score = np.where(ix["ok"], score, -np.inf)
        k = min(k, len(score) - 1)
        top = np.argpartition(-score, k)[:k]
        return list(ix["items"][top[np.argsort(-score[top])]])

    def candidates(self, ctx: dict[str, Any]) -> list[str]:
        s = self.store
        out: dict[str, None] = {}
        bought = ctx["bought"]

        def add(items: Iterable[str], limit: int) -> None:
            n = 0
            for p in items:
                if n >= limit:
                    break
                if p in s.subcat and p not in s.unavailable and p not in bought:
                    out.setdefault(p)
                    n += 1

        ui, cs, views = ctx["ui"], ctx["cs"], ctx["views"]
        pop_b = lambda p: -s.pop.get(p, (0, 0, 0, 0))[1]  # noqa: E731
        # --- session: what they are browsing now (decisive on category pages and for new shoppers)
        add(reversed(views), 20)                                                          # just viewed
        for sc, _ in Counter(s.subcat.get(p) for p in views).most_common(3):             # browsed subcategories
            add(s.top_by_subcat.get(sc, []), 10)
            add(sorted(s.by_subcat.get(sc, []), key=lambda p: -s.created.get(p, 0))[:3], 3)   # their newest products
        add((q for q, _ in ctx["coview"].most_common(40)), 40)                           # co-viewed with them
        for a, b in _pairs([a for a, _ in ctx["v_attrs"].most_common(3)]):              # the browsed attributes
            add(sorted(s.by_attr_pair.get(tuple(sorted((a, b))), []), key=pop_b), 6)
        # --- taste: what they liked before + best affinity over the whole catalogue
        add(sorted((p for p in ui if ui[p][0] > 0), key=lambda p: -ui[p][2]), 15)       # in the cart
        add(sorted(cs, key=lambda p: -cs[p]), 8)                                          # clicked in the carousel
        add(self.retrieve(ctx, self.n_retrieve), self.n_retrieve)
        add(self._new_products(ctx["now"]), 5)                                            # cold start: new products
        add(s.top_global, 15)                                                             # best sellers
        return list(out)[: self.n_candidates]

    def _new_products(self, now: float) -> list[str]:
        """Products created less than NEW_PRODUCT_DAYS ago, most popular first (cached for a minute)."""
        s = self.store
        if abs(now - getattr(self, "_new_at", -1e18)) > 60:
            new = [p for p, t in s.created.items() if now - NEW_PRODUCT_DAYS * DAY <= t <= now]
            self._new_cache = sorted(new, key=lambda p: -s.pop.get(p, (0, 0, 0, 0))[1])
            self._new_at = now
        return self._new_cache

    def features(self, ctx: dict[str, Any], cands: list[str], positions: list[int] | None = None) -> np.ndarray:
        s = self.store
        X = np.zeros((len(cands), len(FEATURES_ALL)), dtype=np.float64)
        now, n_views, coview = ctx["now"], max(len(ctx["views"]), 1), ctx["coview"]
        aff, ses, ratio, v_attrs = ctx["aff"], ctx["ses"], ctx["attr_ratio"], ctx["v_attrs"]
        shown = s.shown.get(ctx["user"], {})
        for i, p in enumerate(cands):
            ic, isc = s.cat_vocab.get(s.category.get(p), -1), s.sc_vocab.get(s.subcat.get(p), -1)
            at = s.attrs.get(p, ())
            ui, cs = ctx["ui"].get(p), ctx["cs"].get(p, 0.0)
            pv, pb, cvr, cart_rate = s.pop.get(p, (0.0, 0.0, 0.005, 0.005))
            ua = [aff["attr"][a] for a in at] or [0.0]
            X[i, :len(FEATURES)] = (
                positions[i] if positions else 0,
                1.0 if p in ctx["in_views"] else 0.0,
                ses["cat"][ic] if ic >= 0 else 0.0,
                ses["sc"][isc] if isc >= 0 else 0.0,
                sum(ses["attr"][a] for a in at),
                sum(v_attrs.get(a, 0) for a in at) / n_views,
                coview.get(p, 0.0),
                len(ctx["views"]),
                ui[1] if ui is not None else 0.0,
                ui[0] if ui is not None else 0.0,
                min((now - ui[3]) / DAY, NEVER) if ui is not None and ui[3] > 0 else NEVER,
                min((now - ui[2]) / DAY, NEVER) if ui is not None and ui[2] > 0 else NEVER,
                cs,
                aff["cat"][ic] if ic >= 0 else 0.0,
                aff["sc"][isc] if isc >= 0 else 0.0,
                sum(ua), min(ua),
                float(np.mean([ratio[a] for a in at])) if at else 1.0,
                pv, pb, cvr, cart_rate,
                min((now - s.created.get(p, 0.0)) / DAY, NEVER),
                s.trend_sc.get(s.subcat.get(p), 1.0),
                shown.get(p, 0),
                ctx["page"], ctx["device"], ctx["region"], ctx["is_new_user"],
            )
        if s.mf_V is not None and (ctx["mf_hist_u"] is not None or ctx["mf_fold_u"] is not None):
            rows = [(i, s.mf_items[p]) for i, p in enumerate(cands) if p in s.mf_items]   # new products: 0 (unknown)
            if rows:
                ri, vi = map(np.array, zip(*rows))
                Vc = s.mf_V[vi]
                if ctx["mf_hist_u"] is not None:
                    X[ri, len(FEATURES)] = Vc @ ctx["mf_hist_u"]
                if ctx["mf_fold_u"] is not None:
                    X[ri, len(FEATURES) + 1] = Vc @ ctx["mf_fold_u"]
        return X


def _recent_items() -> deque:
    """Factory for Store.live_items (module-level so the store can be pickled)."""
    return deque(maxlen=LIVE_ITEMS)


def _pairs(items: list) -> list[tuple]:
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


def request_from_row(row: Any) -> Record:
    """A shop.history_sessions / shop.requests row (dict or namedtuple) -> the dict recommend() expects."""
    get = row.get if isinstance(row, dict) else lambda k, d=None: getattr(row, k, d)
    return {"request_id": get("request_id"), "ts": get("ts"), "user_id": get("user_id"),
            "is_new_user": get("is_new_user", None), "recent_views": get("recent_views") or [],
            "page": get("page"), "device": get("device"), "region": get("region")}
