"""(lynch copy of v7 train_live.py: imports, cache and model folder adapted; replay unchanged.)

Learn from the live stream: train the ranker on what the live simulator actually did with our lists.

    python starter/train_live.py              # replay the day, train variants, compare them on the latest requests
    python starter/train_live.py --final      # same, then train the best variant on ALL live rows -> models/ranker_v6.txt

Why: the live shoppers do not behave like the history said (e.g. a product the shopper already bought is almost
never bought again live, while history suggested the opposite). team_NN.feedback tells, for every request,
the 10 products shown and what was clicked / bought: that is labelled training data from the real simulator.

How (no peeking into the future): we *replay* the day in time order. The store starts with the full history; then
shop.events, shop.catalog_updates and our own feedback are applied as they happened, and every request is turned
into training rows with the features the live system would have computed at that very moment (including the
subcategory trends and how many times we had already shown each product to that shopper).

The live process (starter/live.py) reloads the model file by itself when this script rewrites it.
"""
from __future__ import annotations

import argparse
import heapq
import os
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CATEGORICAL, FEATURES, FEATURES_ALL, TREND_HOURS, Recommender, Store, request_from_row  # noqa: E402
from rawtree import ROOT, TEAM, be_nice, query_df, to_epoch  # noqa: E402
import pickle  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache"


def cached(name: str, build):
    """Pickle cache (lynch/.cache), replacing v7's starter/train.py helper."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    obj = build()
    path.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    return obj


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

MODELS = Path(__file__).resolve().parent / "models"


def fetch_live() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """All live events, catalog updates and our feedback (joined with the request) so far."""
    hours = query_df("SELECT DISTINCT toString(toStartOfHour(ts::DateTime64(3))) AS h FROM shop.events ORDER BY h").h
    parts = []
    for h in hours:
        parts.append(query_df(f"""
            SELECT user_id::String AS u, product_id::String AS p, event_type::String AS e, toString(ts::DateTime64(3)) AS t
            FROM shop.events WHERE toStartOfHour(ts::DateTime64(3)) = toDateTime('{h}')"""))
    events = pd.concat(parts, ignore_index=True)
    updates = query_df("""
        SELECT product_id::String AS product_id, change::String AS change, category::String AS category,
               subcategory::String AS subcategory, brand::String AS brand, price_eur::Float64 AS price_eur,
               is_active::Bool AS is_active, toString(created_at::DateTime64(3)) AS created_at,
               toString(ts::DateTime64(3)) AS t
        FROM shop.catalog_updates ORDER BY ts""")
    # The feedback result outgrew RawTree's 100 MB response limit: fetch it in 4 parts (hash of request_id).
    fb = pd.concat([query_df(f"""
        SELECT r.request_id AS request_id, r.ts AS ts, r.user_id AS user_id, r.is_new_user AS is_new_user,
               r.device AS device, r.region AS region, r.page AS page, r.recent_views AS recent_views,
               f.shown AS shown, f.clicked AS clicked, f.purchased AS purchased, f.fts AS fts,
               s.mv AS model_version
        FROM (SELECT request_id::String AS request_id, shown::Array(String) AS shown, clicked::String AS clicked,
                     purchased::String AS purchased, toString(ts::DateTime64(3)) AS fts
              FROM {TEAM}.feedback WHERE request_id::String != '__init__'
                AND cityHash64(request_id::String) % 4 = {k}) AS f
        INNER JOIN (SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts, user_id::String AS user_id,
                           is_new_user::Bool AS is_new_user, context.device::String AS device,
                           context.region::String AS region, context.page::String AS page, recent_views
                    FROM shop.requests) AS r ON r.request_id = f.request_id
        LEFT JOIN (SELECT request_id::String AS request_id, any(model_version::String) AS mv
                   FROM {TEAM}.submissions GROUP BY request_id) AS s ON s.request_id = f.request_id
        ORDER BY r.ts""") for k in range(4)], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    return events, updates, fb


def _epoch(t: str) -> float:
    return to_epoch(t) + (float("0." + t.split(".")[1]) if "." in t else 0.0)


def replay(store: Store, events: pd.DataFrame, updates: pd.DataFrame, fb: pd.DataFrame, hook=None):
    """Walk the day in time order and build (features, labels) for every request as seen at that moment.
    hook(k, row, ctx), if given, is called for every request with the point-in-time context (offline tests)."""
    rec = Recommender(store)
    ev = sorted(zip((_epoch(t) for t in events.t), events.u, events.p, events.e), key=lambda x: x[0])
    up = [(to_epoch(r["t"]), r) for r in updates.to_dict("records")]
    fb = fb.assign(rt=[_epoch(t) for t in fb.ts], ft=[_epoch(t) for t in fb.fts]).sort_values("rt")
    ref_by_sc: Counter = Counter()                       # usual purchases per subcategory (last week of history)
    for p, (_, _, b) in store.pop_raw.items():
        ref_by_sc[store.subcat.get(p)] += b
    buys_window: deque = deque()                         # (ts, subcategory) of live purchases in the trend window
    live_by_sc: Counter = Counter()
    arrived: list = []                                   # heap of (feedback_ts, user, clicked, bought)
    live_counts: dict[str, list[float]] = {}
    col = {"view": 0, "add_to_cart": 1, "purchase": 2}
    i_ev = i_up = 0
    next_pop = next_trend = 0.0
    X, y_buy, y_click = [], [], []
    t0 = time.time()
    for k, row in enumerate(fb.itertuples(index=False)):
        now = row.rt
        batch = []
        while i_ev < len(ev) and ev[i_ev][0] < now:
            ts, u, p, e = ev[i_ev]
            batch.append((u, p, e, 1, ts))
            live_counts.setdefault(p, [0.0, 0.0, 0.0])[col.get(e, 0)] += 1
            if e == "purchase":
                sc = store.subcat.get(p)
                buys_window.append((ts, sc))
                live_by_sc[sc] += 1
            i_ev += 1
        store.ingest(batch)
        while i_up < len(up) and up[i_up][0] < now:
            store.apply_catalog_update(up[i_up][1])
            i_up += 1
        while arrived and arrived[0][0] < now:          # our feedback becomes visible a few seconds later
            _, u, c = heapq.heappop(arrived)
            clicks = store.live_cs.setdefault(u, {})
            clicks[c] = clicks.get(c, 0.0) + 1
        if now >= next_pop:                              # live refreshes popularity every 5 min
            keys = set(store.pop_raw) | set(live_counts)
            df = pd.DataFrame([(p, *(np.add(store.pop_raw.get(p, [0, 0, 0]), live_counts.get(p, [0, 0, 0]))))
                               for p in keys], columns=["p", "v", "c", "b"])
            store.set_popularity(df)
            next_pop = now + 300
        if now >= next_trend:                            # live refreshes trends every minute
            while buys_window and buys_window[0][0] < now - TREND_HOURS * 3600:
                _, sc = buys_window.popleft()
                live_by_sc[sc] -= 1
            store.set_trends(dict(live_by_sc), dict(ref_by_sc))
            next_trend = now + 60
        ctx = rec.context(request_from_row(row._asdict()))
        if hook is not None:
            hook(k, row, ctx)
        shown = list(row.shown)
        X.append(rec.features(ctx, shown, positions=list(range(len(shown)))))
        y_buy.extend(p == row.purchased for p in shown)
        y_click.extend(p == row.clicked for p in shown)
        for p in shown:                                  # after the features: "times shown BEFORE this request"
            store.shown[row.user_id][p] += 1
        if row.clicked:
            heapq.heappush(arrived, (row.ft, row.user_id, row.clicked))
        if k % 20000 == 0:
            print(f"  replayed {k:,}/{len(fb):,} requests ({time.time() - t0:.0f}s)", flush=True)
    return np.vstack(X), np.array(y_buy, float), np.array(y_click, float)


PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=7,
              num_threads=3)   # leave cores to the live process


def fit(X, y, w=None, cols=None, params=PARAMS, rounds: int | None = None):
    """Train with early stopping on the latest 15% (time order), or a fixed number of rounds."""
    import lightgbm as lgb
    cols = list(range(X.shape[1])) if cols is None else cols
    names = [FEATURES_ALL[i] for i in cols]
    cat = [c for c in CATEGORICAL if c in names]
    Xc = X[:, cols]
    if rounds:
        return lgb.train(params, lgb.Dataset(Xc, y, weight=w, feature_name=names, categorical_feature=cat), rounds)
    cut = int(len(y) * 0.85)
    dtr = lgb.Dataset(Xc[:cut], y[:cut], weight=None if w is None else w[:cut], feature_name=names, categorical_feature=cat)
    dva = lgb.Dataset(Xc[cut:], y[cut:], weight=None if w is None else w[cut:], reference=dtr)
    return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def top1(scores: np.ndarray, y: np.ndarray, n_per: int = 10) -> float:
    """Of the requests with a purchase, how often the purchased product is ranked first among the 10 shown."""
    s, t = scores.reshape(-1, n_per), y.reshape(-1, n_per)
    has = t.sum(1) > 0
    return float((t[has, s[has].argmax(1)] == 1).mean()) if has.any() else float("nan")


# Training targets compared on every run. "click + 3x purchase": clicks give 3x more positives and teach what
# attracts the shopper, purchases (weighted 3x) what they actually buy -- the most stable choice so far.
TARGETS = {
    "click + 3x purchase": lambda b, c: (np.maximum(c, b), np.where(b == 1, 3.0, 1.0)),
    "purchase only": lambda b, c: (b, None),
}


V7_NEW = ["ses_subcat", "coview", "n_views"]


def segment_report(X: np.ndarray, yb: np.ndarray, yc: np.ndarray, scores: dict) -> pd.DataFrame:
    """AUC purchase and top-1 per shopper segment (cold = new user; no cold split by page)."""
    first = X[::10]
    new, page = first[:, FEATURES.index("is_new_user")] == 1, first[:, FEATURES.index("page")]
    segs = {"ALL": np.ones(len(first), bool), "cold (new user)": new,
            "no cold - product": ~new & (page == 1), "no cold - category": ~new & (page == 2),
            "no cold - home": ~new & (page == 0)}
    rows = []
    for seg, m in segs.items():
        rows_m = np.repeat(m, 10)
        row = {"segment": seg, "requests": int(m.sum()), "purchases": int(yb[rows_m].sum())}
        for name, s in scores.items():
            row[f"AUC {name}"] = auc(s[rows_m], yb[rows_m])
            row[f"top1 {name}"] = top1(s[rows_m], yb[rows_m])
        rows.append(row)
    return pd.DataFrame(rows).set_index("segment").round(3)


def main() -> None:
    be_nice()
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="train the best variant on all live rows and save it")
    ap.add_argument("--out", default="ranker_v7", help="model file name (without .txt)")
    args = ap.parse_args()

    print("history store (everything before the event) ...")
    store = cached("store_v7_all", lambda: Store.load(until=None))
    store.live_cs, store.shown = {}, defaultdict(Counter)
    print("live data ...")
    events, updates, fb = cached(f"live_{time.strftime('%Y%m%d%H')}", fetch_live)
    fb = fb[fb.shown.apply(len) == 10].reset_index(drop=True)
    print(f"  {len(events):,} events, {len(updates):,} catalog updates, {len(fb):,} requests with feedback, "
          f"{int((fb.purchased != '').sum())} purchases, {int((fb.clicked != '').sum())} clicks")
    X, yb, yc = replay(store, events, updates, fb)

    # ---------------------------------------------------------------- time split: last 25% of requests = test
    cut = int(len(yb) // 10 * 0.75) * 10
    Xtr, Xte = X[:cut], X[cut:].copy()
    Xte[:, FEATURES.index("pos")] = 0                    # rank as if every product were in slot 1
    print(f"\ntrain {cut // 10:,} requests ({int(yb[:cut].sum())} purchases, {int(yc[:cut].sum())} clicks) | "
          f"test {(len(yb) - cut) // 10:,} requests ({int(yb[cut:].sum())} purchases)")
    best = "click + 3x purchase"                       # the best target in every run so far
    y, w = TARGETS[best](yb[:cut], yc[:cut])
    feature_sets = {"v6 features": [i for i, f in enumerate(FEATURES) if f not in V7_NEW], "v7 features": None}
    trained, scores = {}, {}
    for name, cols in feature_sets.items():
        trained[name] = fit(Xtr, y, w, cols=cols)
        scores[name] = trained[name].predict(Xte if cols is None else Xte[:, cols])
    print(segment_report(Xte, yb[cut:], yc[cut:], scores).to_string())
    trained = {best: trained["v7 features"]}
    imp = pd.Series(trained[best].feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print(f"\nbest: {best}; feature importance %:")
    print((100 * imp / imp.sum()).round(1).to_string())

    if args.final:                                   # same target and number of trees, on ALL live requests
        y, w = TARGETS[best](yb, yc)
        final = fit(X, y, w, rounds=max(trained[best].best_iteration, 50))
        path = MODELS / f"{args.out}.txt"
        final.save_model(str(path.with_suffix(".tmp")))
        os.replace(path.with_suffix(".tmp"), path)  # atomic: the live process reloads when this file changes
        print(f"saved {path.name} ({len(yb) // 10:,} live requests)")


if __name__ == "__main__":
    main()
