"""Learn from the live stream: train the ranker on what the live simulator actually did with our lists.

    python starter/train_live.py              # replay the day, train, compare with the history model
    python starter/train_live.py --final      # same, then train on ALL live rows -> models/ranker_live.txt

Why: the live shoppers do not behave like the history said (e.g. a product the shopper already bought is almost
never bought again live, while history suggested the opposite). team_NN.feedback tells, for every request,
the 10 products shown and what was clicked / bought: that is labelled training data from the real simulator.

How (no peeking into the future): we *replay* the day in time order. The store starts with the full history; then
shop.events, shop.catalog_updates and our own feedback are applied minute by minute, and every request is turned into
training rows with the features the live system would have computed at that very moment.
"""
from __future__ import annotations

import argparse
import heapq
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CATEGORICAL, FEATURES, Ranker, Recommender, Store, request_from_row  # noqa: E402
from rawtree import ROOT, TEAM, query_df, to_epoch  # noqa: E402
from train import auc, cached  # noqa: E402

MODELS = ROOT / "models"


def fetch_live() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """All live events, catalog updates and our feedback (joined with the request) so far."""
    hours = query_df("SELECT DISTINCT toString(toStartOfHour(ts::DateTime64(3))) AS h FROM shop.events ORDER BY h").h
    parts = []
    for h in hours:
        parts.append(query_df(f"""
            SELECT user_id::String AS u, product_id::String AS p, event_type::String AS e, toString(ts::DateTime64(3)) AS t
            FROM shop.events WHERE toStartOfHour(ts::DateTime64(3)) = toDateTime('{h}')"""))
        print(f"  events {h}: {len(parts[-1]):,}", flush=True)
    events = pd.concat(parts, ignore_index=True)
    updates = query_df("""
        SELECT product_id::String AS product_id, change::String AS change, category::String AS category,
               subcategory::String AS subcategory, brand::String AS brand, price_eur::Float64 AS price_eur,
               is_active::Bool AS is_active, toString(created_at::DateTime64(3)) AS created_at,
               toString(ts::DateTime64(3)) AS t
        FROM shop.catalog_updates ORDER BY ts""")
    fb = query_df(f"""
        SELECT r.request_id AS request_id, r.ts AS ts, r.user_id AS user_id, r.is_new_user AS is_new_user,
               r.device AS device, r.region AS region, r.page AS page, r.recent_views AS recent_views,
               f.shown AS shown, f.clicked AS clicked, f.purchased AS purchased, f.fts AS fts, f.served_by AS served_by,
               s.mv AS model_version
        FROM (SELECT request_id::String AS request_id, shown::Array(String) AS shown, clicked::String AS clicked,
                     purchased::String AS purchased, toString(ts::DateTime64(3)) AS fts, served_by::String AS served_by
              FROM {TEAM}.feedback WHERE request_id::String != '__init__') AS f
        INNER JOIN (SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts, user_id::String AS user_id,
                           is_new_user::Bool AS is_new_user, context.device::String AS device,
                           context.region::String AS region, context.page::String AS page, recent_views
                    FROM shop.requests) AS r ON r.request_id = f.request_id
        LEFT JOIN (SELECT request_id::String AS request_id, any(model_version::String) AS mv
                   FROM {TEAM}.submissions GROUP BY request_id) AS s ON s.request_id = f.request_id
        ORDER BY r.ts""")
    return events, updates, fb


def replay(store: Store, events: pd.DataFrame, updates: pd.DataFrame, fb: pd.DataFrame):
    """Walk the day in time order and build (features, labels) for every request as seen at that moment."""
    rec = Recommender(store, adjust=False)
    ev_t = np.array([to_epoch(t) + float("0." + t.split(".")[1]) if "." in t else to_epoch(t) for t in events.t])
    ev = list(zip(ev_t, events.u, events.p, events.e))
    ev.sort(key=lambda x: x[0])
    up = [(to_epoch(r["t"]), r) for r in updates.to_dict("records")]
    fb = fb.assign(rt=[to_epoch(t) for t in fb.ts], ft=[to_epoch(t) for t in fb.fts]).sort_values("rt")
    arrived: list = []                       # heap of (feedback_ts, user, clicked, bought): our carousel, as it arrives
    live_counts: dict[str, list[float]] = {}
    i_ev = i_up = 0
    next_pop = 0.0
    X, y_buy, y_click, t_req, version, new_user = [], [], [], [], [], []
    col = {"view": 0, "add_to_cart": 1, "purchase": 2}
    t0 = time.time()
    for k, row in enumerate(fb.itertuples(index=False)):
        now = row.rt
        batch = []
        while i_ev < len(ev) and ev[i_ev][0] < now:
            ts, u, p, e = ev[i_ev]
            batch.append((u, p, e, 1, ts))
            live_counts.setdefault(p, [0.0, 0.0, 0.0])[col.get(e, 0)] += 1
            i_ev += 1
        store.ingest(batch)
        while i_up < len(up) and up[i_up][0] < now:
            store.apply_catalog_update(up[i_up][1])
            i_up += 1
        while arrived and arrived[0][0] < now:
            _, u, c, b = heapq.heappop(arrived)
            row_cs = store.live_cs.setdefault(u, {}).setdefault(c, [0.0, 0.0])
            row_cs[0] += 1
            row_cs[1] += b
        if now >= next_pop:                  # live refreshes popularity every 5 min: same here
            keys = set(store.pop_raw) | set(live_counts)
            df = pd.DataFrame([(p, *(np.add(store.pop_raw.get(p, [0, 0, 0]), live_counts.get(p, [0, 0, 0]))))
                               for p in keys], columns=["p", "v", "c", "b"])
            store.set_popularity(df)
            next_pop = now + 300
        req = request_from_row(row._asdict())
        ctx = rec.context(req)
        shown = list(row.shown)
        X.append(rec.features(ctx, shown, positions=list(range(len(shown)))))
        y_buy.extend(p == row.purchased for p in shown)
        y_click.extend(p == row.clicked for p in shown)
        t_req.extend([now] * len(shown))
        version.extend([row.model_version or "(fallback)"] * len(shown))
        new_user.extend([bool(row.is_new_user)] * len(shown))
        if row.clicked:
            heapq.heappush(arrived, (row.ft, row.user_id, row.clicked, float(row.purchased == row.clicked)))
        if k % 10000 == 0:
            print(f"  replayed {k:,}/{len(fb):,} requests ({time.time() - t0:.0f}s)", flush=True)
    return (np.vstack(X), np.array(y_buy, float), np.array(y_click, float), np.array(t_req),
            np.array(version), np.array(new_user))


def fit(X, y, w=None, seed: int = 7, rounds: int | None = None):
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=300,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=seed)
    d = lgb.Dataset(X, y, weight=w, feature_name=FEATURES, categorical_feature=CATEGORICAL)
    if rounds:
        return lgb.train(params, d, num_boost_round=rounds)
    n = len(y)
    cut = int(n * 0.85)
    dtr = lgb.Dataset(X[:cut], y[:cut], weight=None if w is None else w[:cut], feature_name=FEATURES,
                      categorical_feature=CATEGORICAL)
    dva = lgb.Dataset(X[cut:], y[cut:], weight=None if w is None else w[cut:], reference=dtr)
    return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


def top1(scores: np.ndarray, y: np.ndarray, n_per: int = 10) -> float:
    """Of the requests with a purchase, how often the model puts the purchased product first among the 10 shown."""
    s, t = scores.reshape(-1, n_per), y.reshape(-1, n_per)
    has = t.sum(1) > 0
    return float((t[has, s[has].argmax(1)] == 1).mean()) if has.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="after evaluating, train on all live rows and save it")
    ap.add_argument("--refresh", action="store_true", help="re-download the live data (default: cached per hour)")
    args = ap.parse_args()

    print("history store (everything before the event) ...")
    store = cached("store_v2_all", lambda: Store.load(until=None))
    store.live_cs = {}
    tag = time.strftime("%Y%m%d%H")
    print("live data ...")
    events, updates, fb = fetch_live() if args.refresh else cached(f"live_{tag}", fetch_live)
    fb = fb[fb.shown.apply(len) == 10].reset_index(drop=True)
    print(f"  {len(events):,} events, {len(updates):,} catalog updates, {len(fb):,} requests with feedback, "
          f"{int((fb.purchased != '').sum())} purchases, {int((fb.clicked != '').sum())} clicks")
    X, yb, yc, t, ver, new = replay(store, events, updates, fb)

    # ---------------------------------------------------------------- time split: last 25% of requests = test
    n_req = len(yb) // 10
    cut = (int(n_req * 0.75)) * 10
    tr, te = slice(0, cut), slice(cut, None)
    print(f"\ntrain {cut // 10:,} requests ({int(yb[tr].sum())} purchases) | test {(len(yb) - cut) // 10:,} requests "
          f"({int(yb[te].sum())} purchases)")
    Xte = X[te].copy()
    Xte[:, FEATURES.index("pos")] = 0                 # rank as if every product were in slot 1

    hist = Ranker.load(MODELS / "ranker_v2.txt")
    models = {"history model (v2)": hist.score(Xte)}
    b_buy = fit(X[tr], yb[tr])
    models["live: purchase"] = b_buy.predict(Xte)
    b_click = fit(X[tr], yc[tr])
    models["live: click"] = b_click.predict(Xte)
    w = np.where(yb[tr] == 1, 3.0, 1.0)                # clicks, purchases count 3x (purchases are what scores)
    b_mix = fit(X[tr], np.maximum(yc[tr], yb[tr]), w)
    models["live: click+3x purchase"] = b_mix.predict(Xte)
    models["slot order we showed"] = -X[te][:, FEATURES.index("pos")]

    rows = {}
    for name, s in models.items():
        rows[name] = {"AUC purchase": auc(s, yb[te]), "AUC click": auc(s, yc[te]), "top1 purchase": top1(s, yb[te])}
    print(pd.DataFrame(rows).T.round(3).to_string())
    imp = pd.Series(b_mix.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("\nlive model (click+3x purchase), feature importance %:")
    print((100 * imp / imp.sum()).round(1).head(15).to_string())

    if args.final:
        best = b_mix
        final = fit(X, np.maximum(yc, yb), np.where(yb == 1, 3.0, 1.0), rounds=max(best.best_iteration, 50))
        tmp = MODELS / "ranker_live.tmp"
        final.save_model(str(tmp))
        os.replace(tmp, MODELS / "ranker_live.txt")     # atomic: the live process never reads a half-written file
        print(f"\nsaved models/ranker_live.txt ({final.num_trees()} trees, all {len(yb) // 10:,} live requests)")


if __name__ == "__main__":
    main()
