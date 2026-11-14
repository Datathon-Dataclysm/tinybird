"""Learn from the live stream: train the ranker on what the live simulator actually did with our lists.

    python starter/train_live.py              # replay the day, train variants, compare them on the latest requests
    python starter/train_live.py --final      # same, then train the best variant on ALL live rows -> models/ranker_v4*.txt

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
from model import CATEGORICAL, FEATURES, TREND_HOURS, Recommender, Store, request_from_row  # noqa: E402
from rawtree import ROOT, TEAM, query_df, to_epoch  # noqa: E402
from train import auc, cached  # noqa: E402

MODELS = ROOT / "models"
ATTR_FEATURES = ["attr_lift_mean", "attr_lift_min", "attr_lift_max", "attr_views_match"]   # added in v5
NO_ATTR = [i for i, f in enumerate(FEATURES) if f not in ATTR_FEATURES]    # the v4 feature set


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
    fb = query_df(f"""
        SELECT r.request_id AS request_id, r.ts AS ts, r.user_id AS user_id, r.is_new_user AS is_new_user,
               r.device AS device, r.region AS region, r.page AS page, r.recent_views AS recent_views,
               f.shown AS shown, f.clicked AS clicked, f.purchased AS purchased, f.fts AS fts,
               s.mv AS model_version
        FROM (SELECT request_id::String AS request_id, shown::Array(String) AS shown, clicked::String AS clicked,
                     purchased::String AS purchased, toString(ts::DateTime64(3)) AS fts
              FROM {TEAM}.feedback WHERE request_id::String != '__init__') AS f
        INNER JOIN (SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts, user_id::String AS user_id,
                           is_new_user::Bool AS is_new_user, context.device::String AS device,
                           context.region::String AS region, context.page::String AS page, recent_views
                    FROM shop.requests) AS r ON r.request_id = f.request_id
        LEFT JOIN (SELECT request_id::String AS request_id, any(model_version::String) AS mv
                   FROM {TEAM}.submissions GROUP BY request_id) AS s ON s.request_id = f.request_id
        ORDER BY r.ts""")
    return events, updates, fb


def _epoch(t: str) -> float:
    return to_epoch(t) + (float("0." + t.split(".")[1]) if "." in t else 0.0)


def replay(store: Store, events: pd.DataFrame, updates: pd.DataFrame, fb: pd.DataFrame):
    """Walk the day in time order and build (features, labels) for every request as seen at that moment."""
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
            _, u, c, b = heapq.heappop(arrived)
            row_cs = store.live_cs.setdefault(u, {}).setdefault(c, [0.0, 0.0])
            row_cs[0] += 1
            row_cs[1] += b
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
        shown = list(row.shown)
        X.append(rec.features(ctx, shown, positions=list(range(len(shown)))))
        y_buy.extend(p == row.purchased for p in shown)
        y_click.extend(p == row.clicked for p in shown)
        for p in shown:                                  # after the features: "times shown BEFORE this request"
            store.shown[row.user_id][p] += 1
        if row.clicked:
            heapq.heappush(arrived, (row.ft, row.user_id, row.clicked, float(row.purchased == row.clicked)))
        if k % 20000 == 0:
            print(f"  replayed {k:,}/{len(fb):,} requests ({time.time() - t0:.0f}s)", flush=True)
    return np.vstack(X), np.array(y_buy, float), np.array(y_click, float)


PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=7)
PARAMS_SMALL = dict(PARAMS, num_leaves=7, min_data_in_leaf=40, lambda_l2=10.0)   # buy | click: few rows


def fit(X, y, w=None, cols=None, params=PARAMS, rounds: int | None = None):
    """Train with early stopping on the latest 15% (time order), or a fixed number of rounds."""
    import lightgbm as lgb
    cols = list(range(X.shape[1])) if cols is None else cols
    names = [FEATURES[i] for i in cols]
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


def variants(X, yb, yc):
    """Every candidate way of scoring: name -> (train function, predict function)."""
    mix_w = lambda y: np.where(y == 1, 3.0, 1.0)  # noqa: E731

    def v4(Xt, b, c):
        m = fit(Xt, b, cols=NO_ATTR)
        return [m], lambda Z: m.predict(Z[:, NO_ATTR])

    def mix_new(Xt, b, c):
        m = fit(Xt, np.maximum(c, b), mix_w(b))
        return [m], lambda Z: m.predict(Z)

    def purchase(Xt, b, c):
        m = fit(Xt, b)
        return [m], lambda Z: m.predict(Z)

    def two_stage(Xt, b, c):
        mc = fit(Xt, c)
        clicked = c == 1
        mb = fit(Xt[clicked], b[clicked], params=PARAMS_SMALL)
        return [mc, mb], lambda Z: mc.predict(Z) * mb.predict(Z)

    return {"v4-raven (purchase, no attributes)": v4, "v5: purchase + attributes": purchase,
            "v5: click+3x buy + attributes": mix_new, "v5: two-stage + attributes": two_stage}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="train the best variant on all live rows and save it")
    ap.add_argument("--out", default="ranker_v5", help="model file name (without .txt)")
    args = ap.parse_args()

    print("history store (everything before the event) ...")
    store = cached("store_v3_all", lambda: Store.load(until=None))
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
    results, trained = {}, {}
    for name, train_fn in variants(X, yb, yc).items():
        models, predict = train_fn(Xtr, yb[:cut], yc[:cut])
        s = predict(Xte)
        trained[name] = (train_fn, models)
        results[name] = {"AUC purchase": auc(s, yb[cut:]), "AUC click": auc(s, yc[cut:]),
                         "top1 purchase": top1(s, yb[cut:])}
    table = pd.DataFrame(results).T
    print(table.round(3).to_string())

    best = table["AUC purchase"].idxmax()
    print(f"\nbest: {best}")
    first = trained[best][1][0]
    imp = pd.Series(first.feature_importance("gain"), index=first.feature_name()).sort_values(ascending=False)
    print("feature importance % (first model of the best variant):")
    print((100 * imp / imp.sum()).round(1).head(15).to_string())

    if args.final:
        save_final(best, X, yb, yc, trained[best][1], args.out)


def save_final(best: str, X, yb, yc, models, out: str) -> None:
    """Retrain the chosen variant on ALL live rows (same number of trees) and write it atomically."""
    main_path, buy_path = MODELS / f"{out}.txt", MODELS / f"{out}_buy.txt"
    if "two-stage" in best:
        mc = fit(X, yc, rounds=max(models[0].best_iteration, 50))
        clicked = yc == 1
        mb = fit(X[clicked], yb[clicked], params=PARAMS_SMALL, rounds=max(models[1].best_iteration, 30))
        mb.save_model(str(buy_path.with_suffix(".tmp")))
        os.replace(buy_path.with_suffix(".tmp"), buy_path)
        final = mc
    else:
        y = np.maximum(yc, yb) if "click+3x" in best else yb
        w = np.where(yb == 1, 3.0, 1.0) if "click+3x" in best else None
        final = fit(X, y, w, rounds=max(models[0].best_iteration, 50))
        if buy_path.exists():
            buy_path.unlink()
    final.save_model(str(main_path.with_suffix(".tmp")))
    os.replace(main_path.with_suffix(".tmp"), main_path)   # the live process reloads when this file changes
    print(f"saved {main_path.name}" + (f" + {buy_path.name}" if buy_path.exists() else "") +
          f" ({len(yb) // 10:,} live requests)")


if __name__ == "__main__":
    main()
