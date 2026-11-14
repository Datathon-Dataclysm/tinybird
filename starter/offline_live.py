"""Offline workbench on the live data: compare features and candidate generators WITHOUT touching production.

    python starter/offline_live.py ablation     # which feature groups matter? (replayed live feedback, AUC purchase)
    python starter/offline_live.py recall       # do our candidates contain what shoppers actually buy?
    python starter/offline_live.py carousel     # same, for the purchases our carousel actually made (v6 vs v7 rules)
    python starter/offline_live.py e2e          # full pipeline per version: real purchases in our top 10 / top 3

ablation: the day is replayed once (point-in-time features, see train_live.py) and cached; then the ranker is retrained
          without each feature group and evaluated on the latest 25% of requests.
recall:   for organic purchases in shop.events (the same shoppers' real choices), rebuild the request as it was just
          before the purchase (their session views, minus the product itself) and check whether each candidate
          generator would have proposed the purchased product. A product that is not a candidate can never be shown.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collections import Counter, defaultdict

from model import FEATURES, Ranker, Recommender, Store  # noqa: E402
from rawtree import ROOT, be_nice, query_df, to_epoch  # noqa: E402
from train import auc, cached  # noqa: E402
from train_live import fetch_live, fit, replay, top1  # noqa: E402

GROUPS = {
    "session": ["in_views", "ses_cat", "ses_subcat", "ses_attr", "attr_views_match", "coview", "n_views"],
    "user x product": ["ui_buys", "ui_carts", "days_since_buy", "days_since_cart"],
    "carousel": ["cs_clicks"],
    "taste (affinity)": ["aff_cat", "aff_subcat", "aff_attr", "aff_attr_min", "attr_lift_mean"],
    "popularity": ["pop_views", "pop_buys", "cvr", "cart_rate"],
    "freshness / trend / fatigue": ["age_days", "trend", "times_shown"],
    "context": ["page", "device", "region", "is_new_user"],
}


def history_store() -> Store:
    return cached("store_v7_all", lambda: Store.load(until=None))


def replayed():
    """Replay the day once (cached per hour): X, y_buy, y_click."""
    def build():
        store = history_store()
        events, updates, fb = cached(f"live_{time.strftime('%Y%m%d%H')}", fetch_live)
        fb = fb[fb.shown.apply(len) == 10].reset_index(drop=True)
        print(f"  {len(fb):,} requests, {int((fb.purchased != '').sum())} purchases")
        X, yb, yc = replay(store, events, updates, fb)
        return X.astype(np.float32), yb, yc
    return cached(f"replay_v7_{time.strftime('%Y%m%d%H')}", build)


def evaluate(cols, Xtr, ytr_b, ytr_c, Xte, yte_b, yte_c) -> dict:
    y = np.maximum(ytr_c, ytr_b)
    m = fit(Xtr, y, np.where(ytr_b == 1, 3.0, 1.0), cols=cols)
    s = m.predict(Xte[:, cols])
    return {"AUC purchase": auc(s, yte_b), "AUC click": auc(s, yte_c), "top1": top1(s, yte_b),
            "features": len(cols)}


def ablation(extra_sets: dict[str, list[str]] | None = None) -> None:
    X, yb, yc = replayed()
    cut = int(len(yb) // 10 * 0.75) * 10
    Xtr, Xte = X[:cut], X[cut:].copy()
    Xte[:, FEATURES.index("pos")] = 0
    print(f"train {cut // 10:,} requests | test {(len(yb) - cut) // 10:,} requests, {int(yb[cut:].sum())} purchases")
    idx = {f: i for i, f in enumerate(FEATURES)}
    all_cols = list(range(len(FEATURES)))
    rows = {"ALL features": evaluate(all_cols, Xtr, yb[:cut], yc[:cut], Xte, yb[cut:], yc[cut:])}
    for g, feats in GROUPS.items():
        cols = [i for i in all_cols if FEATURES[i] not in feats]
        rows[f"without {g}"] = evaluate(cols, Xtr, yb[:cut], yc[:cut], Xte, yb[cut:], yc[cut:])
        print(f"  done: without {g}", flush=True)
    for name, feats in (extra_sets or {}).items():
        cols = [idx[f] for f in feats]
        rows[name] = evaluate(cols, Xtr, yb[:cut], yc[:cut], Xte, yb[cut:], yc[cut:])
    table = pd.DataFrame(rows).T
    table["delta AUC"] = table["AUC purchase"] - table.loc["ALL features", "AUC purchase"]
    print(table.round(4).to_string())


def recall(n: int = 3000) -> None:
    """Organic purchases of the last hours: would each generator have proposed the purchased product?"""
    store = history_store()
    buys = query_df(f"""
        SELECT user_id::String AS u, session_id::String AS sid, product_id::String AS p, toString(ts::DateTime64(3)) AS t
        FROM shop.events WHERE event_type::String = 'purchase' AND ts::DateTime64(3) > now64(3) - INTERVAL 3 HOUR
        ORDER BY cityHash64(sid, p) LIMIT {n}""")
    sids = ",".join(f"'{x}'" for x in buys.sid.unique())
    views = query_df(f"""
        SELECT session_id::String AS sid, product_id::String AS p, toString(ts::DateTime64(3)) AS t
        FROM shop.events WHERE event_type::String = 'view' AND session_id::String IN ({sids}) ORDER BY t""")
    by_sid = {k: g for k, g in views.groupby("sid")}
    gens = {   # v5 (hand-made rules) measured 44.2% recall with 105 candidates on the same test
        "v6 (retrieve 100)": Recommender(store, n_retrieve=100),
        "v6 (retrieve 150, default)": Recommender(store),
        "v6 (retrieve 250)": Recommender(store, n_retrieve=250, n_candidates=320),
    }
    hits = {g: 0 for g in gens}
    sizes = {g: 0 for g in gens}
    ranks = []
    has_hist = repurchases = 0
    for r in buys.itertuples(index=False):
        g = by_sid.get(r.sid)
        v = [] if g is None else [p for p, t in zip(g.p, g.t) if t < r.t and p != r.p][-5:]
        req = {"user_id": r.u, "ts": r.t, "recent_views": [{"product_id": p} for p in v], "page": "product"}
        has_hist += r.u in store.us_index
        if r.p in next(iter(gens.values())).context(req)["bought"]:
            repurchases += 1           # a re-purchase: never bought again in the carousel, we exclude these on purpose
            continue
        for name, rec in gens.items():
            ctx = rec.context(req)
            c = rec.candidates(ctx)
            hits[name] += r.p in c
            sizes[name] += len(c)
        full = gens["v6 (retrieve 150, default)"].retrieve(gens["v6 (retrieve 150, default)"].context(req), 6000)
        ranks.append(full.index(r.p) if r.p in full else len(full))
    n = len(buys) - repurchases
    print(f"{len(buys)} organic purchases, {has_hist} by users with history, {repurchases} re-purchases left out, {n} evaluated")
    for name in gens:
        print(f"  {name:28s} recall {100 * hits[name] / n:5.1f}%   avg pool {sizes[name] / n:5.0f}")
    ranks = np.array(ranks)
    print(f"  rank of the purchased product by affinity over the whole catalogue: median {np.median(ranks):.0f}, "
          f"top-50 {100 * (ranks < 50).mean():.1f}%, top-150 {100 * (ranks < 150).mean():.1f}%, "
          f"top-500 {100 * (ranks < 500).mean():.1f}%")


def organic_requests(store: Store, n: int = 3000) -> list[tuple[dict, str]]:
    """Organic purchases of the last 3 hours as (request just before the purchase, purchased product)."""
    buys = query_df(f"""
        SELECT user_id::String AS u, session_id::String AS sid, product_id::String AS p, toString(ts::DateTime64(3)) AS t
        FROM shop.events WHERE event_type::String = 'purchase' AND ts::DateTime64(3) > now64(3) - INTERVAL 3 HOUR
        ORDER BY cityHash64(sid, p) LIMIT {n}""")
    sids = ",".join(f"'{x}'" for x in buys.sid.unique())
    views = query_df(f"""
        SELECT session_id::String AS sid, product_id::String AS p, toString(ts::DateTime64(3)) AS t
        FROM shop.events WHERE event_type::String = 'view' AND session_id::String IN ({sids}) ORDER BY t""")
    by_sid = {k: g for k, g in views.groupby("sid")}
    out = []
    for r in buys.itertuples(index=False):
        g = by_sid.get(r.sid)
        v = [] if g is None else [p for p, t in zip(g.p, g.t) if t < r.t and p != r.p][-5:]
        out.append(({"user_id": r.u, "ts": r.t, "recent_views": [{"product_id": p} for p in v], "page": "product"}, r.p))
    return out


def tune(k: int = 150) -> None:
    """Grid-search the weights of the retrieval score to maximise recall@k on organic purchases."""
    import itertools
    import model as M
    store = history_store()
    reqs = cached(f"organic_{time.strftime('%Y%m%d%H')}", lambda: organic_requests(store))
    ix = store.item_index()
    pos = {p: i for i, p in enumerate(ix["items"])}
    reqs = [(r, p) for r, p in reqs if p in pos]
    target = np.array([pos[p] for _, p in reqs])
    logpop = ix["logpop"].astype(np.float32)
    results = []
    for alpha in (10.0,):
        M.AFF_ALPHA = alpha
        rec = Recommender(store)
        ctxs = [rec.context(r) for r, _ in reqs]

        def stack(key, part):
            return np.stack([c[key][part] for c in ctxs]).astype(np.float32)
        U = {g: stack("aff", g) for g in ("cat", "sc", "attr")}
        S = {g: stack("ses", g) for g in ("cat", "sc", "attr")}
        comp = {
            "cat": U["cat"][:, ix["cat"]], "sc": U["sc"][:, ix["sc"]] - U["cat"][:, ix["cat"]],
            "attr": np.where(ix["a1"] >= 0, U["attr"][:, ix["a1"]], 0) + np.where(ix["a2"] >= 0, U["attr"][:, ix["a2"]], 0),
            "ses": S["cat"][:, ix["cat"]] + 0.5 * (S["sc"][:, ix["sc"]] - S["cat"][:, ix["cat"]])
                   + np.where(ix["a1"] >= 0, S["attr"][:, ix["a1"]], 0) + np.where(ix["a2"] >= 0, S["attr"][:, ix["a2"]], 0),
        }
        rows = np.arange(len(reqs))
        for wsc, wbr, wattr, wses, wpop in itertools.product((0.0, 0.25, 0.5), (0.0,), (0.75, 1.0, 1.25),
                                                              (0.25, 0.5, 0.75), (0.6, 0.9, 1.2, 1.6)):
            score = comp["cat"] + wsc * comp["sc"] + wattr * comp["attr"] + wses * comp["ses"] + wpop * logpop
            rank = (score > score[rows, target][:, None]).sum(1)
            results.append({"alpha": alpha, "w_subcat": wsc, "w_brand": wbr, "w_attr": wattr, "w_session": wses,
                            "w_pop": wpop, f"recall@{k}": float((rank < k).mean()), "recall@50": float((rank < 50).mean()),
                            "median rank": float(np.median(rank))})
        print(f"  alpha {alpha} done", flush=True)
    df = pd.DataFrame(results).sort_values(f"recall@{k}", ascending=False)
    cur = df[(df.alpha == 10.0) & (df.w_subcat == 0.5) & (df.w_brand == 0.5) & (df.w_attr == 1.0) & (df.w_session == 1.0) & (df.w_pop == 0.3)]
    print(f"{len(reqs)} organic purchases\ncurrent weights:\n{cur.round(3).to_string(index=False)}\nbest 10:")
    print(df.head(10).round(3).to_string(index=False))


def carousel_recall(n: int = 3000) -> None:
    """Products our carousel actually SOLD (team feedback): would each generator have proposed them?
    Rebuilt from the request (views, user, page) with the history store, so it measures the candidate rules."""
    from rawtree import TEAM
    store = history_store()
    df = query_df(f"""
        SELECT r.request_id AS request_id, r.ts AS ts, r.user_id AS user_id, r.is_new_user AS is_new_user,
               r.page AS page, r.recent_views AS recent_views, f.purchased AS purchased, s.mv AS mv
        FROM (SELECT request_id::String AS request_id, purchased::String AS purchased FROM {TEAM}.feedback
              WHERE purchased::String != '') AS f
        INNER JOIN (SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts, user_id::String AS user_id,
                           is_new_user::Bool AS is_new_user, context.page::String AS page, recent_views FROM shop.requests) AS r
            ON r.request_id = f.request_id
        LEFT JOIN (SELECT request_id::String AS request_id, any(model_version::String) AS mv FROM {TEAM}.submissions
                   GROUP BY request_id) AS s ON s.request_id = f.request_id
        ORDER BY cityHash64(r.request_id) LIMIT {n}""")
    full = Recommender(store)
    v6 = Recommender(store)
    v6.candidates = lambda ctx: _v6_candidates(v6, ctx)
    gens = {"v6 (taste only)": v6, "v7 (session + taste)": full}
    rows = []
    for r in df.to_dict("records"):
        ctx = full.context(r)
        seg = "new" if r["is_new_user"] else ("home" if r["page"] == "home" else "returning, with views")
        rows.append({"segment": seg, **{g: r["purchased"] in rec.candidates(ctx) for g, rec in gens.items()},
                     **{f"{g} pool": len(rec.candidates(ctx)) for g, rec in gens.items()}})
    t = pd.DataFrame(rows)
    out = t.groupby("segment").agg(**{"purchases": ("segment", "size")},
                                    **{g: (g, "mean") for g in gens}, **{f"{g} pool": (f"{g} pool", "mean") for g in gens})
    out.loc["ALL"] = [len(t)] + [t[g].mean() for g in gens] + [t[f"{g} pool"].mean() for g in gens]
    print(f"{len(t)} carousel purchases: share that each generator would have proposed")
    print(out.round(3).to_string())


def _v6_candidates(rec: Recommender, ctx) -> list[str]:
    """The v6 candidate rules (taste family only), kept to measure the v7 fix."""
    s, out, bought = rec.store, {}, ctx["bought"]

    def add(items, limit):
        n = 0
        for p in items:
            if n >= limit:
                break
            if p in s.subcat and p not in s.unavailable and p not in bought:
                out.setdefault(p)
                n += 1
    ui, cs = ctx["ui"], ctx["cs"]
    add(reversed(ctx["views"]), 20)
    add(sorted((p for p in ui if ui[p][0] > 0), key=lambda p: -ui[p][2]), 15)
    add(sorted(cs, key=lambda p: -cs[p]), 8)
    add(rec.retrieve(ctx, 250), 250)
    add(rec._new_products(ctx["now"]), 5)
    add(s.top_global, 15)
    return list(out)


class _ColRanker(Ranker):
    """A booster trained on a subset of FEATURES (to score an older feature set with today's feature matrix)."""
    def __init__(self, booster, cols) -> None:
        super().__init__(booster)
        self.cols = cols

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X[:, self.cols])


def end_to_end() -> None:
    """Full pipeline (candidates + ranker) on the LAST 25% of the day: for every real carousel purchase, would each
    version have put the purchased product in its top 10? Models are trained on the first 75% only, and the store
    is replayed point-in-time (live events, new products, prices). Also: on no-cold category pages, the share of the
    top 10 in the subcategory being browsed (the v6 symptom)."""
    from train_live import TARGETS, V7_NEW
    X, yb, yc = replayed()
    cut = int(len(yb) // 10 * 0.75) * 10
    y, w = TARGETS["click + 3x purchase"](yb[:cut], yc[:cut])
    v6cols = [i for i, f in enumerate(FEATURES) if f not in V7_NEW]
    print("training the v6-feature and v7-feature rankers on the first 75% ...")
    m6, m7 = fit(X[:cut], y, w, cols=v6cols), fit(X[:cut], y, w)
    store = history_store()
    store.live_cs, store.shown = {}, defaultdict(Counter)
    rec = Recommender(store)
    pipes = {"v6 (cands v6 + model v6)": ("v6", _ColRanker(m6, v6cols)),
             "v7 cands + model v6": ("v7", _ColRanker(m6, v6cols)),
             "v7 (cands v7 + model v7)": ("v7", _ColRanker(m7, list(range(len(FEATURES)))))}
    first_test = cut // 10
    hits, mix = [], []

    def top10(ctx, kind, ranker):
        cands = _v6_candidates(rec, ctx) if kind == "v6" else rec.candidates(ctx)
        scores = ranker.score(rec.features(ctx, cands))
        return rec.fill([cands[i] for i in np.argsort(-scores, kind="stable")], 10, ctx["bought"])

    def hook(k, row, ctx):
        if k < first_test:
            return
        seg = ("cold" if row.is_new_user else f"no cold - {row.page}")
        if row.purchased:
            out = {"segment": seg}
            for name, (kind, rk) in pipes.items():
                top = top10(ctx, kind, rk)
                out[name] = (top.index(row.purchased) + 1) if row.purchased in top else 0
            hits.append(out)
        elif seg == "no cold - category" and k % 5 == 0 and ctx["views"]:
            vsc = {store.subcat.get(p) for p in ctx["views"]}
            out = {}
            for name, (kind, rk) in pipes.items():
                top = top10(ctx, kind, rk)
                out[name] = np.mean([store.subcat.get(p) in vsc for p in top])
            mix.append(out)

    events, updates, fb = cached(f"live_{time.strftime('%Y%m%d%H')}", fetch_live)
    fb = fb[fb.shown.apply(len) == 10].reset_index(drop=True)
    print("replaying the day point-in-time ...")
    replay(store, events, updates, fb, hook=hook)
    h = pd.DataFrame(hits)
    names = list(pipes)
    res = h.groupby("segment").agg(purchases=("segment", "size"),
                                   **{f"{n} | hit@10": (n, lambda s: (s > 0).mean()) for n in names},
                                   **{f"{n} | hit@3": (n, lambda s: s.between(1, 3).mean()) for n in names})
    res.loc["ALL"] = [len(h)] + [(h[n] > 0).mean() for n in names] + [h[n].between(1, 3).mean() for n in names]
    print()
    print(f"{len(h)} carousel purchases in the last 25% of the day: share each pipeline would have put in its top 10 / 3")
    for n in names:
        print()
        print(n)
        print(res[["purchases", f"{n} | hit@10", f"{n} | hit@3"]].round(3).to_string())
    print()
    print(f"no-cold category pages ({len(mix)} sampled requests): share of the top 10 in the browsed subcategory")
    print(pd.DataFrame(mix).mean().round(3).to_string())


if __name__ == "__main__":
    be_nice()
    {"ablation": ablation, "recall": recall, "tune": tune, "carousel": carousel_recall,
     "e2e": end_to_end}[sys.argv[1] if len(sys.argv) > 1 else "ablation"]()
