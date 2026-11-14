"""Train the LightGBM ranker on shop.history_sessions and evaluate it offline against the baseline.

    python starter/train.py                  # offline experiment: train week -> test week, writes models/ranker_offline.txt
    python starter/train.py --final          # trained on the most recent week, for the live system: models/ranker_v2.txt

Time split (no peeking into the future):

    history ......... T0 ........ CUT ........ END
    features from <T0 | train sessions |
    features from <CUT              | test sessions (evaluation)

Every training row is (session, shown product) with label "was it purchased". The features of a row only use
data from before the session's window, exactly like live, where they only use data from before the request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CATEGORICAL, FEATURES, Ranker, Recommender, Store, request_from_row  # noqa: E402
from rawtree import ROOT, query_df  # noqa: E402

CACHE = ROOT / ".cache"
MODELS = ROOT / "models"

SESSIONS = """
SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts, user_id::String AS user_id,
       context.device::String AS device, context.region::String AS region, context.page::String AS page,
       recent_views, shown::Array(String) AS shown, propensity::Array(Float64) AS propensity,
       clicked::String AS clicked, purchased::String AS purchased
FROM shop.history_sessions
WHERE ts >= '{start}' AND ts < '{end}' AND ({sample})
"""


def cached(name: str, build):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    obj = build()
    path.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    return obj


def load_store(until: str) -> Store:
    print(f"store until {until}")
    return cached(f"store_v2_{until}", lambda: Store.load(until=until))


def sessions(start: str, end: str, sample: str) -> pd.DataFrame:
    key = f"sessions_{start}_{end}_{hashlib.md5(sample.encode()).hexdigest()[:8]}"
    return cached(key, lambda: query_df(SESSIONS.format(start=start, end=end, sample=sample)))


def build_rows(rec: Recommender, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per (session, shown product): features, label (purchased), session id (for grouping)."""
    X, y, g = [], [], []
    for i, row in enumerate(df.itertuples(index=False)):
        ctx = rec.context(request_from_row(row))
        shown = list(row.shown)
        X.append(rec.features(ctx, shown, positions=list(range(len(shown)))))
        y.extend(p == row.purchased for p in shown)
        g.extend([i] * len(shown))
    return np.vstack(X), np.array(y, dtype=float), np.array(g)


def train(X: np.ndarray, y: np.ndarray, g: np.ndarray, seed: int = 7):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    sessions_ = np.unique(g)
    val = set(rng.choice(sessions_, size=len(sessions_) // 5, replace=False))
    is_val = np.array([s in val for s in g])
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed)
    dtrain = lgb.Dataset(X[~is_val], y[~is_val], feature_name=FEATURES, categorical_feature=CATEGORICAL)
    dval = lgb.Dataset(X[is_val], y[is_val], reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"  trained {booster.best_iteration} trees, validation logloss {booster.best_score['valid_0']['binary_logloss']:.5f}")
    return booster


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(policy, df: pd.DataFrame) -> dict[str, float]:
    """The notebook's metrics, plus the same ones for purchases (what the leaderboard scores)."""
    recs = [policy(row) for row in df.itertuples(index=False)]
    clicked = df.clicked.to_numpy()
    bought = df.purchased.to_numpy()
    has_click, has_buy = clicked != "", bought != ""
    first = np.array([s[0] for s in df.shown])
    match = np.array([r[0] == s for r, s in zip(recs, first)])
    w = match / np.array([p[0] for p in df.propensity])
    out = {
        "hit@10 click": np.mean([c in r for c, r in zip(clicked[has_click], np.array(recs, dtype=object)[has_click])]),
        "hit@10 purchase": np.mean([b in r for b, r in zip(bought[has_buy], np.array(recs, dtype=object)[has_buy])]),
        "SNIPS ctr slot1": float((w * (clicked == first)).sum() / max(w.sum(), 1e-9)),
        "SNIPS buy slot1": float((w * (bought == first)).sum() / max(w.sum(), 1e-9)),
        "slot1 matches": int(match.sum()),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=7)
    ap.add_argument("--test-n", type=int, default=30000)
    ap.add_argument("--neg-share", type=float, default=0.25, help="share of sessions without a click kept for training")
    ap.add_argument("--final", action="store_true", help="train on the latest week (for live), skip the test")
    args = ap.parse_args()

    last_day = pd.Timestamp(query_df("SELECT toDate(max(ts::DateTime64(3))) AS d FROM shop.history_sessions").d[0])
    end = last_day + pd.Timedelta(days=1)
    cut = end if args.final else last_day - pd.Timedelta(days=6)                   # test = last 7 days
    t0 = cut - pd.Timedelta(days=args.train_days)
    CUT, T0, END = (str(d.date()) for d in (cut, t0, end))
    print(f"train sessions [{T0}, {CUT})" + ("" if args.final else f"   test sessions [{CUT}, {END})"))

    # ---------------------------------------------------------------- training set
    sample = f"clicked != '' OR cityHash64(request_id::String) % 1000 < {int(args.neg_share * 1000)}"
    tr = sessions(T0, CUT, sample)
    print(f"  {len(tr):,} training sessions, {int((tr.purchased != '').sum()):,} with a purchase")
    rec_train = Recommender(load_store(T0))
    t = time.time()
    X, y, g = build_rows(rec_train, tr)
    print(f"  {len(y):,} rows built in {time.time() - t:.0f}s")
    booster = train(X, y, g)
    MODELS.mkdir(exist_ok=True)
    name = "ranker_v2.txt" if args.final else "ranker_offline.txt"
    booster.save_model(str(MODELS / name))
    imp = pd.Series(booster.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("  feature importance (gain, %):")
    print((100 * imp / imp.sum()).round(1).head(15).to_string())
    if args.final:
        print(f"saved models/{name}")
        return

    # ---------------------------------------------------------------- offline test
    te = sessions(CUT, END, "cityHash64(request_id::String) % 10 = 0")
    te = te.sample(n=min(args.test_n, len(te)), random_state=1).reset_index(drop=True)
    print(f"test: {len(te):,} sessions, {int((te.clicked != '').sum())} clicks, {int((te.purchased != '').sum())} purchases")
    store = load_store(CUT)
    heur = Recommender(store)
    lgbm = Recommender(store, Ranker(booster))

    # 1) re-ranking the products the old system showed: does the model put the purchased one higher?
    Xt, yt, _ = build_rows(lgbm, te)
    Xt[:, FEATURES.index("pos")] = 0
    print(f"\nre-ranking shown products, AUC of purchase:  lightgbm {auc(lgbm.ranker.score(Xt), yt):.3f}   "
          f"heuristic {auc(heur.ranker.score(Xt), yt):.3f}   old order {auc(-np.tile(np.arange(10), len(te))[:len(yt)], yt):.3f}")

    # 2) full policies
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from recommender import best_sellers
    best = best_sellers(until=CUT)
    policies = {
        "baseline": lambda r: best[:10],
        "heuristic": lambda r: heur.recommend(request_from_row(r)),
        "lightgbm": lambda r: lgbm.recommend(request_from_row(r)),
    }
    results = {}
    for name_, pol in policies.items():
        t = time.time()
        results[name_] = evaluate(pol, te)
        print(f"  evaluated {name_} in {time.time() - t:.0f}s")
    table = pd.DataFrame(results).T
    print(table.round(4).to_string())
    (MODELS / "offline_results.json").write_text(json.dumps(
        {"train": [T0, CUT], "test": [CUT, END], "results": table.to_dict("index")}, indent=1, default=float))


if __name__ == "__main__":
    main()
