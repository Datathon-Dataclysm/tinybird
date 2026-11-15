"""Train and validate the Lynch (v8) ranker on the live feedback, then (with --final) save it.

    .venv/bin/python datathon-fme-2026/lynch/train_v8.py            # evaluate only (time split), writes models/eval_v8.json
    .venv/bin/python datathon-fme-2026/lynch/train_v8.py --final    # + train the chosen variant on ALL rows -> models/ranker_v8.txt

1. Replay the live day point-in-time with v7's replay (train_live.replay), MF factors loaded -> 31 features per row.
2. Time split: first 75% of requests train, last 25% test. Candidates: spike-s1 (= v7 features, the reference),
   spike-s4-age, spike-s4-age-mf. LightGBM, v7 params and target (click + 3x purchase); slot = 0 at test time.
3. Paired request-level bootstrap vs spike-s1: MRR of the purchased item among the 10 shown, purchase AUC; per segment.
4. Decision rule (fixed before looking at the numbers):
     non-inferior  := MRR difference CI lower bound > -0.02 (all requests)
     choose spike-s4-age-mf if non-inferior AND new-user purchase AUC difference >= 0
     else spike-s4-age if non-inferior, else spike-s1.
5. New-product calibration for the chosen variant: Platt-calibrate on half the test requests, observed/expected purchases
   of products < 2 days old on the other half. new_boost = clip(obs/exp, 1, 2) if the 95% CI lower bound > 1, else 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import FEATURE_SETS, FEATURES_ALL, NEW_DAYS_CAL, Store  # noqa: E402
from rawtree import be_nice  # noqa: E402
from train_live import TARGETS, auc, cached, fetch_live, fit, replay  # noqa: E402

MODELS = HERE / "models"
REF = "spike-s1"


def ranks_of(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rank (1 = best, ties averaged) of the purchased item among the 10 shown; NaN if no purchase."""
    S, Y = scores.reshape(-1, 10), y.reshape(-1, 10)
    has = Y.sum(1) > 0
    tgt = np.where(has, (S * Y).sum(1) / np.maximum(Y.sum(1), 1), np.nan)
    return np.where(has, 1 + (S > tgt[:, None]).sum(1) + 0.5 * ((S == tgt[:, None]).sum(1) - 1), np.nan)


def compare(scores: dict, yb: np.ndarray, mask: np.ndarray, rng, B: int = 1000) -> dict:
    """Paired bootstrap (requests) of MRR and purchase AUC differences vs REF on the requests in `mask`."""
    req = np.where(mask)[0]
    rk = {k: ranks_of(v, yb) for k, v in scores.items()}
    has = req[~np.isnan(rk[REF][req])]
    boots = rng.integers(0, len(has), (B, len(has)))
    out = {}
    rows_all = (req[:, None] * 10 + np.arange(10)).ravel()
    auc_b = {k: [] for k in scores}
    for _ in range(200):
        b = rng.choice(req, len(req)); rows = (b[:, None] * 10 + np.arange(10)).ravel()
        base = auc(scores[REF][rows], yb[rows])
        for k in scores:
            auc_b[k].append(auc(scores[k][rows], yb[rows]) - base)
    for k in scores:
        rr = 1 / rk[k][has]; d = rr - 1 / rk[REF][has]; db = d[boots].mean(1)
        out[k] = {"purchases": int(len(has)), "mrr": float(rr.mean()), "top1": float(np.mean(np.clip(2 - rk[k][has], 0, 1))),
                  "d_mrr": float(d.mean()), "d_mrr_lo": float(np.quantile(db, .025)), "d_mrr_hi": float(np.quantile(db, .975)),
                  "auc": auc(scores[k][rows_all], yb[rows_all]), "d_auc": float(np.nanmean(auc_b[k])),
                  "d_auc_lo": float(np.nanquantile(auc_b[k], .025)), "d_auc_hi": float(np.nanquantile(auc_b[k], .975))}
    return out


def new_product_boost(score: np.ndarray, yb: np.ndarray, age: np.ndarray, rng) -> dict:
    from sklearn.linear_model import LogisticRegression
    n_req = len(score) // 10
    half = np.repeat(rng.random(n_req) < 0.5, 10)
    lo = np.log(np.clip(score, 1e-6, 1 - 1e-6) / (1 - np.clip(score, 1e-6, 1 - 1e-6)))
    cal = LogisticRegression().fit(lo[half][:, None], yb[half])
    p = cal.predict_proba(lo[:, None])[:, 1]
    m = ~half & (age < NEW_DAYS_CAL)
    obs, exp = yb[m].sum(), p[m].sum()
    idx = np.where(m)[0]
    bs = [yb[i].sum() / max(p[i].sum(), 1e-9) for i in (rng.choice(idx, len(idx)) for _ in range(500))]
    ratio, lo95 = float(obs / exp), float(np.quantile(bs, .025))
    boost = float(np.clip(ratio, 1.0, 2.0)) if lo95 > 1 else 1.0
    return {"rows": int(m.sum()), "purchases": int(obs), "expected": float(exp), "obs_over_exp": ratio,
            "ci": [lo95, float(np.quantile(bs, .975))], "new_boost": boost}


def main() -> None:
    be_nice()
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="train the chosen variant on ALL rows and save ranker_v8")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    t0 = time.time()
    store = cached("store_v8_history", lambda: Store.load(until=None))
    if not store.load_mf(MODELS / "mf_v8.npz"):
        sys.exit("models/mf_v8.npz missing: run lynch/build_mf.py first")
    store.live_cs, store.shown = {}, defaultdict(Counter)
    events, updates, fb = fetch_live()
    fb = fb[fb.shown.apply(len) == 10].reset_index(drop=True)
    print(f"{len(fb):,} requests with feedback, {int((fb.purchased != '').sum())} purchases", flush=True)
    meta = []
    X, yb, yc = replay(store, events, updates, fb, hook=lambda k, row, ctx: meta.append((bool(row.is_new_user), row.ts)))
    n_req = len(meta); cut_req = int(n_req * 0.75); cut = cut_req * 10
    new_user = np.array([m[0] for m in meta])
    print(f"replayed {n_req:,} requests ({time.time() - t0:.0f}s); test from {meta[cut_req][1]}", flush=True)

    y, w = TARGETS["click + 3x purchase"](yb[:cut], yc[:cut])
    Xte = X[cut:].copy(); Xte[:, FEATURES_ALL.index("pos")] = 0
    models, scores = {}, {}
    for name, feats in FEATURE_SETS.items():
        cols = [FEATURES_ALL.index(f) for f in ["pos"] + [f for f in feats if f != "pos"]]
        models[name] = (fit(X[:cut], y, w, cols=cols), cols)
        scores[name] = models[name][0].predict(Xte[:, cols])
    te_new = new_user[cut_req:]
    age = Xte[:, FEATURES_ALL.index("age_days")]
    segs = {"all": np.ones(len(te_new), bool), "new users": te_new, "returning": ~te_new,
            "lists with a product < 2 days old": (age.reshape(-1, 10) < NEW_DAYS_CAL).any(1)}
    report = {seg: compare(scores, yb[cut:], m, rng) for seg, m in segs.items()}

    ni = lambda k: report["all"][k]["d_mrr_lo"] > -0.02  # noqa: E731
    if ni("spike-s4-age-mf") and report["new users"]["spike-s4-age-mf"]["d_auc"] >= 0:
        chosen = "spike-s4-age-mf"
    elif ni("spike-s4-age"):
        chosen = "spike-s4-age"
    else:
        chosen = REF
    cal = new_product_boost(scores[chosen], yb[cut:], age, rng)
    boosted = scores[chosen] * np.where(age < NEW_DAYS_CAL, cal["new_boost"], 1.0)
    with_boost = compare({REF: scores[REF], chosen: scores[chosen], f"{chosen} + new_boost": boosted}, yb[cut:], segs["all"], rng)

    for seg, r in report.items():
        print(f"\n{seg}")
        for k, v in r.items():
            print(f"  {k:18s} purchases {v['purchases']:4d}  MRR {v['mrr']:.3f}  ΔMRR {v['d_mrr']:+.3f} [{v['d_mrr_lo']:+.3f}, {v['d_mrr_hi']:+.3f}]"
                  f"  AUC {v['auc']:.3f}  ΔAUC {v['d_auc']:+.4f} [{v['d_auc_lo']:+.4f}, {v['d_auc_hi']:+.4f}]")
    print(f"\nchosen: {chosen}; new-product calibration: {cal}")
    print(f"with boost (all): ΔMRR {with_boost[f'{chosen} + new_boost']['d_mrr']:+.3f} vs ref, AUC {with_boost[f'{chosen} + new_boost']['auc']:.3f}")
    ev = {"base_commit": (HERE / "BASE_COMMIT").read_text().strip(), "requests": n_req, "purchases": int(yb.sum()),
          "train_requests": cut_req, "test_from": meta[cut_req][1], "last_request": meta[-1][1], "chosen": chosen,
          "decision_rule": "non-inferior: dMRR CI lo > -0.02; MF if also new-user dAUC >= 0", "report": report,
          "new_product_calibration": cal, "with_boost": with_boost, "secs": round(time.time() - t0)}
    MODELS.mkdir(exist_ok=True)
    (MODELS / "eval_v8.json").write_text(json.dumps(ev, indent=1, default=float))

    if args.final:
        cols = models[chosen][1]
        yA, wA = TARGETS["click + 3x purchase"](yb, yc)
        final = fit(X, yA, wA, cols=cols, rounds=max(models[chosen][0].best_iteration, 50))
        path = MODELS / "ranker_v8.txt"
        final.save_model(str(path.with_suffix(".tmp")))
        os.replace(path.with_suffix(".tmp"), path)       # atomic: a live process hot-reloads it
        side = {"name": "lynch-v8", "feature_set": chosen, "features": [FEATURES_ALL[c] for c in cols],
                "new_boost": cal["new_boost"], "trained_on_requests": n_req, "data_until": meta[-1][1],
                "rounds": final.current_iteration(), "base_commit": ev["base_commit"]}
        path.with_suffix(".json").write_text(json.dumps(side, indent=1))
        print(f"saved {path.name} + {path.with_suffix('.json').name}: {side}")


if __name__ == "__main__":
    main()
