# Lynch (v8): v7 pipeline + evidence-driven changes

Self-contained folder. Base: v7 `e986b41` (origin/vulture), copied, not modified in place, so v7 remains the rollback.
Evidence: `dataclysm/docs/experiments.md` (E1–E4 and the spike-s1…s4 feature-set run).

## What changed vs v7
| Change | Why (evidence) |
|---|---|
| Feature set **spike-s4-age-mf** (25 cols): v7 minus context (page/device/region/is_new_user) and minus trend/fatigue, **keeping `age_days`**, plus `mf_hist`, `mf_foldin` | context/trend/fatigue: no measurable value (CIs span 0). Context is near-constant if Sunday is all new shoppers. `age_days` fixes new-product under-prediction. MF fold-in helps new users |
| **MF factors** from the history: RawTree SQL aggregation of 8.8 M user×product weights, SVD k=48 (`build_mf.py`); live fold-in from each shopper's events | the only S8 ingredient that adds to v7 (cold users) |
| Optional **new-product calibration** (`new_boost` in the model's .json) | estimated at train time; currently **1.0** (with `age_days` kept, new products are calibrated: obs/exp 1.12 [0.75, 1.51]) |
| Model declares its own features; one builder computes all 31 | the same code serves v7-style and v8 models |
| `live.py` is **shadow by default**; `--submit` writes | nothing is submitted by accident |

## lynch-v9a = lynch-v8 + at most 3 products per subcategory (`--max-per-subcat 3`)
Same model; only the order of the list changes (the 4th+ product of a subcategory moves to the end; on live requests
this replaces 0.7 of the 10 products on average). A request ends in at most one purchase, so near-identical products
compete for it. Evidence so far is mixed: on v9 (vulture, 00:15-01:01) the cap raised conversion 2.18 -> 2.39, mostly
in cold requests; a 15-minute lynch test (01:50-02:05) went the other way, within noise. Hence the overnight A/B.

## lynch-v9c = lynch-v9a + price tilt 0.05 (`--price-tilt 0.05`): rank by score x price^0.05
The model has no price feature and under-predicts expensive products: on held-out live data, real/predicted purchases
go 0.82 (cheapest fifth) -> 1.32 (most expensive fifth), i.e. purchases ~ price^0.16 (95% CI 0.10-0.22) beyond the
model. A tilt of 0.05 changes 0.5 of the 10 products (0.2 in the top 3): EUR/request +11%, conversion -0.2% by the
model alone, +1.1% once corrected with that calibration. Observational, hence also in the overnight A/B.

Tried and dropped (not in the code): a cap per category (only 10 exist: it replaced 3 of the 10 products and was
worse), one model per segment (cold-only / no-cold-only models were worse in their own segment; a 4-model blend
gained < 0.5% MRR, within noise).

Sunday plan (A/B tonight, decision rule, retraining, final launch): `PLAN_DOMINGO.md`.

## A/B and reading it
`--ab lynch-v8=0,lynch-v9a=3` runs both versions in ONE process (one store, one model): each request goes to one version
by a hash of its id, so both see the same shoppers at the same time and drift during the night cancels out. Do not
compare versions that ran one after the other: the whole market moves ±10% from one hour to the next.

`ab_report.py START [END]` reads the leaderboard API in 15-minute pieces and prints conversion, cold, no-cold and EUR per
version, with each difference ± 2 SE. Overnight (3 arms, ~5 h), only differences above ~0.15-0.25 points are real.

## Windows note
`/.gitattributes` keeps `models/*.txt` byte-exact and `Ranker.load` strips carriage returns: with `core.autocrlf=true`,
git used to check the model out with CRLF line endings and LightGBM failed ("Model format error, expect a tree here").

## Validation (train_v8.py, time split: train until 19:12 UTC, test 19:12–22:06 UTC, 653 test purchases)
Reference = the v7 feature set retrained on the same data. 95% CIs from a paired request-level bootstrap.

| Segment | ΔMRR (purchased item among 10 shown) | Δ purchase AUC |
|---|---|---|
| all | +0.010 [−0.007, +0.029] | **+0.0097 [+0.0045, +0.0148]** |
| new users (291 purchases) | +0.017 [−0.010, +0.044] | **+0.0151 [+0.0078, +0.0234]** |
| returning | +0.005 [−0.018, +0.028] | **+0.0079 [+0.0015, +0.0140]** |
| lists with a product < 2 days old | +0.018 [−0.009, +0.044] | **+0.0131 [+0.0058, +0.0196]** |

Shadow run against the live stream (6 min, 1,079 requests): **0 invalid**, ready **p50 0.72 s / p95 1.30 s** after the request
`ts` (excludes the submission insert, typically 0.2–0.4 s), compute p50 4.5 ms / p95 8.3 ms.

## Expected results (honest)
- **Ranking quality:** a small, consistent improvement over v7's features (AUC +0.01; +0.015 for new users), significant on
  AUC, not yet on MRR/top-1. The final model is also trained on ~125 k requests vs ~70 k for the deployed v7 file, which is an
  additional (unmeasured) gain.
- **Expected purchases per request:** likely **0 to +5% relative** over v7. That's too small to confirm with an overnight A/B
  (≈ +13% needed at 3 req/s, ~1.5% conversion). So an A/B run is a **non-inferiority / safety check**, not a proof of gain.
- **Where it should matter most:** the final round. Fewer features that depend on context, better new-user and
  new-product ranking.
- **Risks:** the MF fold-in depends on `mf_v8.npz` (if it's missing, MF features are 0 and the model degrades towards
  spike-s4-age). Startup takes ~1 min (store + factors).

## Commands (from `datathon-fme-2026/`)
```bash
../.venv/bin/python lynch/build_mf.py              # once (history is immutable): models/mf_v8.npz
../.venv/bin/python lynch/train_v8.py              # evaluate only -> models/eval_v8.json
../.venv/bin/python lynch/train_v8.py --final      # + retrain on all feedback -> models/ranker_v8.txt (+ .json); hot-reloaded by live.py
../.venv/bin/python lynch/live.py --minutes 5      # shadow test
../.venv/bin/python lynch/live.py --submit         # PRODUCTION (only one process per team!)
../.venv/bin/python lynch/live.py --submit --version lynch-v9a --max-per-subcat 3   # PRODUCTION, v9a
../.venv/bin/python lynch/live.py --submit --version lynch-v9c --max-per-subcat 3 --price-tilt 0.05   # v9c
../.venv/bin/python lynch/live.py --submit --ab lynch-v8=0,lynch-v9a=3,lynch-v9c=3:0.05   # A/B, one process
../.venv/bin/python lynch/ab_report.py 02:45 06:45                                 # read the A/B (local times)
```
Rollback: stop `lynch/live.py --submit` and start v7's `starter/live.py --version v7` from origin/vulture.
