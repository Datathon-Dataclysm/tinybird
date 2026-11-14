"""Live recommender: the personalised model answering shop.requests in real time.

    python starter/live.py --shadow                 # dry run: compute answers, measure, DO NOT write submissions
    python starter/live.py --version v4             # production: write answers to team_NN.submissions
    python starter/live.py --version v4 --adjust ab # production, A/B test of the new-product/trend multipliers

Only ONE production process per team: if two processes answer the same request, only one answer counts.

Two threads:
- main loop (every 0.25 s): pull new requests -> recommend (in memory, milliseconds) -> push one batch
- refresher: keeps the in-memory store fresh without slowing the answers
    every 2 s   new shop.events          -> user x item / subcategory / brand tables (carts made a minute ago count)
    every 1 s   new shop.catalog_updates -> new products, prices, stock-outs (stock-outs reach the table ~2 s late;
                                            the organisers forgive the last 5 s, so 1 s polling keeps us inside)
    every 60 s  team feedback            -> what each shopper clicked / bought in our carousel (a strong signal)
    every 60 s  trends                   -> purchase share of each subcategory in the last 2 h vs usual
    every 5 min popularity of the last 7 days (history + live)

--adjust ab splits shoppers in two halves (by user id) and writes model_version "<version>-adj" / "<version>-noadj",
so the dashboard compares them on the same traffic.

Shadow mode also scores itself: for each request, did the shopper add to cart / buy (organically, shop.events)
within 10 minutes a product that was in our list? And in the baseline's list? Printed every minute.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
import zlib
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Ranker, Recommender, Store  # noqa: E402
from rawtree import ROOT, TEAM, Record, insert, query, to_epoch  # noqa: E402

REQUESTS = """
SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts, user_id::String AS user_id,
       is_new_user::Bool AS is_new_user, context.device::String AS device, context.region::String AS region,
       context.page::String AS page, recent_views
FROM shop.requests WHERE ts::DateTime64(3) > now64(3) - INTERVAL 20 SECOND"""

EVENTS = """
SELECT user_id::String AS u, product_id::String AS p, event_type::String AS e, toString(ts::DateTime64(3)) AS t
FROM shop.events WHERE ts::DateTime64(3) > toDateTime64('{since}', 3) - INTERVAL 10 SECOND
ORDER BY ts LIMIT 50000"""

UPDATES = """
SELECT product_id::String AS product_id, change::String AS change, category::String AS category,
       subcategory::String AS subcategory, brand::String AS brand, price_eur::Float64 AS price_eur,
       is_active::Bool AS is_active, toString(created_at::DateTime64(3)) AS created_at, toString(ts::DateTime64(3)) AS t
FROM shop.catalog_updates WHERE ts::DateTime64(3) > toDateTime64('{since}', 3) ORDER BY ts"""

CAROUSEL = f"""
SELECT r.user_id::String AS u, f.clicked::String AS p, count() AS c,
       countIf(f.purchased::String = f.clicked::String) AS b
FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.clicked::String != '' GROUP BY u, p"""


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Live:
    def __init__(self, model_path: Path, version: str, shadow: bool, adjust: str = "off",
                 ab_candidates: bool = False) -> None:
        self.version, self.shadow, self.adjust, self.ab_candidates = version, shadow, adjust, ab_candidates
        log("loading store (history + today's events), ~2 min ...")
        self.store = Store.load(live=True)
        ranker = Ranker.load(model_path) if model_path.exists() else Ranker()
        log(f"ranker: {model_path.name if model_path.exists() else 'heuristic (no model file found)'} | adjust: {adjust}")
        self.rec = Recommender(self.store, ranker, adjust=adjust != "off")
        self.rec_noadj = Recommender(self.store, ranker, adjust=False)
        self.rec_narrow = Recommender(self.store, ranker, adjust=adjust == "on", wide=False)
        self.model_path = model_path
        self.model_mtime = model_path.stat().st_mtime if model_path.exists() else 0.0
        self.lock = threading.Lock()              # the refresher mutates the store, the main loop reads it
        self.seen: dict[str, float] = {}          # request_id -> when we saw it (dedupe, pruned)
        self.recent_keys: deque = deque()         # event dedupe across overlapping polls
        self.recent_set: set = set()
        self.updates_since = "2026-01-01 00:00:00.000"
        self._max_update = self.updates_since
        self.stats = {"answered": 0, "invalid": 0, "latency": deque(maxlen=2000), "compute_ms": deque(maxlen=2000)}
        # shadow scoring: compare with the baseline (all-time best sellers in stock)
        from recommender import best_sellers
        self.best = best_sellers()
        self.pending: deque = deque()             # (request_id, user, ts_epoch, ours, baseline)
        self.proxy = {"n": 0, "ours": 0, "baseline": 0, "any": 0}
        self.shadow_log = (ROOT / "logs").joinpath("shadow.jsonl") if shadow else None
        if self.shadow_log:
            self.shadow_log.parent.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        threading.Thread(target=self.refresher, daemon=True).start()
        for r in query(REQUESTS, timeout=10):     # already past the deadline at start-up: the fallback served them
            self.seen[r["request_id"]] = time.time()
        last_report = time.time()
        while True:
            try:
                fresh = [r for r in query(REQUESTS, timeout=5) if r["request_id"] not in self.seen]
                if fresh:
                    answers = self.answer(fresh)
                    if not self.shadow:
                        insert("submissions", answers, timeout=5)
                    now = time.time()
                    for r in fresh:
                        self.stats["latency"].append(now - to_epoch(r["ts"]) - _frac(r["ts"]))
                if time.time() - last_report > 60:
                    last_report = time.time()
                    self.report()
            except Exception as e:  # keep running: shoppers keep arriving while you sleep
                log(f"loop error: {e!r}")
                time.sleep(1)
            self._prune_seen()
            time.sleep(0.25)

    def answer(self, reqs: list[Record]) -> list[Record]:
        out = []
        with self.lock:
            for r in reqs:
                t = time.perf_counter()
                rec, version = self.rec, self.version
                half = zlib.crc32(r["user_id"].encode()) % 2  # stable half of the shoppers each
                if self.adjust == "ab":                        # A/B of the live multipliers
                    rec, version = (self.rec, f"{self.version}-adj") if half else (self.rec_noadj, f"{self.version}-noadj")
                elif self.ab_candidates:                       # A/B of the candidate generation
                    rec, version = (self.rec, f"{self.version}-wide") if half else (self.rec_narrow, f"{self.version}-narrow")
                items = rec.recommend(r)
                self.stats["compute_ms"].append(1000 * (time.perf_counter() - t))
                if len(items) != 10 or len(set(items)) != 10 or any(p in self.store.unavailable for p in items):
                    self.stats["invalid"] += 1
                    items = self.rec.fill([], 10)            # never send an invalid list
                self.seen[r["request_id"]] = time.time()
                for p in items:                               # fatigue feature: times we showed it to this shopper
                    self.store.shown[r["user_id"]][p] += 1
                out.append({"request_id": r["request_id"], "items": items, "model_version": version})
                if self.shadow:
                    base = [p for p in self.best if p not in self.store.unavailable][:10]
                    self.pending.append((r["request_id"], r["user_id"], to_epoch(r["ts"]), items, base))
        self.stats["answered"] += len(out)
        if self.shadow_log:
            with self.shadow_log.open("a", encoding="utf-8") as f:
                for a in out:
                    f.write(json.dumps(a) + "\n")
        return out

    def _prune_seen(self) -> None:
        if len(self.seen) > 20000:
            cutoff = time.time() - 120
            self.seen = {k: v for k, v in self.seen.items() if v > cutoff}

    # ------------------------------------------------------------------ refresher thread
    def refresher(self) -> None:
        jobs = [(2, self.refresh_events), (1, self.refresh_catalog), (60, self.refresh_carousel),
                (60, self.refresh_trends), (300, self.refresh_popularity), (30, self.reload_model)]
        if self.shadow:
            jobs.append((60, self.score_shadow))
        last = {f: 0.0 for _, f in jobs}
        while True:
            for every, job in jobs:
                if time.time() - last[job] >= every:
                    last[job] = time.time()
                    try:
                        job()
                    except Exception as e:
                        log(f"{job.__name__} failed: {e!r}")
            time.sleep(0.5)

    def refresh_events(self) -> None:
        rows = query(EVENTS.format(since=self.store.last_event_ts or "2026-11-14 00:00:00.000"), timeout=10)
        new = []
        for r in rows:
            key = (r["u"], r["p"], r["e"], r["t"])
            if key in self.recent_set:
                continue
            self.recent_set.add(key)
            self.recent_keys.append(key)
            new.append((r["u"], r["p"], r["e"], 1, to_epoch(r["t"])))
        while len(self.recent_keys) > 50000:
            self.recent_set.discard(self.recent_keys.popleft())
        if new:
            with self.lock:
                self.store.ingest(new)
        if rows:
            self.store.last_event_ts = max(self.store.last_event_ts, rows[-1]["t"])

    def refresh_catalog(self) -> None:
        rows = query(UPDATES.format(since=self.updates_since), timeout=10)
        if rows:
            with self.lock:
                for u in rows:
                    self.store.apply_catalog_update(u)
            self.updates_since = rows[-1]["t"]

    def refresh_carousel(self) -> None:
        """What each shopper clicked / bought in our carousel today (feedback). Bought there -> often bought again."""
        live: dict[str, dict[str, list[float]]] = {}
        for r in query(CAROUSEL, timeout=20):
            live.setdefault(r["u"], {})[r["p"]] = [float(r["c"]), float(r["b"])]
        self.store.live_cs = live                    # swapped at once

    def refresh_trends(self) -> None:
        self.store.load_trends()
        top = sorted(self.store.trend_sc.items(), key=lambda kv: -kv[1])[:3]
        log("trending: " + ", ".join(f"{sc} x{t:.2f}" for sc, t in top))

    def refresh_popularity(self) -> None:
        self.store.load_popularity(None, live=True)

    def reload_model(self) -> None:
        """Hot reload: when the model file changes (e.g. train_live.py --final retrained it), swap it in
        without restarting, so no request goes unanswered. A broken file is ignored and the old model stays."""
        if not self.model_path.exists():
            return
        mtime = self.model_path.stat().st_mtime
        if mtime == self.model_mtime:
            return
        ranker = Ranker.load(self.model_path)
        self.rec.ranker = self.rec_noadj.ranker = self.rec_narrow.ranker = ranker
        self.model_mtime = mtime
        log(f"reloaded model {self.model_path.name}")

    # ------------------------------------------------------------------ shadow scoring
    def score_shadow(self, horizon: float = 600) -> None:
        """Requests older than `horizon` s: did the shopper cart/buy a product of our list? of the baseline's?"""
        ready = []
        while self.pending and self.pending[0][2] < time.time() - horizon:
            ready.append(self.pending.popleft())
        if not ready:
            return
        users = sorted({u for _, u, _, _, _ in ready})
        events: dict[str, list[tuple[str, float]]] = {}
        for i in range(0, len(users), 500):
            chunk = ",".join(f"'{u}'" for u in users[i:i + 500])
            for r in query(f"""SELECT user_id::String AS u, product_id::String AS p, toString(ts::DateTime64(3)) AS t
                               FROM shop.events WHERE user_id::String IN ({chunk})
                               AND event_type::String IN ('add_to_cart', 'purchase')
                               AND ts::DateTime64(3) > now64(3) - INTERVAL 40 MINUTE""", timeout=30):
                events.setdefault(r["u"], []).append((r["p"], to_epoch(r["t"])))
        for _, u, ts, ours, base in ready:
            later = {p for p, t in events.get(u, []) if ts <= t <= ts + horizon}
            self.proxy["n"] += 1
            self.proxy["any"] += bool(later)
            self.proxy["ours"] += bool(later & set(ours))
            self.proxy["baseline"] += bool(later & set(base))

    # ------------------------------------------------------------------ report
    def report(self) -> None:
        lat = np.array(self.stats["latency"]) if self.stats["latency"] else np.zeros(1)
        cm = np.array(self.stats["compute_ms"]) if self.stats["compute_ms"] else np.zeros(1)
        msg = (f"{'SHADOW' if self.shadow else self.version}: answered {self.stats['answered']} "
               f"(invalid fixed {self.stats['invalid']}) | ready after p50 {np.median(lat):.2f}s p95 "
               f"{np.percentile(lat, 95):.2f}s | compute p50 {np.median(cm):.1f}ms p95 {np.percentile(cm, 95):.1f}ms")
        if not self.shadow:
            msg += " (+ insert time)"
        log(msg)
        if self.shadow and self.proxy["n"]:
            n = self.proxy["n"]
            log(f"  organic proxy over {n} requests (10 min later): shopper carted/bought something in "
                f"OUR list {100 * self.proxy['ours'] / n:.1f}% | BASELINE list {100 * self.proxy['baseline'] / n:.1f}% "
                f"| anything at all {100 * self.proxy['any'] / n:.1f}%")


def _frac(ts: str) -> float:
    """Fractional seconds of '... HH:MM:SS.fff' (to_epoch truncates them)."""
    return float("0." + ts.split(".")[1]) if "." in ts else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", action="store_true", help="do not write submissions, only measure")
    ap.add_argument("--version", default="v4", help="model_version written with each answer")
    ap.add_argument("--model", default=str(ROOT / "models" / "ranker_v4.txt"))
    ap.add_argument("--adjust", choices=["on", "off", "ab"], default="off",
                    help="live price-drop / trend multipliers: on, off, or A/B by user")
    ap.add_argument("--ab-candidates", action="store_true",
                    help="A/B by user: wide candidates (with category best sellers) vs the v3 candidates")
    args = ap.parse_args()
    Live(Path(args.model), args.version, args.shadow, args.adjust, args.ab_candidates).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
