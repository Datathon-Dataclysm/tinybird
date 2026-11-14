"""Live recommender: the personalised model answering shop.requests in real time.

    python starter/live.py --shadow                 # dry run: compute answers, measure, DO NOT write submissions
    python starter/live.py --version v1-lgbm        # production: write answers to team_NN.submissions

Only ONE production process per team: if two processes answer the same request, only one answer counts.

Two threads:
- main loop (every 0.25 s): pull new requests -> recommend (in memory, milliseconds) -> push one batch
- refresher: keeps the in-memory store fresh without slowing the answers
    every 2 s   new shop.events          -> user x item / user x subcategory tables (carts made a minute ago count)
    every 5 s   new shop.catalog_updates -> new products, prices, stock-outs (an out-of-stock item invalidates a list)
    every 60 s  team feedback            -> products a shopper already bought from our carousel are not shown again
    every 5 min popularity of the last 7 days (history + live)

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

BOUGHT = f"""
SELECT r.user_id::String AS u, f.purchased::String AS p
FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.purchased::String != '' AND f.served_by::String = 'team'"""


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Live:
    def __init__(self, model_path: Path, version: str, shadow: bool) -> None:
        self.version, self.shadow = version, shadow
        log("loading store (history + today's events), ~1-2 min ...")
        self.store = Store.load(live=True)
        ranker = Ranker.load(model_path) if model_path.exists() else Ranker()
        log(f"ranker: {model_path.name if model_path.exists() else 'heuristic (no model file found)'}")
        self.rec = Recommender(self.store, ranker)
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
                items = self.rec.recommend(r)
                self.stats["compute_ms"].append(1000 * (time.perf_counter() - t))
                if len(items) != 10 or len(set(items)) != 10 or any(p in self.store.unavailable for p in items):
                    self.stats["invalid"] += 1
                    items = self.rec.fill([], 10)            # never send an invalid list
                self.seen[r["request_id"]] = time.time()
                out.append({"request_id": r["request_id"], "items": items, "model_version": self.version})
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
        jobs = [(2, self.refresh_events), (5, self.refresh_catalog), (60, self.refresh_bought),
                (300, self.refresh_popularity)]
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

    def refresh_bought(self) -> None:
        rows = query(BOUGHT, timeout=20)
        with self.lock:
            for r in rows:
                self.store.bought_from_us[r["u"]].add(r["p"])

    def refresh_popularity(self) -> None:
        tmp = Store.__new__(Store)
        tmp.subcat = self.store.subcat
        tmp._load_popularity(None, live=True)
        with self.lock:
            self.store.pop, self.store.top_global, self.store.top_by_subcat = tmp.pop, tmp.top_global, tmp.top_by_subcat

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
    ap.add_argument("--version", default="v1-lgbm", help="model_version written with each answer")
    ap.add_argument("--model", default=str(ROOT / "models" / "ranker.txt"))
    args = ap.parse_args()
    model = Path(args.model)
    if not model.exists() and (ROOT / "models" / "ranker_offline.txt").exists():
        model = ROOT / "models" / "ranker_offline.txt"
    Live(model, args.version, args.shadow).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
