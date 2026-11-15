"""Lynch (v8) live recommender. Copy of v7's starter/live.py; changes: SHADOW BY DEFAULT, v8 model + MF factors.

    python lynch/live.py                         # shadow (default): compute answers and measure, write NOTHING
    python lynch/live.py --submit                # production: write answers to team_NN.submissions

--- original v7 docstring ---
Live recommender: the personalised model answering shop.requests in real time.

Only ONE production process per team: if two processes answer the same request, only one answer counts.
Start it before the final round and do not restart it: retrained models are hot-reloaded (train_live.py --final).

Two threads:
- main loop (every 0.25 s): pull new requests -> recommend (in memory, milliseconds) -> push one batch
- refresher: keeps the in-memory store fresh without slowing the answers
    every 1 s   shop.catalog_updates -> new products (with their attributes), stock-outs (they reach the table ~2 s
                                        late and the organisers forgive the last 5 s, so 1 s polling keeps us inside)
    every 2 s   shop.events          -> user x product / subcategory / attribute tables (a cart a minute ago counts)
    every 30 s  model file           -> hot reload when train_live.py rewrites it
    every 60 s  team feedback        -> products each shopper clicked in our carousel
    every 60 s  trends               -> purchases per subcategory in the last 2 h vs usual
    every 5 min popularity           -> last 7 days (history + live)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Ranker, Recommender, Store  # noqa: E402
from rawtree import ROOT, TEAM, Record, be_first, insert, query, to_epoch  # noqa: E402

MODELS = Path(__file__).resolve().parent / "models"

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
       subcategory::String AS subcategory, is_active::Bool AS is_active, description::String AS description,
       toString(created_at::DateTime64(3)) AS created_at, toString(ts::DateTime64(3)) AS t
FROM shop.catalog_updates WHERE ts::DateTime64(3) > toDateTime64('{since}', 3) ORDER BY ts"""

CAROUSEL = f"""
SELECT r.user_id::String AS u, f.clicked::String AS p, count() AS c
FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.clicked::String != '' GROUP BY u, p"""


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Live:
    def __init__(self, model_path: Path, version: str, shadow: bool, slot: int | None = None, slots: int = 5,
                 salt: str = "spikes-v1") -> None:
        self.version, self.shadow, self.model_path = version, shadow, model_path
        self.slot, self.slots, self.salt = slot, slots, salt     # concurrent A/B: answer only our hash slot
        log("loading store (history + today's events), ~2 min ...")
        self.store = Store.load(live=True)
        log(f"MF factors: {'loaded' if self.store.load_mf(MODELS / 'mf_v8.npz') else 'MISSING (mf features = 0)'}")
        self.rec = Recommender(self.store, Ranker.load(model_path) if model_path.exists() else Ranker())
        self.model_mtime = model_path.stat().st_mtime if model_path.exists() else 0.0
        log(f"ranker: {model_path.name if model_path.exists() else 'hand-made score (no model file found)'}")
        self.lock = threading.Lock()              # the refresher mutates the store, the main loop reads it
        self.seen: dict[str, float] = {}          # request_id -> when we answered it
        self.recent_keys: deque = deque()         # event dedupe across overlapping polls
        self.recent_set: set = set()
        self.updates_since = "2026-01-01 00:00:00.000"
        self.stats = {"answered": 0, "invalid": 0, "latency": deque(maxlen=2000), "compute_ms": deque(maxlen=2000)}

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        threading.Thread(target=self.refresher, daemon=True).start()
        for r in query(REQUESTS, timeout=10):     # already past the deadline at start-up: the fallback served them
            self.seen[r["request_id"]] = time.time()
        last_report = time.time()
        while True:
            try:
                fresh = [r for r in query(REQUESTS, timeout=5) if r["request_id"] not in self.seen and self.mine(r["request_id"])]
                if fresh:
                    answers = self.answer(fresh)
                    if not self.shadow:
                        insert("submissions", answers, timeout=5)
                    now = time.time()
                    self.stats["latency"].extend(now - to_epoch(r["ts"]) - _frac(r["ts"]) for r in fresh)
                if time.time() - last_report > 60:
                    last_report = time.time()
                    self.report()
            except Exception as e:  # keep running: shoppers keep arriving while you sleep
                log(f"loop error: {e!r}")
                time.sleep(1)
            if len(self.seen) > 20000:
                cutoff = time.time() - 120
                self.seen = {k: v for k, v in self.seen.items() if v > cutoff}
            time.sleep(0.25)

    def mine(self, request_id: str) -> bool:
        """Same assignment as the codex spikes harness: sha256(salt:request_id)[:8] % slots == slot."""
        if self.slot is None:
            return True
        h = int.from_bytes(hashlib.sha256((self.salt + ":" + request_id).encode()).digest()[:8], "big")
        return h % self.slots == self.slot

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
                for p in items:                               # fatigue feature: times we showed it to this shopper
                    self.store.shown[r["user_id"]][p] += 1
                out.append({"request_id": r["request_id"], "items": items, "model_version": self.version})
        self.stats["answered"] += len(out)
        return out

    # ------------------------------------------------------------------ refresher thread
    def refresher(self) -> None:
        jobs = [(1, self.refresh_catalog), (2, self.refresh_events), (30, self.reload_model),
                (60, self.refresh_carousel), (60, self.refresh_trends), (300, self.refresh_popularity)]
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

    def refresh_catalog(self) -> None:
        rows = query(UPDATES.format(since=self.updates_since), timeout=10)
        if rows:
            with self.lock:
                for u in rows:
                    self.store.apply_catalog_update(u)
            self.updates_since = rows[-1]["t"]

    def refresh_events(self) -> None:
        rows = query(EVENTS.format(since=self.store.last_event_ts or "2026-11-14 00:00:00.000"), timeout=10)
        new = []
        for r in rows:
            key = (r["u"], r["p"], r["e"], r["t"])
            if key not in self.recent_set:
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

    def refresh_carousel(self) -> None:
        live: dict[str, dict[str, float]] = {}
        for r in query(CAROUSEL, timeout=20):
            live.setdefault(r["u"], {})[r["p"]] = float(r["c"])
        self.store.live_cs = live                    # swapped at once

    def refresh_trends(self) -> None:
        self.store.load_trends()

    def refresh_popularity(self) -> None:
        self.store.load_popularity(None, live=True)

    def reload_model(self) -> None:
        """Hot reload: when the model file changes, swap it in without restarting. A broken file is ignored."""
        if not self.model_path.exists() or self.model_path.stat().st_mtime == self.model_mtime:
            return
        self.model_mtime = self.model_path.stat().st_mtime
        self.rec.ranker = Ranker.load(self.model_path)
        log(f"reloaded model {self.model_path.name}")

    # ------------------------------------------------------------------ report
    def report(self) -> None:
        lat = np.array(self.stats["latency"] or [0.0])
        cm = np.array(self.stats["compute_ms"] or [0.0])
        log(f"{'SHADOW' if self.shadow else self.version}: answered {self.stats['answered']} "
            f"(invalid fixed {self.stats['invalid']}) | ready after p50 {np.median(lat):.2f}s p95 "
            f"{np.percentile(lat, 95):.2f}s | compute p50 {np.median(cm):.1f}ms p95 {np.percentile(cm, 95):.1f}ms")


def _frac(ts: str) -> float:
    """Fractional seconds of '... HH:MM:SS.fff' (to_epoch truncates them)."""
    return float("0." + ts.split(".")[1]) if "." in ts else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="WRITE answers to team_NN.submissions (default: shadow)")
    ap.add_argument("--version", default="lynch-v8", help="model_version written with each answer")
    ap.add_argument("--model", default=str(MODELS / "ranker_v8.txt"))
    ap.add_argument("--minutes", type=float, default=0, help="stop after N minutes (0 = run forever)")
    ap.add_argument("--slot", type=int, default=None, help="concurrent A/B: answer only requests in this hash slot")
    ap.add_argument("--slots", type=int, default=5, help="number of hash slots (must match the other harness)")
    ap.add_argument("--salt", default="spikes-v1", help="hash salt (must match the other harness)")
    args = ap.parse_args()
    if args.submit:
        be_first()                                   # answers before anything else running on this machine
    live = Live(Path(args.model), args.version, shadow=not args.submit, slot=args.slot, slots=args.slots, salt=args.salt)
    if args.minutes:
        threading.Timer(args.minutes * 60, lambda: (live.report(), __import__("os")._exit(0))).start()
    live.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
