"""Live recommender: the personalised model answering shop.requests in real time.

    python starter/live.py --version v7-name     # production (supervised: restarted by itself if it dies or hangs)
    python starter/live.py --shadow              # dry run: compute answers and measure, write NOTHING

Reliability (10 points, and every late answer is served the fallback):
- supervisor: the command above runs a small supervisor that starts the real process and restarts it if it crashes or
  stops beating for 60 s. Ctrl+C stops everything.
- handover: to deploy a new version just START IT (another terminal) and leave the old one running. When the new one
  has loaded (~2.5 min) it takes over, and the old one notices within a second and exits by itself. No gap.
- answers are sent by a pool of sender threads, so a slow insert never delays reading the next requests.
- everything is also written to logs/live_<version>.log (to find out afterwards what happened).
Only ONE production process per team (the handover enforces it on this machine; not across machines).

Threads of the live process:
- main loop (every 0.1 s): pull new requests -> recommend (in memory, milliseconds) -> hand the batch to a sender
- senders: insert the batch into team_NN.submissions (retried if it fails)
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
import os
import subprocess
import sys
import threading
import time
import traceback
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Ranker, Recommender, Store  # noqa: E402
from rawtree import ROOT, TEAM, Record, be_first, insert, new_session, query, to_epoch  # noqa: E402

POLL_S = 0.1                       # pause between two reads of shop.requests (a read itself takes ~0.2 s)
SENDERS = 4                        # parallel insert threads
HEARTBEAT_TIMEOUT_S = 60           # supervisor restarts the process if it stops beating for this long
LOGS = ROOT / "logs"
REHEARSAL = os.environ.get("LIVE_REHEARSAL") == "1"   # tests: production behaviour, but nothing is inserted
OWNER = LOGS / ("live_owner_rehearsal.txt" if REHEARSAL else "live_owner.txt")   # who owns the stream now

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
       toString(created_at::DateTime64(3)) AS created_at, price_eur::Float64 AS price_eur,
       toString(ts::DateTime64(3)) AS t
FROM shop.catalog_updates WHERE ts::DateTime64(3) > toDateTime64('{since}', 3) ORDER BY ts"""

CAROUSEL = f"""
SELECT r.user_id::String AS u, f.clicked::String AS p, count() AS c
FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.clicked::String != '' GROUP BY u, p"""

_log_file: Path | None = None


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    if _log_file is not None:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
    print(line, flush=True)


def quiet_console() -> None:
    """Windows: turn off the console's QuickEdit mode. With it on, clicking inside the terminal window FREEZES the
    program at its next print until you press a key: in production that is a silent outage."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.GetStdHandle(-10)                     # console input
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(handle, ctypes.byref(mode)):
            k32.SetConsoleMode(handle, (mode.value & ~0x0040) | 0x0080)   # - QUICK_EDIT, + EXTENDED_FLAGS
    except Exception:
        pass


def server_clock_offset() -> float:
    """Local clock minus the RawTree clock, in seconds (latencies are measured on the organisers' clock)."""
    t0 = time.time()
    ms = int(query("SELECT toUnixTimestamp64Milli(now64(3)) AS ms", timeout=5)[0]["ms"])
    return (t0 + time.time()) / 2 - ms / 1000


def _ts(ts: str) -> float:
    """'2026-11-14 11:46:02.371000000' (UTC) -> unix seconds with the fraction."""
    return to_epoch(ts) + (float("0." + ts.split(".")[1]) if "." in ts else 0.0)


class Live:
    def __init__(self, model_path: Path, arms: list[tuple[str, float]], shadow: bool) -> None:
        self.version = " / ".join(a for a, _ in arms)
        self.shadow, self.model_path = shadow, model_path
        log("loading store (history + today's events), ~2.5 min ...")
        self.store = Store.load(live=True)
        ranker = Ranker.load(model_path) if model_path.exists() else Ranker()
        # A/B: every arm shares the store and the model; each request goes to one arm by a hash of its id
        self.arms = [(name, Recommender(self.store, ranker, price_tilt=tilt)) for name, tilt in arms]
        self.rec = self.arms[0][1]
        if len(arms) > 1 or arms[0][1]:
            log("arms: " + ", ".join(f"{n} (price tilt {t:g})" for n, t in arms))
        self.model_mtime = model_path.stat().st_mtime if model_path.exists() else 0.0
        log(f"ranker: {model_path.name if model_path.exists() else 'hand-made score (no model file found)'}")
        self.lock = threading.Lock()              # the refresher mutates the store, the main loop reads it
        self.stats_lock = threading.Lock()
        self.seen: dict[str, float] = {}          # request_id -> when we answered it
        self.recent_keys: deque = deque()         # event dedupe across overlapping polls
        self.recent_set: set = set()
        self.updates_since = "2026-01-01 00:00:00.000"
        self.stats = {"answered": 0, "invalid": 0, "lost": 0, "latency": deque(maxlen=3000),
                      "compute_ms": deque(maxlen=3000)}
        self.clock = server_clock_offset()
        self.token = f"{os.getpid()}-{time.time():.3f}"
        self.heartbeat = LOGS / f"heartbeat_{os.getpid()}.txt"
        self.pool = ThreadPoolExecutor(max_workers=SENDERS, thread_name_prefix="sender")
        self.tls = threading.local()

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        threading.Thread(target=self.refresher, daemon=True).start()
        for r in query(REQUESTS, timeout=10):     # already past the deadline at start-up: the fallback served them
            self.seen[r["request_id"]] = time.time()
        if not self.shadow:
            if not REHEARSAL:
                be_first()                        # high priority only once loaded (loading must not slow the old one)
            OWNER.write_text(self.token)          # take over: an older production process will now step aside
            log(f"{self.version}: answering (clock offset vs server {self.clock:+.2f}s)")
        last_report = last_check = time.time()
        while True:
            try:
                fresh = [r for r in query(REQUESTS, timeout=5) if r["request_id"] not in self.seen]
                if fresh:
                    answers = self.answer(fresh)
                    tss = [_ts(r["ts"]) for r in fresh]
                    if self.shadow:
                        self.record_latency(tss)
                    else:
                        self.pool.submit(self.send, answers, tss)
                if time.time() - last_report > 60:
                    last_report = time.time()
                    self.report()
            except Exception as e:  # keep running: shoppers keep arriving while you sleep
                log(f"loop error: {e!r}")
                time.sleep(0.5)
            if not self.shadow and time.time() - last_check > 1:   # outside the try: beats even if RawTree is down
                last_check = time.time()
                try:
                    self.heartbeat.write_text(str(last_check))
                    newer = OWNER.read_text().strip() != self.token
                except OSError:
                    newer = False
                if newer:
                    log("handover: a newer production process is ready -> this one stops now")
                    self.pool.shutdown(wait=True)
                    return
            if len(self.seen) > 20000:
                cutoff = time.time() - 120
                self.seen = {k: v for k, v in self.seen.items() if v > cutoff}
            time.sleep(POLL_S)

    def answer(self, reqs: list[Record]) -> list[Record]:
        out = []
        with self.lock:
            for r in reqs:
                t = time.perf_counter()
                version, rec = self.arms[zlib.crc32(r["request_id"].encode()) % len(self.arms)]
                items = rec.recommend(r)
                self.stats["compute_ms"].append(1000 * (time.perf_counter() - t))
                if len(items) != 10 or len(set(items)) != 10 or any(p in self.store.unavailable for p in items):
                    self.stats["invalid"] += 1
                    items = self.rec.fill([], 10)            # never send an invalid list
                self.seen[r["request_id"]] = time.time()
                for p in items:                               # fatigue feature: times we showed it to this shopper
                    self.store.shown[r["user_id"]][p] += 1
                out.append({"request_id": r["request_id"], "items": items, "model_version": version})
        self.stats["answered"] += len(out)
        return out

    def send(self, answers: list[Record], tss: list[float]) -> None:
        """Sender thread: insert one batch; retry while it can still arrive in time."""
        if not hasattr(self.tls, "session"):
            self.tls.session = new_session()
        for attempt in range(3):
            try:
                if REHEARSAL:
                    time.sleep(0.15)                          # what an insert takes
                else:
                    insert("submissions", answers, timeout=3, session=self.tls.session)
                self.record_latency(tss)
                return
            except Exception as e:
                log(f"insert failed (attempt {attempt + 1}/3, {len(answers)} answers): {e!r}")
                if time.time() - self.clock - min(tss) > 2.8:   # too late anyway
                    break
                time.sleep(0.05)
        with self.stats_lock:
            self.stats["lost"] += len(answers)

    def record_latency(self, tss: list[float]) -> None:
        """Seconds from the request ts to our answer being stored (sent, or computed in shadow), server clock."""
        now = time.time() - self.clock
        with self.stats_lock:
            self.stats["latency"].extend(now - t for t in tss)

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
        ranker = Ranker.load(self.model_path)
        for _, rec in self.arms:
            rec.ranker = ranker
        log(f"reloaded model {self.model_path.name}")

    # ------------------------------------------------------------------ report
    def report(self) -> None:
        with self.stats_lock:
            lat = np.array(self.stats["latency"] or [0.0])
            lost = self.stats["lost"]
        cm = np.array(self.stats["compute_ms"] or [0.0])
        log(f"{'SHADOW' if self.shadow else self.version}: answered {self.stats['answered']} "
            f"(invalid fixed {self.stats['invalid']}, lost {lost}) | "
            f"{'computed' if self.shadow else 'stored'} after p50 {np.median(lat):.2f}s p95 {np.percentile(lat, 95):.2f}s "
            f"(>2.5s: {100 * (lat > 2.5).mean():.1f}%) | compute p50 {np.median(cm):.1f}ms p95 {np.percentile(cm, 95):.1f}ms")


# ---------------------------------------------------------------------- supervisor
def supervise(argv: list[str], version: str) -> None:
    """Run the live process as a child; restart it if it crashes or stops beating. Exit when it ends normally
    (Ctrl+C, or handover to a newer version)."""
    restarts = 0
    while True:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *argv, "--child"])
        log(f"supervisor (pid {os.getpid()}): started {version}, live process pid {child.pid}"
            + (f" (restart #{restarts})" if restarts else ""))
        beat = LOGS / f"heartbeat_{child.pid}.txt"
        try:
            while child.poll() is None:
                time.sleep(2)
                if beat.exists() and time.time() - beat.stat().st_mtime > HEARTBEAT_TIMEOUT_S:
                    log(f"supervisor: no heartbeat for {HEARTBEAT_TIMEOUT_S}s -> killing and restarting")
                    child.kill()
                    child.wait()
        except KeyboardInterrupt:
            log("supervisor: Ctrl+C -> stopping")
            try:
                child.wait(timeout=10)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                child.kill()
            return
        finally:
            beat.unlink(missing_ok=True)
        if child.returncode == 0:
            log("supervisor: live process ended normally (handover or Ctrl+C)")
            return
        restarts += 1
        log(f"supervisor: live process died (exit code {child.returncode}) -> restarting in 3 s")
        time.sleep(3)


def main() -> None:
    global _log_file
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", action="store_true", help="do not write submissions, only measure")
    ap.add_argument("--version", default="v7", help="model_version written with each answer")
    ap.add_argument("--price-tilt", type=float, default=0.0, help="rank by P(buy) x price^tilt (0 = conversion)")
    ap.add_argument("--ab", default="", help='A/B arms "name=tilt,name=tilt": traffic split evenly by request id')
    ap.add_argument("--model", default=str(ROOT / "models" / "ranker_v7.txt"))
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)   # started by the supervisor
    args = ap.parse_args()
    arms = ([(a.split("=")[0].strip(), float(a.split("=")[1])) for a in args.ab.split(",")] if args.ab
            else [(args.version, args.price_tilt)])
    name = "+".join(n for n, _ in arms)
    LOGS.mkdir(exist_ok=True)
    _log_file = LOGS / f"live_{'shadow' if args.shadow else name}.log"
    quiet_console()
    if args.shadow or args.child:
        Live(Path(args.model), arms, args.shadow).run()
    else:
        supervise([a for a in sys.argv[1:] if a != "--child"], name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
