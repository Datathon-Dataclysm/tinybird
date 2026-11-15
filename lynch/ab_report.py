"""Compare the versions of an A/B that ran AT THE SAME TIME, from the public leaderboard API (expected conversion).

    python lynch/ab_report.py 02:45 06:45                  # local times; the end defaults to now
    python lynch/ab_report.py 02:45 06:45 --base lynch-v8  # differences against this version (default: the first)

The window is cut into 15-minute pieces (the API minimum). Totals are request-weighted. The uncertainty of each
difference comes from how much it varies between pieces (paired by time, so drift during the night cancels out):
a difference smaller than its +-2 SE is not a real difference. No-cold = all requests minus the cold ones.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

import numpy as np

API = "https://datathon.rawtree.dev/api/state?by=model"
TEAM = "team_14"
PIECE = 15 * 60


def local(hhmm: str) -> int:
    """The latest moment at or before now with this local hh:mm, rounded down to the API's 5-minute step."""
    h, m = map(int, hhmm.split(":"))
    now = time.localtime()
    t = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, h, m, 0, 0, 0, -1))
    return int(t - 86400 if t > time.time() else t) // 300 * 300


def piece(start: int) -> dict[str, dict]:
    """Our versions in [start, start + 15 min): {version: {n, nc, conv, cold, warm, eur}}."""
    d = json.load(urllib.request.urlopen(f"{API}&from={start}&to={start + PIECE}", timeout=30))
    us = next((t for t in d["teams"] if t["team_id"] == TEAM), {"versions": []})
    out = {}
    for v in us["versions"]:
        n, nc = v["requests"], v["cold_requests"]
        out[v["model_version"]] = {"n": n, "nc": nc, "conv": v["conversion"], "cold": v["cold_conversion"],
                                   "warm": (v["conversion"] * n - v["cold_conversion"] * nc) / max(n - nc, 1),
                                   "eur": v["revenue_per_request"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end", nargs="?", default=time.strftime("%H:%M"))
    ap.add_argument("--base", default=None, help="version the others are compared with (default: the first)")
    args = ap.parse_args()
    a, b = local(args.start), local(args.end)
    starts = list(range(a, b - PIECE + 1, PIECE))
    pieces = []
    for s in starts:
        pieces.append(piece(s))
        print(f"\r  read {len(pieces)}/{len(starts)} pieces", end="", flush=True)
    print()
    versions = sorted({v for p in pieces for v in p})
    if not versions:
        raise SystemExit("no requests answered by our versions in that window")
    base = args.base or versions[0]
    hhmm = lambda t: time.strftime("%H:%M", time.localtime(t))  # noqa: E731
    print(f"window {hhmm(a)}-{hhmm(starts[-1] + PIECE)} ({len(pieces)} pieces of 15 min)\n")
    print(f"{'version':14}{'requests':>9}{'cold':>7}{'conv':>7}{'cold':>7}{'no-cold':>9}{'EUR/req':>9}")
    total = {}
    for v in versions:
        rows = [p[v] for p in pieces if v in p]
        n, nc = sum(r["n"] for r in rows), sum(r["nc"] for r in rows)
        total[v] = {"conv": sum(r["conv"] * r["n"] for r in rows) / n,
                    "cold": sum(r["cold"] * r["nc"] for r in rows) / max(nc, 1),
                    "warm": sum(r["warm"] * (r["n"] - r["nc"]) for r in rows) / max(n - nc, 1),
                    "eur": sum(r["eur"] * r["n"] for r in rows) / n}
        t = total[v]
        print(f"{v:14}{n:9}{nc:7}{t['conv']:7.2f}{t['cold']:7.2f}{t['warm']:9.2f}{t['eur']:9.3f}")
    print(f"\ndifference vs {base} (+- 2 SE from the 15-minute pieces):")
    for v in versions:
        if v == base:
            continue
        both = [p for p in pieces if v in p and base in p]
        parts = []
        for key, name in [("conv", "conv"), ("cold", "cold"), ("warm", "no-cold"), ("eur", "EUR/req")]:
            d = np.array([p[v][key] - p[base][key] for p in both])
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            parts.append(f"{name} {total[v][key] - total[base][key]:+.3f} +- {2 * se:.3f}")
        print(f"  {v:14}" + " | ".join(parts) + f"   ({len(both)} pieces with both)")


if __name__ == "__main__":
    main()
