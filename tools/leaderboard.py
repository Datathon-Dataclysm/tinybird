"""The public leaderboard, per team and per model_version, from the terminal.

    python tools/leaderboard.py            # last hour
    python tools/leaderboard.py 180        # last 3 hours (15 to 180 minutes)

Conversion here is the organisers' EXPECTED conversion (each shopper's probability of buying what we showed), so it
has no luck in it: the best way to compare versions. Compare versions that ran AT THE SAME TIME (A/B with
`live.py --ab`): trends and stock change during the day. Revenue per request only breaks ties.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

API = "https://datathon.rawtree.dev/api/state?by=model"
TEAM = "team_14"


def main() -> None:
    minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    start = int(time.time() - 60 * min(max(minutes, 15), 180)) // 300 * 300
    d = json.load(urllib.request.urlopen(f"{API}&from={start}", timeout=20))
    hhmm = lambda t: time.strftime("%H:%M", time.localtime(t))  # noqa: E731
    print(f"window {hhmm(d['window'][0])}-{hhmm(d['window'][1])} (local time)\n")
    print(f"{'':3}{'team':10}{'conv':>6}{'cold':>6}{'on time':>9}{'EUR/req':>9}{'EUR/buy':>9}  best model")
    rows = sorted(d["teams"], key=lambda t: -t["conversion"])
    for name, ref in [("reference", d.get("reference")), ("baseline", d.get("baseline"))]:
        if ref:
            rows.append({**ref, "team_id": name, "model_version": ""})
    for i, t in enumerate(rows[:12] + [t for t in rows[12:] if t["team_id"] in (TEAM, "reference", "baseline")]):
        mark = ">>" if t["team_id"] == TEAM else "  "
        buy = 100 * t["revenue_per_request"] / t["conversion"] if t["conversion"] else 0
        print(f"{mark} {t['team_id']:10}{t['conversion']:6.2f}{t['cold_conversion']:6.2f}{t['on_time']:8.1f}%"
              f"{t['revenue_per_request']:9.3f}{buy:9.1f}  {t.get('model_version', '')}")
    us = next((t for t in d["teams"] if t["team_id"] == TEAM), None)
    if us:
        print(f"\nour versions ({TEAM}):")
        print(f"   {'model_version':26}{'requests':>9}{'conv':>6}{'cold':>6}{'EUR/req':>9}{'EUR/buy':>9}  ran")
        for v in sorted(us["versions"], key=lambda v: -v["conversion"]):
            buy = 100 * v["revenue_per_request"] / v["conversion"] if v["conversion"] else 0
            grey = " (few requests)" if v["requests"] < 300 else ""
            print(f"   {v['model_version']:26}{v['requests']:9}{v['conversion']:6.2f}{v['cold_conversion']:6.2f}"
                  f"{v['revenue_per_request']:9.3f}{buy:9.1f}  {hhmm(v['first'])}-{hhmm(v['last'])}{grey}")


if __name__ == "__main__":
    main()
