"""Tiny RawTree client shared by the live recommender, training and the notebook.

Reads TEAM_ID and RAWTREE_API_KEY from the environment or from .env at the project root.
"""
from __future__ import annotations

import calendar
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
for _line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    _key, _, _value = _line.partition("=")
    os.environ.setdefault(_key.strip(), _value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"

Record = dict[str, Any]


def query(sql: str, timeout: float = 600) -> list[Record]:
    """Run SQL and return the rows as dicts."""
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"RawTree {r.status_code}: {r.text[:500]}")
    return r.json()["data"]


def query_df(sql: str, timeout: float = 600) -> pd.DataFrame:
    return pd.DataFrame(query(sql, timeout))


def query_parts(sql: str, parts: int, key: str = "user_id::String") -> pd.DataFrame:
    """Large results: RawTree caps a response at 100 MB, so split by hash of `key`.

    `sql` must contain a `{part}` placeholder inside its WHERE clause.
    """
    frames = [query_df(sql.format(part=f"cityHash64({key}) % {parts} = {k}")) for k in range(parts)]
    return pd.concat(frames, ignore_index=True)


def insert(table: str, rows: list[Record], timeout: float = 10) -> None:
    """Insert JSON rows into team_NN.<table>."""
    r = http.post(f"{API}/v1/tables/{table}", params={"database": TEAM}, json=rows, timeout=timeout)
    r.raise_for_status()


def to_epoch(ts: str) -> float:
    """'2026-11-14 11:46:02.371000000' (UTC) -> unix seconds. Fast enough to call per row."""
    return calendar.timegm(time.strptime(ts[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S"))
