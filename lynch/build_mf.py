"""Build the matrix-factorisation factors once from the 60-day history (immutable), using RawTree for the aggregation.

    .venv/bin/python datathon-fme-2026/lynch/build_mf.py      # -> lynch/models/mf_v8.npz (~12 MB)

Interaction weight per (user, product) = views + 3 x carts + 6 x purchases, aggregated in SQL; then log1p and a
truncated SVD (k = 48). Serving uses the item factors for every shopper (fold-in: mean factor of what they touched
live + their recent views) and the user factors only for shoppers present in the history.
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rawtree import query_parts  # noqa: E402

K = 48
OUT = Path(__file__).resolve().parent / "models" / "mf_v8.npz"

t0 = time.time()
df = query_parts("""
    SELECT user_id::String AS u, product_id::String AS p,
           toFloat64(countIf(event_type::String = 'view') + 3 * countIf(event_type::String = 'add_to_cart')
                     + 6 * countIf(event_type::String = 'purchase')) AS w
    FROM shop.history_events WHERE {part} GROUP BY u, p""", parts=8)
print(f"{len(df):,} user x product pairs from RawTree in {time.time() - t0:.0f}s", flush=True)
users, ui = np.unique(df.u.to_numpy(), return_inverse=True)
items, ii = np.unique(df.p.to_numpy(), return_inverse=True)
M = sp.csr_matrix((np.log1p(df.w.to_numpy(dtype=float)), (ui, ii)), shape=(len(users), len(items)))
U, S, Vt = svds(M, k=K)
OUT.parent.mkdir(exist_ok=True)
np.savez_compressed(OUT, users=users, items=items, U=(U * S).astype(np.float32), V=Vt.T.astype(np.float32))
print(f"saved {OUT.name}: {len(users):,} users, {len(items):,} items, k={K} ({time.time() - t0:.0f}s)")
