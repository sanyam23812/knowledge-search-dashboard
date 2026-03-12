# Break/Fix Log

This document records intentional failure scenarios introduced to test
system resilience, along with how each was diagnosed and fixed.

---

## Scenario A — Semantic Index Dimension Mismatch

### What was injected
Changed the embedding model name in `vector.py` from `all-MiniLM-L6-v2`
to `paraphrase-MiniLM-L3-v2` without rebuilding the FAISS index.
This causes a dimension mismatch on startup (384 vs 128 dimensions).

### How to reproduce
1. Build indexes normally with default model
2. Open `backend/app/search/vector.py`
3. Change `DEFAULT_MODEL = "all-MiniLM-L6-v2"` to `DEFAULT_MODEL = "paraphrase-MiniLM-L3-v2"`
4. Restart the API without rebuilding indexes

### What failed
API startup threw a RuntimeError:
```
Dimension mismatch! Index has 384 but model produces 128. Rebuild the index.
```
Search returned 503 for all queries.

### Fix applied
Added startup validation in `VectorIndex.load()` that reads saved
dimension from `meta.json` and compares it to the loaded model's
actual dimension. If they don't match, a clear error is raised
immediately with instructions to rebuild.

### How it was verified
- Reverted model name back to `all-MiniLM-L6-v2`
- Restarted API — loaded successfully
- `/health` returned 200, `/search` returned results normally

---

## Scenario B — SQLite Schema Migration Break

### What was injected
Added a `NOT NULL` column `user_id TEXT NOT NULL` to the
`request_logs` table in `db.py` without providing a default value
or running a migration.

### How to reproduce
1. Run the API and generate some logs (existing DB exists)
2. Add `user_id TEXT NOT NULL` to the CREATE TABLE statement in `db.py`
3. Restart the API — old DB rows have no user_id value

### What failed
Every search request threw an IntegrityError:
```
NOT NULL constraint failed: request_logs.user_id
```
Dashboard showed no logs. KPI page broke with empty data.

### Fix applied
- Removed the NOT NULL constraint (made it nullable)
- Added a safe migration check on startup using ALTER TABLE
- db.py now checks if column exists before adding it
- Old rows remain intact, new rows populate user_id as NULL safely

### How it was verified
- Restarted API with existing DB
- Old logs still visible in dashboard
- New search requests logged successfully with NULL user_id

---

## Scenario C — Hybrid Scoring Divide-by-Zero Bug

### What was injected
Temporarily replaced `minmax_norm` with a broken version that did
not handle the case where all scores are equal (max - min = 0),
causing a divide-by-zero and returning NaN scores.

### How to reproduce
1. Replace minmax_norm in `hybrid.py` with:
```python
def minmax_norm(scores):
    arr = np.array(scores)
    return ((arr - arr.min()) / (arr.max() - arr.min())).tolist()
```
2. Run a query where all BM25 scores are identical (e.g. very common word)

### What failed
- All hybrid scores returned as NaN
- Results ranked randomly
- Eval metrics dropped to 0.0
- pytest caught it with test_minmax_norm_all_equal

### Fix applied
Added a guard clause in both `minmax_norm` and `zscore_norm`:
```python
if mx - mn < 1e-9:
    return [0.0] * len(scores)
```
This safely returns zero scores instead of NaN when all values
are equal, preserving correct ranking behaviour.

### How it was verified
- pytest test_minmax_norm_all_equal passed
- Ran eval again — nDCG/Recall/MRR metrics recovered to normal values
- No NaN values in any search response