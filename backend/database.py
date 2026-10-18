import sqlite3, json, os
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "fruitvision.db")

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                label TEXT,
                confidence REAL,
                probs TEXT,
                created_at TEXT
            )""")

def add_prediction(filename, label, confidence, probs):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO predictions (filename, label, confidence, probs, created_at) VALUES (?,?,?,?,?)",
            (filename, label, confidence, json.dumps(probs), datetime.now().isoformat(timespec="seconds")))
        return cur.lastrowid

def list_predictions(limit=50):
    with _conn() as c:
        rows = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r); d["probs"] = json.loads(d["probs"]); out.append(d)
    return out

def delete_prediction(pred_id):
    with _conn() as c:
        return c.execute("DELETE FROM predictions WHERE id=?", (pred_id,)).rowcount > 0

def stats():
    with _conn() as c:
        rows = c.execute("SELECT label, COUNT(*) n FROM predictions GROUP BY label").fetchall()
    return {r["label"]: r["n"] for r in rows}
