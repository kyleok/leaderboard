"""
Leaderboard database — SQLite for standalone lecture tool.
"""
import sqlite3
import json
import secrets
import logging
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, date

logger = logging.getLogger(__name__)

DB_PATH = Path("/app/data/leaderboard.db")


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS competitions (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                invite_code TEXT UNIQUE,
                metrics TEXT NOT NULL DEFAULT '["fid"]',
                primary_metric TEXT DEFAULT 'fid',
                max_submissions_per_day INTEGER DEFAULT 5,
                max_images_per_zip INTEGER DEFAULT 1000,
                max_zip_size_mb INTEGER DEFAULT 500,
                start_date TEXT,
                end_date TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS reference_datasets (
                id INTEGER PRIMARY KEY,
                competition_id INTEGER REFERENCES competitions(id) ON DELETE CASCADE,
                file_path TEXT NOT NULL,
                num_images INTEGER,
                cached_features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY,
                competition_id INTEGER REFERENCES competitions(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                description TEXT,
                is_disqualified BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(competition_id, name)
            );

            CREATE TABLE IF NOT EXISTS team_members (
                id INTEGER PRIMARY KEY,
                team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
                display_name TEXT NOT NULL,
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(team_id, display_name)
            );

            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY,
                team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
                competition_id INTEGER REFERENCES competitions(id) ON DELETE CASCADE,
                file_path TEXT NOT NULL,
                num_images INTEGER,
                status TEXT DEFAULT 'queued',
                error_message TEXT,
                progress_current INTEGER DEFAULT 0,
                progress_total INTEGER DEFAULT 0,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                submitted_by TEXT
            );

            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY,
                submission_id INTEGER REFERENCES submissions(id) ON DELETE CASCADE,
                metric_name TEXT NOT NULL,
                score REAL NOT NULL,
                is_higher_better BOOLEAN DEFAULT 0,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(submission_id, metric_name)
            );
        """)
    logger.info(f"Database initialized at {DB_PATH}")


# --- Competition CRUD ---

def create_competition(name: str, description: str = "", metrics: list[str] = None,
                       primary_metric: str = "fid", max_submissions_per_day: int = 5,
                       max_images_per_zip: int = 1000, max_zip_size_mb: int = 500,
                       start_date: str = None, end_date: str = None,
                       invite_code: str = None) -> dict:
    metrics = metrics or ["fid"]
    invite_code = invite_code or secrets.token_urlsafe(6)
    with get_db() as db:
        cur = db.execute(
            """INSERT INTO competitions (name, description, invite_code, metrics, primary_metric,
               max_submissions_per_day, max_images_per_zip, max_zip_size_mb, start_date, end_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, description, invite_code, json.dumps(metrics), primary_metric,
             max_submissions_per_day, max_images_per_zip, max_zip_size_mb, start_date, end_date)
        )
        return get_competition(cur.lastrowid)


def get_competition(competition_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute("SELECT * FROM competitions WHERE id = ?", (competition_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metrics"] = json.loads(d["metrics"])
        return d


def get_competition_by_invite(invite_code: str) -> dict | None:
    with get_db() as db:
        row = db.execute("SELECT * FROM competitions WHERE invite_code = ?", (invite_code,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metrics"] = json.loads(d["metrics"])
        return d


def get_competitions(active_only: bool = False) -> list[dict]:
    with get_db() as db:
        q = "SELECT * FROM competitions"
        if active_only:
            q += " WHERE is_active = 1"
        q += " ORDER BY created_at DESC"
        rows = db.execute(q).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["metrics"] = json.loads(d["metrics"])
            result.append(d)
        return result


def update_competition(competition_id: int, **kwargs) -> dict | None:
    allowed = {"name", "description", "invite_code", "metrics", "primary_metric", "max_submissions_per_day",
               "max_images_per_zip", "max_zip_size_mb", "start_date", "end_date", "is_active"}
    updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not updates:
        return get_competition(competition_id)
    if "metrics" in updates and isinstance(updates["metrics"], list):
        updates["metrics"] = json.dumps(updates["metrics"])
    sets = ", ".join(f"{k} = ?" for k in updates)
    vals = list(updates.values()) + [competition_id]
    with get_db() as db:
        db.execute(f"UPDATE competitions SET {sets} WHERE id = ?", vals)
    return get_competition(competition_id)


# --- Reference Dataset ---

def set_reference_dataset(competition_id: int, file_path: str, num_images: int) -> dict:
    with get_db() as db:
        # Replace existing
        db.execute("DELETE FROM reference_datasets WHERE competition_id = ?", (competition_id,))
        cur = db.execute(
            "INSERT INTO reference_datasets (competition_id, file_path, num_images) VALUES (?, ?, ?)",
            (competition_id, file_path, num_images)
        )
        row = db.execute("SELECT * FROM reference_datasets WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def get_reference_dataset(competition_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM reference_datasets WHERE competition_id = ? ORDER BY id DESC LIMIT 1",
            (competition_id,)
        ).fetchone()
        return dict(row) if row else None


def update_cached_features(ref_id: int, cached_features_path: str):
    with get_db() as db:
        db.execute("UPDATE reference_datasets SET cached_features = ? WHERE id = ?",
                    (cached_features_path, ref_id))


# --- Team CRUD ---

def create_team(competition_id: int, name: str, description: str = "",
                creator_name: str = None) -> dict:
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO teams (competition_id, name, description) VALUES (?, ?, ?)",
            (competition_id, name, description)
        )
        team_id = cur.lastrowid
        if creator_name:
            db.execute(
                "INSERT INTO team_members (team_id, display_name) VALUES (?, ?)",
                (team_id, creator_name)
            )
    return get_team(team_id)


def get_team(team_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute("SELECT * FROM teams WHERE id = ?", (team_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        members = db.execute(
            "SELECT display_name, joined_at FROM team_members WHERE team_id = ?", (team_id,)
        ).fetchall()
        d["members"] = [dict(m) for m in members]
        return d


def get_teams(competition_id: int, include_disqualified: bool = False) -> list[dict]:
    with get_db() as db:
        q = "SELECT * FROM teams WHERE competition_id = ?"
        if not include_disqualified:
            q += " AND is_disqualified = 0"
        q += " ORDER BY created_at"
        rows = db.execute(q, (competition_id,)).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            members = db.execute(
                "SELECT display_name, joined_at FROM team_members WHERE team_id = ?", (d["id"],)
            ).fetchall()
            d["members"] = [dict(m) for m in members]
            result.append(d)
        return result


def join_team(team_id: int, display_name: str) -> bool:
    with get_db() as db:
        try:
            db.execute(
                "INSERT INTO team_members (team_id, display_name) VALUES (?, ?)",
                (team_id, display_name)
            )
            return True
        except sqlite3.IntegrityError:
            return False  # already a member


def get_user_team(competition_id: int, display_name: str) -> dict | None:
    """Get the team a user belongs to in a competition."""
    with get_db() as db:
        row = db.execute(
            """SELECT t.* FROM teams t
               JOIN team_members tm ON t.id = tm.team_id
               WHERE t.competition_id = ? AND tm.display_name = ?""",
            (competition_id, display_name)
        ).fetchone()
        if not row:
            return None
        return get_team(row["id"])


def disqualify_team(team_id: int, disqualify: bool = True):
    with get_db() as db:
        db.execute("UPDATE teams SET is_disqualified = ? WHERE id = ?", (int(disqualify), team_id))


# --- Submission CRUD ---

def create_submission(team_id: int, competition_id: int, file_path: str,
                      num_images: int, submitted_by: str) -> dict:
    with get_db() as db:
        cur = db.execute(
            """INSERT INTO submissions (team_id, competition_id, file_path, num_images, submitted_by)
               VALUES (?, ?, ?, ?, ?)""",
            (team_id, competition_id, file_path, num_images, submitted_by)
        )
        return get_submission(cur.lastrowid)


def get_submission(submission_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        scores = db.execute(
            "SELECT metric_name, score, is_higher_better FROM scores WHERE submission_id = ?",
            (submission_id,)
        ).fetchall()
        d["scores"] = {s["metric_name"]: {"score": s["score"], "is_higher_better": bool(s["is_higher_better"])}
                       for s in scores}
        return d


def get_team_submissions(team_id: int, competition_id: int) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            """SELECT * FROM submissions WHERE team_id = ? AND competition_id = ?
               ORDER BY submitted_at DESC""",
            (team_id, competition_id)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            scores = db.execute(
                "SELECT metric_name, score, is_higher_better FROM scores WHERE submission_id = ?",
                (d["id"],)
            ).fetchall()
            d["scores"] = {s["metric_name"]: {"score": s["score"], "is_higher_better": bool(s["is_higher_better"])}
                           for s in scores}
            result.append(d)
        return result


def get_all_submissions(competition_id: int) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            """SELECT s.*, t.name as team_name FROM submissions s
               JOIN teams t ON s.team_id = t.id
               WHERE s.competition_id = ? ORDER BY s.submitted_at DESC""",
            (competition_id,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            scores = db.execute(
                "SELECT metric_name, score, is_higher_better FROM scores WHERE submission_id = ?",
                (d["id"],)
            ).fetchall()
            d["scores"] = {s["metric_name"]: {"score": s["score"], "is_higher_better": bool(s["is_higher_better"])}
                           for s in scores}
            result.append(d)
        return result


def update_submission_status(submission_id: int, status: str, error_message: str = None,
                             progress_current: int = None, progress_total: int = None):
    with get_db() as db:
        updates = ["status = ?"]
        vals = [status]
        if error_message is not None:
            updates.append("error_message = ?")
            vals.append(error_message)
        if progress_current is not None:
            updates.append("progress_current = ?")
            vals.append(progress_current)
        if progress_total is not None:
            updates.append("progress_total = ?")
            vals.append(progress_total)
        if status == "done" or status == "failed":
            updates.append("processed_at = ?")
            vals.append(datetime.utcnow().isoformat())
        vals.append(submission_id)
        db.execute(f"UPDATE submissions SET {', '.join(updates)} WHERE id = ?", vals)


def save_score(submission_id: int, metric_name: str, score: float, is_higher_better: bool = False):
    with get_db() as db:
        db.execute(
            """INSERT OR REPLACE INTO scores (submission_id, metric_name, score, is_higher_better)
               VALUES (?, ?, ?, ?)""",
            (submission_id, metric_name, score, int(is_higher_better))
        )


def count_team_submissions_today(team_id: int, competition_id: int) -> int:
    today = date.today().isoformat()
    with get_db() as db:
        row = db.execute(
            """SELECT COUNT(*) as cnt FROM submissions
               WHERE team_id = ? AND competition_id = ? AND DATE(submitted_at) = ?""",
            (team_id, competition_id, today)
        ).fetchone()
        return row["cnt"]


# --- Leaderboard ---

def get_leaderboard(competition_id: int, metric_name: str = "fid") -> list[dict]:
    """Get leaderboard sorted by best score per team for a given metric."""
    with get_db() as db:
        # Get metric direction
        row = db.execute(
            "SELECT is_higher_better FROM scores WHERE metric_name = ? LIMIT 1", (metric_name,)
        ).fetchone()
        is_higher_better = bool(row["is_higher_better"]) if row else False
        agg = "MAX" if is_higher_better else "MIN"

        rows = db.execute(f"""
            SELECT t.id as team_id, t.name as team_name, t.is_disqualified,
                   {agg}(sc.score) as best_score,
                   COUNT(DISTINCT s.id) as num_submissions,
                   MAX(s.submitted_at) as last_submission
            FROM teams t
            JOIN submissions s ON s.team_id = t.id AND s.competition_id = ? AND s.status = 'done'
            JOIN scores sc ON sc.submission_id = s.id AND sc.metric_name = ?
            WHERE t.competition_id = ? AND t.is_disqualified = 0
            GROUP BY t.id
            ORDER BY best_score {"DESC" if is_higher_better else "ASC"}
        """, (competition_id, metric_name, competition_id)).fetchall()

        return [dict(r) for r in rows]


def get_queue_position(submission_id: int) -> int:
    """Get position in the processing queue (0 = currently processing)."""
    with get_db() as db:
        rows = db.execute(
            "SELECT id FROM submissions WHERE status IN ('queued', 'processing') ORDER BY submitted_at"
        ).fetchall()
        for i, row in enumerate(rows):
            if row["id"] == submission_id:
                return i
        return -1


def delete_submission(submission_id: int):
    with get_db() as db:
        db.execute("DELETE FROM scores WHERE submission_id = ?", (submission_id,))
        db.execute("DELETE FROM submissions WHERE id = ?", (submission_id,))
