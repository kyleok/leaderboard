"""
LAIT Leaderboard — Image generation competition scoreboard.
Auth: competition invite code → team creation/join → cookie session.
"""
import hashlib
import hmac
import json
import logging
import os
import shutil
import zipfile
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import database as db
from metrics.registry import create_default_registry
from worker import SubmissionWorker, UPLOAD_DIR, REFERENCE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = os.getenv("BASE_URL", "https://leaderboard.lait-lab.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
SECRET_KEY = os.getenv("SECRET_KEY", "leaderboard-default-secret-change-me")

registry = create_default_registry()
worker = SubmissionWorker(registry)


# --- Cookie signing ---

def _sign(value: str) -> str:
    sig = hmac.new(SECRET_KEY.encode(), value.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{value}.{sig}"


def _unsign(signed: str) -> str | None:
    if "." not in signed:
        return None
    value, sig = signed.rsplit(".", 1)
    expected = hmac.new(SECRET_KEY.encode(), value.encode(), hashlib.sha256).hexdigest()[:16]
    if hmac.compare_digest(sig, expected):
        return value
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    worker.start()
    logger.info("Leaderboard service started")
    yield
    await worker.stop()
    logger.info("Leaderboard service stopped")


app = FastAPI(title="LAIT Leaderboard", lifespan=lifespan)
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# --- Auth helpers ---

def get_display_name(request: Request) -> str | None:
    cookie = request.cookies.get("leaderboard_user")
    if not cookie:
        return None
    return _unsign(cookie)


def is_admin(request: Request) -> bool:
    cookie = request.cookies.get("leaderboard_admin")
    if not cookie:
        return False
    return _unsign(cookie) == "admin"


def require_admin(request: Request) -> str:
    if not is_admin(request):
        accept = request.headers.get("accept", "")
        if request.url.path.startswith("/api/") or "application/json" in accept:
            raise HTTPException(status_code=403, detail="Admin access required")
        raise HTTPException(status_code=403, detail="Admin access required")
    return get_display_name(request) or "Admin"


# --- Page Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Scoreboard is the homepage."""
    competitions = db.get_competitions(active_only=True)
    if len(competitions) == 1:
        return RedirectResponse(f"/c/{competitions[0]['id']}", status_code=302)
    if len(competitions) == 0:
        name = get_display_name(request)
        return templates.TemplateResponse(request, "leaderboard.html", {
            "user": {"display_name": name, "is_admin": is_admin(request)} if name else None,
            "competition": None,
            "leaderboard": [],
            "my_team": None,
            "primary_metric": None,
            "available_metrics": registry.list_metrics(),
        })
    # Multiple competitions — show list (unlikely but handle it)
    name = get_display_name(request)
    return templates.TemplateResponse(request, "leaderboard.html", {
        "user": {"display_name": name, "is_admin": is_admin(request)} if name else None,
        "competitions": competitions,
        "competition": None,
        "leaderboard": [],
        "my_team": None,
        "primary_metric": None,
        "available_metrics": registry.list_metrics(),
    })


@app.get("/c/{competition_id}", response_class=HTMLResponse)
async def competition_leaderboard(request: Request, competition_id: int):
    """Public scoreboard — no auth needed to view."""
    name = get_display_name(request)
    competition = db.get_competition(competition_id)
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")

    primary = competition.get("primary_metric", competition["metrics"][0])
    leaderboard = db.get_leaderboard(competition_id, primary)
    my_team = db.get_user_team(competition_id, name) if name else None

    return templates.TemplateResponse(request, "leaderboard.html", {
        "user": {"display_name": name, "is_admin": is_admin(request)} if name else None,
        "competition": competition,
        "leaderboard": leaderboard,
        "my_team": my_team,
        "primary_metric": primary,
        "available_metrics": registry.list_metrics(),
    })


@app.get("/c/{competition_id}/team/{team_id}", response_class=HTMLResponse)
async def team_detail(request: Request, competition_id: int, team_id: int):
    name = get_display_name(request)
    competition = db.get_competition(competition_id)
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    team = db.get_team(team_id)
    if not team or team["competition_id"] != competition_id:
        raise HTTPException(status_code=404, detail="Team not found")
    submissions = db.get_team_submissions(team_id, competition_id)
    my_team = db.get_user_team(competition_id, name) if name else None
    is_member = my_team and my_team["id"] == team_id

    return templates.TemplateResponse(request, "team.html", {
        "user": {"display_name": name, "is_admin": is_admin(request)} if name else None,
        "competition": competition,
        "team": team,
        "submissions": submissions,
        "is_member": is_member,
    })


@app.get("/c/{competition_id}/upload", response_class=HTMLResponse)
async def upload_page(request: Request, competition_id: int):
    name = get_display_name(request)
    if not name:
        return RedirectResponse(f"/c/{competition_id}", status_code=302)
    competition = db.get_competition(competition_id)
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    my_team = db.get_user_team(competition_id, name)
    if not my_team:
        return RedirectResponse(f"/c/{competition_id}", status_code=302)
    today_count = db.count_team_submissions_today(my_team["id"], competition_id)

    return templates.TemplateResponse(request, "upload.html", {
        "user": {"display_name": name, "is_admin": is_admin(request)},
        "competition": competition,
        "team": my_team,
        "today_count": today_count,
    })


@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("leaderboard_user")
    response.delete_cookie("leaderboard_admin")
    return response


# --- API: Join via invite code ---

@app.post("/api/join")
async def api_join(request: Request):
    """Join a competition: validate invite code, create/join team, set session cookie."""
    data = await request.json()
    invite_code = data.get("invite_code", "").strip()
    team_name = data.get("team_name", "").strip()
    display_name = data.get("display_name", "").strip()

    if not invite_code:
        raise HTTPException(status_code=400, detail="Invite code required")
    if not team_name:
        raise HTTPException(status_code=400, detail="Team name required")
    if not display_name or len(display_name) > 50:
        raise HTTPException(status_code=400, detail="Display name required (max 50 chars)")
    if len(team_name) > 50:
        raise HTTPException(status_code=400, detail="Team name too long (max 50)")

    # Validate invite code against competition
    competition = db.get_competition_by_invite(invite_code)
    if not competition:
        raise HTTPException(status_code=404, detail="Invalid invite code")
    if not competition["is_active"]:
        raise HTTPException(status_code=400, detail="Competition is not active")

    competition_id = competition["id"]

    # Check if user already in a team
    existing = db.get_user_team(competition_id, display_name)
    if existing:
        # Already in a team — just set cookie and return
        response = JSONResponse({"team": existing, "competition_id": competition_id})
        response.set_cookie("leaderboard_user", _sign(display_name),
                            max_age=60 * 60 * 24 * 30, httponly=True, samesite="lax")
        return response

    # Try to find existing team with this name
    teams = db.get_teams(competition_id)
    existing_team = next((t for t in teams if t["name"] == team_name), None)

    if existing_team:
        # Join existing team
        ok = db.join_team(existing_team["id"], display_name)
        if not ok:
            raise HTTPException(status_code=400, detail="Already a member of this team")
        team = db.get_team(existing_team["id"])
    else:
        # Create new team
        try:
            team = db.create_team(competition_id, team_name, "", display_name)
        except Exception as e:
            if "UNIQUE" in str(e):
                raise HTTPException(status_code=400, detail="Team name already taken")
            raise

    response = JSONResponse({"team": team, "competition_id": competition_id})
    response.set_cookie("leaderboard_user", _sign(display_name),
                        max_age=60 * 60 * 24 * 30, httponly=True, samesite="lax")
    return response


# --- API: Submissions ---

@app.post("/api/submissions")
async def api_upload_submission(
    request: Request,
    competition_id: int = Form(...),
    file: UploadFile = File(...),
):
    name = get_display_name(request)
    if not name:
        raise HTTPException(status_code=401, detail="Not authenticated")

    competition = db.get_competition(competition_id)
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    if not competition["is_active"]:
        raise HTTPException(status_code=400, detail="Competition is not active")

    today = date.today().isoformat()
    if competition.get("start_date") and today < competition["start_date"]:
        raise HTTPException(status_code=400, detail="Competition has not started yet")
    if competition.get("end_date") and today > competition["end_date"]:
        raise HTTPException(status_code=400, detail="Competition has ended")

    my_team = db.get_user_team(competition_id, name)
    if not my_team:
        raise HTTPException(status_code=400, detail="You must join a team first")
    if my_team.get("is_disqualified"):
        raise HTTPException(status_code=400, detail="Your team has been disqualified")

    today_count = db.count_team_submissions_today(my_team["id"], competition_id)
    if today_count >= competition["max_submissions_per_day"]:
        raise HTTPException(status_code=429,
                            detail=f"Rate limit: {competition['max_submissions_per_day']} submissions/day")

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")

    upload_dir = UPLOAD_DIR / str(competition_id) / str(my_team["id"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = competition["max_zip_size_mb"] * 1024 * 1024
    file_path = upload_dir / f"{date.today().isoformat()}_{today_count + 1}.zip"
    total_written = 0
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                total_written += len(chunk)
                if total_written > max_bytes:
                    f.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=400,
                                        detail=f"File too large (max {competition['max_zip_size_mb']}MB)")
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)[:100]}")

    try:
        num_images = _count_images_in_zip(file_path)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid zip: {str(e)[:100]}")

    if num_images == 0:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No valid images found in zip")
    if num_images > competition["max_images_per_zip"]:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400,
                            detail=f"Too many images ({num_images}, max {competition['max_images_per_zip']})")

    submission = db.create_submission(
        team_id=my_team["id"], competition_id=competition_id,
        file_path=str(file_path), num_images=num_images, submitted_by=name,
    )
    await worker.enqueue(submission["id"])
    queue_pos = db.get_queue_position(submission["id"])
    return {"submission": submission, "queue_position": queue_pos}


def _count_images_in_zip(zip_path: Path) -> int:
    supported = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX") or "/." in name:
                continue
            ext = Path(name).suffix.lower()
            if ext in supported and not name.endswith("/"):
                count += 1
    return count


# --- API: Leaderboard ---

@app.get("/api/leaderboard/{competition_id}")
async def api_leaderboard(competition_id: int, metric: str = None):
    competition = db.get_competition(competition_id)
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    metric = metric or competition.get("primary_metric", competition["metrics"][0])
    leaderboard = db.get_leaderboard(competition_id, metric)
    return {"leaderboard": leaderboard, "metric": metric}


@app.get("/api/submissions/{submission_id}/status")
async def api_submission_status(submission_id: int):
    submission = db.get_submission(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    queue_pos = db.get_queue_position(submission_id) if submission["status"] in ("queued", "processing") else -1
    return {
        "status": submission["status"],
        "progress_current": submission["progress_current"],
        "progress_total": submission["progress_total"],
        "queue_position": queue_pos,
        "scores": submission.get("scores", {}),
        "error_message": submission.get("error_message"),
    }


# --- Admin Pages ---

@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request, redirect: str = "/admin"):
    if is_admin(request):
        return RedirectResponse(redirect, status_code=302)
    return templates.TemplateResponse(request, "admin/login.html", {"redirect": redirect})


@app.post("/admin/login")
async def admin_login_submit(request: Request):
    form = await request.form()
    password = form.get("password", "")
    redirect = form.get("redirect", "/admin")
    if not hmac.compare_digest(password, ADMIN_PASSWORD):
        return templates.TemplateResponse(request, "admin/login.html", {
            "redirect": redirect, "error": "Incorrect password.",
        })
    display_name = get_display_name(request) or "Admin"
    response = RedirectResponse(redirect, status_code=302)
    response.set_cookie("leaderboard_admin", _sign("admin"),
                        max_age=60 * 60 * 24 * 7, httponly=True, samesite="lax")
    if not get_display_name(request):
        response.set_cookie("leaderboard_user", _sign(display_name),
                            max_age=60 * 60 * 24 * 30, httponly=True, samesite="lax")
    return response


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    if not is_admin(request):
        return RedirectResponse("/admin/login?redirect=/admin", status_code=302)
    name = get_display_name(request) or "Admin"
    competitions = db.get_competitions()
    return templates.TemplateResponse(request, "admin/dashboard.html", {
        "user": {"display_name": name, "is_admin": True},
        "competitions": competitions,
        "available_metrics": registry.list_metrics(),
    })


@app.get("/admin/competition/new", response_class=HTMLResponse)
async def admin_new_competition(request: Request, user: str = Depends(require_admin)):
    return templates.TemplateResponse(request, "admin/competition.html", {
        "user": {"display_name": user, "is_admin": True},
        "competition": None,
        "available_metrics": registry.list_metrics(),
    })


@app.get("/admin/competition/{competition_id}", response_class=HTMLResponse)
async def admin_edit_competition(request: Request, competition_id: int,
                                 user: str = Depends(require_admin)):
    competition = db.get_competition(competition_id)
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    ref = db.get_reference_dataset(competition_id)
    return templates.TemplateResponse(request, "admin/competition.html", {
        "user": {"display_name": user, "is_admin": True},
        "competition": competition,
        "reference": ref,
        "available_metrics": registry.list_metrics(),
    })


@app.get("/admin/competition/{competition_id}/submissions", response_class=HTMLResponse)
async def admin_submissions(request: Request, competition_id: int,
                            user: str = Depends(require_admin)):
    competition = db.get_competition(competition_id)
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    submissions = db.get_all_submissions(competition_id)
    teams = db.get_teams(competition_id, include_disqualified=True)
    return templates.TemplateResponse(request, "admin/submissions.html", {
        "user": {"display_name": user, "is_admin": True},
        "competition": competition,
        "submissions": submissions,
        "teams": teams,
    })


# --- Admin API ---

@app.post("/api/admin/competitions")
async def api_create_competition(request: Request, user: str = Depends(require_admin)):
    data = await request.json()
    comp = db.create_competition(
        name=data["name"],
        description=data.get("description", ""),
        metrics=data.get("metrics", ["fid"]),
        primary_metric=data.get("primary_metric", "fid"),
        max_submissions_per_day=data.get("max_submissions_per_day", 5),
        max_images_per_zip=data.get("max_images_per_zip", 1000),
        max_zip_size_mb=data.get("max_zip_size_mb", 500),
        start_date=data.get("start_date"),
        end_date=data.get("end_date"),
        invite_code=data.get("invite_code"),
    )
    return comp


@app.put("/api/admin/competitions/{competition_id}")
async def api_update_competition(competition_id: int, request: Request,
                                 user: str = Depends(require_admin)):
    data = await request.json()
    comp = db.update_competition(competition_id, **data)
    if not comp:
        raise HTTPException(status_code=404, detail="Competition not found")
    return comp


@app.post("/api/admin/competitions/{competition_id}/reference")
async def api_upload_reference(request: Request, competition_id: int,
                               file: UploadFile = File(...),
                               user: str = Depends(require_admin)):
    competition = db.get_competition(competition_id)
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files accepted")

    ref_dir = REFERENCE_DIR / str(competition_id)
    ref_dir.mkdir(parents=True, exist_ok=True)
    for old in ref_dir.iterdir():
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()

    zip_path = ref_dir / "reference.zip"
    with open(zip_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    images_dir = ref_dir / "images"
    images_dir.mkdir(exist_ok=True)
    supported = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    num_images = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.filename.startswith("/") or ".." in info.filename:
                continue
            if info.filename.startswith("__MACOSX") or "/." in info.filename:
                continue
            ext = Path(info.filename).suffix.lower()
            if ext in supported and not info.is_dir():
                data = zf.read(info.filename)
                dest = images_dir / f"{num_images:06d}{ext}"
                dest.write_bytes(data)
                num_images += 1
    zip_path.unlink()
    ref = db.set_reference_dataset(competition_id, str(images_dir), num_images)
    return {"reference": ref, "num_images": num_images}


@app.post("/api/admin/teams/{team_id}/disqualify")
async def api_disqualify_team(team_id: int, request: Request,
                              user: str = Depends(require_admin)):
    data = await request.json()
    db.disqualify_team(team_id, data.get("disqualify", True))
    return {"status": "ok"}


@app.post("/api/admin/submissions/{submission_id}/rerun")
async def api_rerun_submission(submission_id: int, user: str = Depends(require_admin)):
    submission = db.get_submission(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    db.update_submission_status(submission_id, "queued")
    await worker.enqueue(submission_id)
    return {"status": "queued"}


@app.delete("/api/admin/submissions/{submission_id}")
async def api_delete_submission(submission_id: int, user: str = Depends(require_admin)):
    submission = db.get_submission(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    Path(submission["file_path"]).unlink(missing_ok=True)
    db.delete_submission(submission_id)
    return {"status": "deleted"}


@app.get("/api/admin/submissions/{submission_id}/download")
async def api_download_submission(submission_id: int, user: str = Depends(require_admin)):
    submission = db.get_submission(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    file_path = Path(submission["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(file_path, filename=file_path.name, media_type="application/zip")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "leaderboard"}
