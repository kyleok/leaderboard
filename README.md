# Leaderboard

Image generation competition leaderboard for university lectures. Students form teams, upload zips of generated images, and compete on metrics like FID, IS, LPIPS, and KID.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/kyleok/leaderboard.git
cd leaderboard

# 2. Configure
cp .env.example .env
# Edit .env — set ADMIN_PASSWORD and SECRET_KEY

# 3. Run
docker compose up -d --build

# 4. Open http://localhost:8085
```

## How It Works

**For instructors:**
1. Go to `/admin/login` → enter admin password
2. Create a competition → set metrics, upload reference images
3. Share the auto-generated **invite code** with students

**For students:**
1. Visit the leaderboard URL → scoreboard is public
2. Enter invite code + team name + display name → joined
3. Upload zip of generated images → scores appear on the board

## Features

- **Public scoreboard** — no login needed to view rankings
- **Invite code auth** — instructor shares code, students join in seconds
- **Team-based** — create or join teams, track submissions per team
- **Configurable metrics** — FID, IS (Inception Score), LPIPS, KID
- **GPU-accelerated** — runs on NVIDIA GPU if available, falls back to CPU
- **Rate limiting** — configurable max submissions per team per day
- **Admin panel** — create competitions, upload reference data, manage teams, re-run scoring
- **Real-time updates** — leaderboard auto-refreshes every 10 seconds

## GPU Support

For faster metric computation, install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and uncomment `runtime: nvidia` in `docker-compose.yml`.

Without GPU, metrics run on CPU (slower but functional).

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ADMIN_PASSWORD` | `admin` | Password for `/admin` panel |
| `SECRET_KEY` | `change-me` | Secret for signing session cookies |
| `BASE_URL` | `http://localhost:8085` | Public-facing URL |

## Architecture

- **Backend:** FastAPI + SQLite (zero external dependencies)
- **Frontend:** Jinja2 + vanilla JS, Cafe au LAIT design system
- **Metrics:** PyTorch-based, modular `MetricBackend` system
- **Processing:** Async background worker queue
- **Auth:** Cookie-based sessions, no external auth service needed

## Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| FID | Lower is better | Frechet Inception Distance |
| IS | Higher is better | Inception Score |
| LPIPS | Lower is better | Learned Perceptual Image Patch Similarity |
| KID | Lower is better | Kernel Inception Distance |

## License

MIT
