# DGM Leaderboard

Image generation competition leaderboard for the Deep Generative Models (DGM) course at UNIST.
Students form teams, upload ZIP files of generated face images, and are ranked on four evaluation metrics.

**Live leaderboard:** https://leaderboard.lait-lab.com/c/1

---

## System Overview

### Infrastructure

| Component | Location | Details |
|-----------|----------|---------|
| Leaderboard server | `bourborn` (10.20.26.126) | FastAPI + SQLite, Docker, port 8085 |
| n8n (email notifications) | `bourborn` | Docker, port 5678 |
| Reference images | `bourborn` bind-mounted | 32,550 CelebV-HQ frames at `/home/rlagkdus705/celebvhq_ref/` |
| Source videos | `potato` NAS | `/nas1/5000_Dataset_reform/Video/CelebV-HQ/processed/*.mp4` |
| Domain / reverse proxy | `leaderboard.lait-lab.com` | HTTPS, proxied to port 8085 |

Access chain: `local → potato (ProxyJump) → bourborn`

### Architecture

```
Student browser
    │  HTTPS upload ZIP
    ▼
FastAPI (main.py)
    │  saves ZIP to /app/data/uploads/{competition_id}/{team_id}/
    │  writes submission record to SQLite
    ▼
Async Queue (worker.py)
    │  extracts ZIP → flattens images to _flat/
    │  computes metrics using cached reference embeddings
    │  updates scores in DB
    │  calls n8n webhook → email notification
    ▼
SQLite (leaderboard.db)
    └─ competitions, teams, submissions, scores tables

n8n (Docker)
    └─ receives webhook → sends result email via Gmail SMTP
```

---

## Submission Pipeline

1. Student uploads a `.zip` containing up to **1,000 face images** (JPEG/PNG)
2. Worker extracts ZIP and flattens all images into a single directory
3. Each metric is computed against the **CelebV-HQ reference set** (32,550 images)
4. Pre-cached reference embeddings are loaded — no re-extraction of reference features per submission
5. Scores are written to DB; leaderboard updates immediately
6. n8n sends an email to team members who provided an email address at registration

Estimated processing time per submission: **~40–60 seconds** on GPU (A6000).

---

## Evaluation Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| **FID** | Lower ↓ | Fréchet Inception Distance — distribution similarity |
| **IS** | Higher ↑ | Inception Score — quality + diversity |
| **KID** | Lower ↓ | Kernel Inception Distance — distribution similarity (unbiased) |
| **TopPR** | Higher ↑ | Top-P & Top-R F1 — fidelity + diversity balance |
| **Total** | Lower ↓ | Average rank across all four metrics |

Reference: 32,550 middle frames extracted from CelebV-HQ (680×680, face-cropped), resized to 256×256 JPEG.

### Reference Feature Caching

Reference embeddings are pre-extracted once and reused for every submission:

| Cache file | Contents |
|------------|---------|
| `/app/data/references/cache/fid.pt` | InceptionV3 feature statistics (sum, cov, count) |
| `/app/data/references/cache/kid.pt` | InceptionV3 features tensor (32550 × 2048) |
| `/app/data/references/cache/toppr.npy` | InceptionV3 features numpy array (32550 × 2048) |

To regenerate cache (e.g. after changing reference images):
```bash
ssh bourborn
docker exec leaderboard python3 /app/precache_reference.py --competition_id 2
```

---

## Key Files

```
leaderboard/
├── main.py                  # FastAPI routes, session auth, file upload
├── worker.py                # Async submission queue, metric dispatch, n8n webhook
├── database.py              # SQLite helpers, leaderboard queries, avg-rank total
├── precache_reference.py    # One-shot script to pre-cache reference embeddings
├── metrics/
│   ├── base.py              # MetricBackend ABC, MetricResult dataclass
│   ├── registry.py          # Registers FID, IS, KID, TopPR
│   └── local.py             # PyTorch metric implementations + caching logic
├── templates/
│   ├── leaderboard.html     # Public scoreboard (Total + per-metric views)
│   ├── upload.html          # Submission upload page
│   ├── team.html            # Team detail + submission history
│   └── admin/              # Admin panel templates
├── docker-compose.yml       # Leaderboard + n8n services
├── Dockerfile               # Python 3.12, torch, torchmetrics, top-pr, torch-fidelity
└── .env                     # ADMIN_PASSWORD, SECRET_KEY, BASE_URL (not in git)
```

---

## Running & Maintenance

### Start / Stop
```bash
ssh bourborn
cd /home/rlagkdus705/leaderboard
docker compose up -d          # start all services
docker compose restart leaderboard  # restart after code changes
docker logs leaderboard -f    # live logs
```

### Deploy code changes
```bash
# For files in templates/ or static/ — live-mounted, no restart needed
# For .py files — must copy into container:
docker cp database.py leaderboard:/app/database.py
docker cp worker.py leaderboard:/app/worker.py
docker restart leaderboard
```

### n8n (email notifications)
- Running at `bourborn:5678` (internal only)
- Access via SSH tunnel: `ssh -L 5678:localhost:5678 bourborn -N` → `http://localhost:5678`
- Credentials: admin / admin1234
- Workflow: Webhook (`/webhook/submission-done`) → Send Email (Gmail SMTP)
- Activated — no action needed during competition

### Admin panel
- URL: https://leaderboard.lait-lab.com/admin
- Password: stored in `/home/rlagkdus705/leaderboard/.env` as `ADMIN_PASSWORD`
- Functions: create/edit competitions, manage teams, view/rerun/delete submissions

---

## Competition Settings (Spring 2026)

| Setting | Value |
|---------|-------|
| Name | DGM Spring 2026 — Face Generation Challenge |
| Dates | April 20 – June 10, 2026 |
| Invite code | `xUf8hxhX` |
| Submissions / day | 4 |
| Max images / ZIP | 1,000 |
| Max ZIP size | 200 MB |
| Primary metric | Total (average rank) |
| Reference set | CelebV-HQ, 32,550 frames |

---

## License

MIT
