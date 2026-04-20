# 📢 DGM Spring 2026 — Face Generation Challenge

---

## Overview

As part of the Deep Generative Models course, you will participate in a **face image generation competition**.
Your goal is to train a generative model and submit **1,000 generated face images** to the leaderboard.
Teams are evaluated on four metrics and ranked by their **combined (average) rank** across all metrics.

**Leaderboard:** https://leaderboard.lait-lab.com

---

## Schedule

| | Date |
|--|------|
| **Start** | April 20, 2026 |
| **End** | June 10, 2026 |
| **Max submissions per day** | 4 |

---

## How to Join

This is an **individual competition** — each student participates as a solo team.

1. Visit https://leaderboard.lait-lab.com
2. Enter the following information:
   - **Invite code:** `xUf8hxhX`
   - **Team name:** use your own name in `Firstname_Lastname` format
     (e.g., `Hayeon_Kim`, `Jaejun_Yoo`)
   - **Display name:** same as your team name is fine
   - **Email** *(optional)*: if provided, you will receive an email with your scores after each submission
3. Click **Join** — you're in!

> **Re-login:** If you close the browser and come back, just enter the same invite code and your team name again — your session will be restored.

> **Team name is permanent.** You cannot change it after joining, so please type it carefully.

---

## Submission Format

- A single **`.zip`** file containing your generated images
- Images must be **JPEG or PNG** format
- Minimum resolution: **64×64** (higher is better — reference is 256×256)
- **Exactly 1,000 images** recommended (maximum allowed: 1,000)
- Maximum file size: **200 MB**
- Do **not** include subfolders inside the zip — place images directly at the root

### Example zip structure
```
submission.zip
├── img_0000.jpg
├── img_0001.jpg
├── ...
└── img_0999.jpg
```

---

## Evaluation Metrics

Your submission is compared against **32,550 real face images** from the CelebV-HQ dataset.

| Metric | Direction | What it measures |
|--------|-----------|-----------------|
| **FID** | Lower ↓ | Feature distribution distance (overall quality) |
| **IS** | Higher ↑ | Image quality and diversity (Inception Score) |
| **KID** | Lower ↓ | Kernel-based distribution distance (unbiased FID variant) |
| **TopPR** | Higher ↑ | Fidelity + diversity balance (Top-P & Top-R F1) |
| **Total** | Lower ↓ | **Average rank** across all four metrics — used for final ranking |

The **Total** rank is computed as the average of your rank on each individual metric.
Lower average rank = better overall performance.

---

## Tips

- You can view rankings per individual metric (FID / IS / KID / TopPR) using the dropdown on the leaderboard
- Scores update within ~1 minute of submission
- Only your **best score** per metric is shown on the leaderboard
- Previous ZIP files are automatically deleted — only your latest submission is stored
- Scores of **-1** on a metric indicate a processing error (usually too few images)

---

## FAQ

**Q: Can I submit solo?**
A: Yes. Just create a team with your own name.

**Q: Can I change my team?**
A: No — once you join a team, you cannot switch. Choose carefully.

**Q: What if my submission fails?**
A: Check the leaderboard for an error message under your submission. Common issues: wrong file format, fewer than 2 images, or file too large.

**Q: Do I need a GPU to generate images?**
A: For training, yes. The leaderboard server handles evaluation — you just need to submit a ZIP.

---

*For questions, contact the course TA.*
