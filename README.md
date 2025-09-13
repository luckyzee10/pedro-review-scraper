Movie Review Sentiment Notifier (Cloud Ready)

What this does
- Polls a set of movie/film RSS feeds every 2 minutes
- Classifies each new headline/snippet with OpenAI (Positive/Negative/Neutral)
- Stores results in SQLite (reviews table)
- Sends Telegram alerts with rolling per-movie sentiment counts
- Adds a simple Tomatometer-like percentage: Positive/(Positive+Negative)

Environment variables
- OPENAI_API_KEY
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
- TMDB_API_KEY (optional, enables movie release date lookups for better sorting)
- REVIEW_DB_PATH (optional, defaults to reviews.db; use /data/reviews.db in containers)

Quick local run
1) Python 3.11, then: pip install -r requirements.txt
2) Create .env with the variables above
3) python main.py

Cloud deployment (Render – easiest managed option)
Render gives you a “worker” with a persistent disk in a few clicks.

1) Push this repo to your GitHub account.
2) Create a new Render Blueprint from this repo (render.yaml provided).
3) When prompted, set environment variables:
   - OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
   - Optional: TMDB_API_KEY for release date lookups
   - REVIEW_DB_PATH is pre-set by the blueprint to /data/reviews.db
4) Deploy. The worker will start polling and persist the SQLite DB on the mounted disk.

Cloud deployment (Docker anywhere: Fly.io, Railway, your VPS)
We provide a Dockerfile. You only need to pass env vars and mount a volume for the DB.

Build
  docker build -t movie-review-scraper:latest .

Run locally via Docker
  docker run --rm -it \
    -e OPENAI_API_KEY=... \
    -e TELEGRAM_BOT_TOKEN=... \
    -e TELEGRAM_CHAT_ID=... \
    -e REVIEW_DB_PATH=/data/reviews.db \
    -v $(pwd)/data:/data \
    movie-review-scraper:latest

Fly.io (example)
  flyctl launch --no-deploy  # create app
  flyctl volumes create data --size 1 --region <your-region>
  flyctl secrets set OPENAI_API_KEY=... TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
  flyctl deploy
  flyctl scale count 1
  flyctl secrets set REVIEW_DB_PATH=/data/reviews.db
  # Ensure your fly.toml mounts the volume at /data (see docs)

VPS with systemd (Ubuntu example)
1) Create a small VM (1 vCPU/512MB+). SSH in.
2) Install Python 3.11 and git. Clone repo.
3) python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
4) Create .env with keys. Optionally set REVIEW_DB_PATH=/opt/reviews/reviews.db
5) Create a systemd unit (as root) at /etc/systemd/system/review-notifier.service:

  [Unit]
  Description=Movie Review Notifier
  After=network.target

  [Service]
  WorkingDirectory=/opt/movie-review
  Environment=PYTHONUNBUFFERED=1
  EnvironmentFile=/opt/movie-review/.env
  ExecStart=/opt/movie-review/.venv/bin/python -u main.py
  Restart=always
  RestartSec=5

  [Install]
  WantedBy=multi-user.target

6) sudo systemctl daemon-reload && sudo systemctl enable --now review-notifier
7) Check logs: sudo journalctl -u review-notifier -f

Notes
- Poll interval is 120s by default. To change, set POLL_SECONDS.
- Duplicate avoidance uses (outlet, headline) unique index. With a persistent DB, restarts won’t re-alert prior items.
- Movie title extraction is heuristic; tune in main.extract_movie_title if needed.
- Freshness: Neutral reviews are excluded from the percentage denominator; shows N/A until at least one Positive or Negative exists.
 - Minimum reviews for percentage: requires at least 3 (Positive+Negative) reviews to show a percentage; configure with `MIN_REVIEWS_FOR_PERCENT`.

 Learn more
- See docs/HOW_IT_WORKS.md for a concise overview of architecture, data flow, configuration, and operations.
