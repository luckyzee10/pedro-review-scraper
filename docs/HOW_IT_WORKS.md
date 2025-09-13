How The Movie Review Bot Works

Overview
- Polls film RSS feeds on a fixed interval.
- Detects new review headlines/snippets and classifies sentiment via OpenAI.
- Persists each review to SQLite and keeps rolling per‑movie sentiment counts.
- Sends a Telegram message for every new review.

Components
- scraper.py
  - Uses feedparser to read each RSS URL in `FEEDS`.
  - Filters entries with a light heuristic (requires the word “review” in title/summary).
  - Outputs items with outlet, title, summary, link, published.
- sentiment.py
  - Calls OpenAI Chat Completions API with a strict instruction to return exactly: Positive, Negative, or Neutral.
  - Returns Neutral on errors/timeouts/unexpected output.
- telegram.py
  - Sends messages using python‑telegram‑bot when available; falls back to Telegram HTTP API otherwise.
  - Works with private DMs, groups, or channels.
- main.py
  - Loads env via dotenv; initializes SQLite.
  - Polls feeds every `POLL_SECONDS` (default 120s).
  - De‑dupes via DB before spending on OpenAI.
  - Extracts an approximate movie title from the headline/summary.
  - Classifies sentiment, stores row, updates in‑memory counts, and notifies Telegram.
  - Handles simple Telegram commands via `getUpdates` (see below).

Data Flow
1) Fetch all feeds → list of entries.
2) Filter: review candidates only.
3) Duplicate check: `SELECT 1 FROM reviews WHERE outlet=? AND headline=?`.
4) Movie title extraction heuristics (quoted phrases, “Review:” patterns, separators).
5) Sentiment classification via OpenAI (temperature 0, tiny max tokens).
6) Insert into SQLite: `reviews(outlet, movie, headline, sentiment, timestamp)`.
7) Update in‑memory aggregate counts for that movie.
8) Send Telegram message with current per‑movie tally.
9) Include a Rotten Tomatoes‑like freshness percentage: Positive / (Positive + Negative).

De‑duplication
- Enforced by a unique index on `(outlet, headline)`.
- Prevents reprocessing/renotifying even across restarts (as long as the DB persists).

Aggregation
- On startup, the app loads counts from SQLite into memory.
- Each new review increments the appropriate sentiment count for its movie.

Tomatometer‑like Percentage
- Definition
  - The bot computes a simple “freshness” percentage inspired by the Tomatometer:
    Fresh% = Positive / (Positive + Negative) * 100, rounded to the nearest integer.
  - Neutral reviews are ignored in the percentage (they neither help nor hurt).
  - A minimum of 3 (Positive+Negative) reviews is required before showing a percentage; otherwise it shows “N/A”.
  - You can adjust the threshold via env var `MIN_REVIEWS_FOR_PERCENT` (default 3).
- Delivery
  - The Telegram message adds a line: `Tomatometer-like: <XX%|N/A> Fresh`.
- Rationale
  - Mirrors the idea of “fresh” vs “rotten” while keeping the model’s Neutral outcome from skewing the score.

Database Schema
- Table: `reviews`
  - `id INTEGER PRIMARY KEY`
  - `outlet TEXT`
  - `movie TEXT`
  - `headline TEXT`
  - `sentiment TEXT` (Positive|Negative|Neutral)
  - `timestamp TEXT` (UTC ISO‑8601)
- Unique Index: `ON reviews(outlet, headline)`

Configuration
- Required env vars
  - `OPENAI_API_KEY`
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID` (user, group, or channel ID)
- Optional
  - `REVIEW_DB_PATH` (default `reviews.db`; in Docker we use `/data/reviews.db`)
  - `POLL_SECONDS` (default `120`)

Runtime Behavior
- Loop interval: sleeps the remainder of `POLL_SECONDS` after each cycle.
- Graceful stop: Ctrl‑C (SIGINT) sets a flag and exits after the current cycle.

Error Handling & Resilience
- Feed parsing errors are caught per‑feed; the loop continues.
- OpenAI failures fall back to Neutral to keep the pipeline moving.
- Telegram send uses python‑telegram‑bot if present; else raw HTTP API.

Rate/Load Considerations
- First run on a fresh DB may process many “new” items and send multiple messages.
- Each new item triggers one OpenAI classification request; consider costs and rate limits.
- You can reduce feeds, increase `POLL_SECONDS`, or add stricter filters in `scraper.py`.

Customization
- Feeds: edit `FEEDS` in `scraper.py`.
- Filter strictness: tweak `_is_review_candidate`.
- Movie title heuristic: adjust `extract_movie_title` in `main.py`.
- Model selection: change `model` in `sentiment.classify_sentiment`.
- Message format: adjust `format_message` in `main.py`.
 - URL filter: set env `STRICT_URL_REVIEW=1` to only accept items whose URL contains the word "review". This reduces noise but may exclude genuine reviews from outlets without "review" in their URLs.

Security
- Treat `OPENAI_API_KEY` and `TELEGRAM_BOT_TOKEN` as secrets. Rotate if leaked.
- Prefer managed secret stores (Render env vars, etc.).

Operations Runbook
- Logs: standard output shows cycle counts and send failures.
- DB: persisted at `REVIEW_DB_PATH` (on Render, `/data/reviews.db`).
- Restart: redeploy or restart the worker on your cloud platform.
- Common issues
  - No Telegram messages: check token, chat ID, bot membership/permissions.
  - “chat not found”: wrong ID or bot not in the chat.
  - Empty first cycles: feeds may not have fresh items; wait for new posts.

Telegram Delivery
- Library and fallback
  - `telegram.py` tries `python-telegram-bot` first. It supports both modern async and legacy sync Bot clients.
  - If the library is missing or fails, it falls back to Telegram’s HTTPS Bot API via `requests`.
- Required configuration
  - `TELEGRAM_BOT_TOKEN`: from BotFather for your bot.
  - `TELEGRAM_CHAT_ID`: where to send messages.
    - Private DM: positive numeric ID (from @userinfobot).
    - Group/Supergroup: negative numeric ID (often starting with `-100`, but can be other negatives as well, e.g. `-4809244001`).
    - Channel: either negative numeric ID or `@channelusername` if the bot is an admin.
- Membership and permissions
  - The bot must be a member of the group to send messages; for channels, the bot must be an admin with post permission.
  - Privacy mode (BotFather /setprivacy) affects receiving group messages, not sending. This app only needs to send.
- Message format (exactly what gets delivered)
  - Line 1: `🎬 New Review from {outlet}`
  - Line 2: `"{headline}" → {sentiment}`
  - Line 3: `Aggregate Sentiment for {movie}: {pos}P / {neg}N / {neu}M`
  - Link previews are disabled to keep messages compact.
- Sending behavior
  - One Telegram message per newly detected review.
  - Duplicate suppression via DB ensures no repeat messages for the same (outlet, headline).
  - Commands supported: `/status` or `/movies` in any chat where the bot is present. For groups, `/status@YourBotUsername` also works.
- Testing and troubleshooting
  - Token check: `curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"` → expect `{ "ok": true }`.
  - Send test: `curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" -d chat_id="$TELEGRAM_CHAT_ID" -d text="Hello"`.
  - Common errors:
    - `Forbidden: bot is not a member of the chat` → add the bot to the group/channel.
    - `Bad Request: chat not found` → wrong ID or using a user ID for a group.
    - `not enough rights to send text messages` → grant the bot permission or admin role.
- Rate and volume
  - Telegram allows generous per‑bot rates, but chats may throttle bursts. First run on a fresh DB may send many messages; mitigate by increasing `POLL_SECONDS`, limiting feeds, or warming up with a private DM/chat before switching to a group.

Telegram Commands (Status Summary)
- What it does
  - When someone sends `/status` or `/movies`, the bot replies with a summary of up to 100 movies, sorted by proximity to future release dates (soonest upcoming first), then unknown dates, then past releases. Each line shows counts and a Tomatometer‑like percentage.
  - `
    /status <movie name>
    /movies <movie name>
    ` returns a single movie’s stats (best match by title).
- How it works
  - The worker checks Telegram `getUpdates` between feed polling cycles.
  - It stores an offset in SQLite table `kv` (key `tg_offset`) to avoid reprocessing the same updates.
  - It responds to `message` or `channel_post` updates containing plain text commands.
- Output format
  - First line: `📊 Movies Summary (Top 100):`
  - Per line: `- Movie Name: 3P/1N/2M • 75% • 6 reviews`
- Batching
  - Telegram messages are limited to ~4096 chars; the bot splits long summaries into multiple messages.
  - If there are no reviews yet, it replies: `No reviews yet. Check back soon!`

Data Hygiene Commands
- `/backfill`
  - Populates missing release dates for all known movies via TMDb (if `TMDB_API_KEY` is set). Safe to run repeatedly.
- `/normalize` or `/normalize guardian`
  - Rewrites stored movie titles that are raw URLs into cleaner titles, with Guardian-specific handling for slugs like `…/movie-name-review-…`.
  - `/normalize all` will attempt URL-based normalization for all domains, not just Guardian.

Release Dates & Sorting
- Where release dates come from
  - The app attempts to look up release dates via TMDb when a new movie title appears and caches the result in SQLite table `movies`.
  - Set `TMDB_API_KEY` to enable lookups. Without it, release dates remain unknown and summaries fall back to a generic order.
- Sorting rule for summaries
  - Future releases (today or later): ordered from soonest → farthest.
  - Unknown release date: shown after all known future releases.
  - Past releases: shown last.
