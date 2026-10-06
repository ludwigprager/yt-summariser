# Notes for coding agents

Read `README.md` for the full picture. This file is the short list of
operational facts and rules that are easy to get wrong.

## Rules

- **Never run `yt-dlp`, `whisper` or `entrypoint.sh` directly on the host.**
  Everything runs inside the containers. Running yt-dlp here dumps
  half-downloaded video files into the repo root.
- The host has no `sqlite3` command. Touch the database through the `web`
  container (`docker compose exec web python -c ...`), or better, through the
  HTTP routes below.
- Code in `web/` is copied into the image, not mounted. After editing
  `web/app.py` or `web/templates/`, run `docker compose up -d --build web`.
  Restarting the container requeues any job that was running, which is fine.
- Don't delete `web-data/jobs.db`, `data/` or `whisper-cache/`.

## Jobs

Jobs live in `web-data/jobs.db` (inside the container: `/web-data/jobs.db`),
table `jobs`. Status goes `queued` → `running` → `done` or `error`.
`worker_loop()` in `web/app.py` polls for the oldest `queued` row every few
seconds, so **retrying a job just means setting it back to `queued`**.

### Retry failed jobs

The web UI is on port 8090 (nginx). From the host:

```bash
# one job
curl -X POST http://localhost:8090/jobs/<id>/retry
# every failed job
curl -X POST http://localhost:8090/jobs/retry-failed
```

Or in the UI: the "retry" button on a failed job's page, or "retry N failed"
next to the status filters.

### List failed jobs

```bash
docker compose exec web python -c "
import sqlite3; c = sqlite3.connect('/web-data/jobs.db')
for r in c.execute(\"SELECT id, url, substr(error, -300) FROM jobs WHERE status='error'\"): print(r, end='\n\n')"
```

Read the error before retrying. A retry only helps with temporary failures
(network, rate limits, a restart). It won't help when:

- the URL has no video, e.g. a YouTube community post (`youtube.com/post/...`)
  → delete the job instead;
- yt-dlp is too old for YouTube's current changes (`WARNING: [youtube] ...`
  followed by a download error) → rebuild the images so they pick up a new
  yt-dlp (`docker compose up -d --build`; the Dockerfiles refetch yt-dlp whenever PyPI has a newer release), *then* retry;
- the site isn't supported by yt-dlp at all.

### Waiting for GPU

Only when `CUSTODES_CONTAINER` is set (off by default; empty means jobs call
Ollama directly and this section doesn't apply).

Before calling Ollama, a job takes the lease `yt-summarize` in g1-custodes
(`docker exec custodes ...`, see `OllamaLease` in `web/app.py`). While another
stack holds the GPU (`custodes status`), jobs sit in stage "waiting for GPU";
that is not a hang. The lease is given back when no job is summarising.

### Watch progress

```bash
docker compose logs -f web
```
