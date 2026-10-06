# yt-summarise

Takes a YouTube video, downloads the audio, transcribes it, and produces a
bullet-point summary. Two ways to use it: a one-shot CLI script, or a
long-running web UI that queues jobs.

A video walks through the repo: <https://youtu.be/0bTe_J2ip74>

This is a personal setup, not a polished product. Expect to adapt it to
your environment (hosts, GPU, models), and let a coding assistant such as
ChatGPT or Claude do that work: point it at this repo, this README and
`AGENTS.md`, and describe your setup.

> **Trusted networks only.** The web UI has no authentication: anyone who
> can reach port 8090 can queue jobs. With `CUSTODES_CONTAINER` set, the
> web container also mounts the docker socket, which is root-equivalent
> access to the host. Don't expose it to the internet.

## Setup

You need Docker with Compose and an [Ollama](https://ollama.com) server
that has the summary model pulled (`ollama pull qwen3.8:27b`, or pick
another model with `SUMMARY_MODEL`).

```
cp .env.example .env    # set OLLAMA_HOST unless Ollama runs on this host
docker compose up -d
open http://localhost:8090
```

Settings, all optional, in `.env`:

| Variable | Default | What |
|---|---|---|
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | Ollama used for summarizing; the default is Ollama on the docker host |
| `SUMMARY_MODEL` | `qwen3.8:27b` | default model |
| `OLLAMA_NUM_CTX` | `32768` | context window asked of Ollama, see below |
| `HETZNER_INFERENCE_API_KEY` | empty | enables `hetzner:<model>` |
| `LAYA_HOST` | empty | topic tagging with laya; off when empty |
| `AI_GATEWAY_API_KEY` | empty | also tag with Jev via Vercel AI Gateway |
| `CUSTODES_CONTAINER` | empty | wait for a GPU lease from custodes before calling Ollama |
| `WHISPER_MODEL_SIZE`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE` | `medium`, `cpu`, `int8` | speech-to-text |
| `STT_MAX_CONCURRENT` | `2` | parallel transcriptions |

## Pipeline

Every run — CLI or web — goes through the same three steps, all driven by
[`entrypoint.sh`](entrypoint.sh):

1. **Download** — [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) pulls the
   audio track (converted to mp3) and the video's metadata (title, video ID).
2. **Transcribe (STT)** — [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
   (a CTranslate2 reimplementation of OpenAI's Whisper) transcribes the audio
   **locally, on CPU**, by default with the `medium` model at `int8`
   quantization. This runs inside the same container as the download step —
   nothing is sent anywhere for STT. Model weights (~1.5GB, pulled from
   Hugging Face on first use) are cached in `whisper-cache/` so they're only
   downloaded once, not on every run.
3. **Summarize (LLM)** — the transcript is sent to one of two backends,
   never run locally. Which one is decided entirely by the `SUMMARY_MODEL`
   value (see `call_llm()` in `entrypoint.sh`):
   - **No prefix** → [Ollama](https://ollama.com)'s `/api/generate`, at
     `OLLAMA_HOST` (default: Ollama on the docker host; any machine with a
     GPU running Ollama works — this container only talks to its HTTP
     port, sharing no Docker network or other resource with it). Model defaults to
     `qwen3.8:27b`. The request carries an explicit `num_ctx`
     (`OLLAMA_NUM_CTX`, default 32768) because Ollama allots the prompt
     only **half** of `num_ctx` and, past that, drops tokens from the
     *front* without erroring — at the server's own 32768 default a
     ~38000-token transcript silently lost most of its length before the
     model saw it. Sized at roughly 2× the longest transcript seen here;
     lower it if that GPU's VRAM is wanted elsewhere, since the KV cache
     scales with it.
   - **`hetzner:<model>` prefix** (e.g. `hetzner:Qwen3.8-27B`) →
     [Hetzner's hosted inference](https://docs.hetzner.com/general/company-and-policy/experiments/inference/),
     an OpenAI-compatible `/chat/completions` endpoint — a *different*
     request/response shape than Ollama's native API, which is why
     `call_llm()` branches on this prefix rather than just swapping a URL.
     Requires `HETZNER_INFERENCE_API_KEY` (see `.env.example`); free while
     experimental, rate-limited to 10 requests/min. The prefix is stripped
     before the request goes upstream — it only exists to route within
     this app, mirroring the same trick the `assistant` stack's own Open
     WebUI config uses to disambiguate its Hetzner-hosted `Qwen3.8-27B`
     from the *different*, locally-hosted `qwen3.8:27b` model.

   Every summarization also gets a second `call_llm()` pass — translating
   the summary to German — using the same backend/model as the first call.

Neither backend ever sees the prompt as a command-line argument: a
transcript easily exceeds `MAX_ARG_STRLEN` (128KB, the kernel's per-argument
cap), so `call_llm()` hands the prompt to `jq` as a `--rawfile` and the
assembled body to `curl` as `--data-binary @file`. Both prompts also put
their instruction line *after* the transcript rather than before it, so that
front-truncation eats transcript instead of the task.

Both `OLLAMA_HOST` and every `WHISPER_*` setting are plain env vars
(`docker-compose.yaml`), so STT and summarization can each be pointed at a
different device/host independently — e.g. `WHISPER_DEVICE=cuda` to
transcribe on a local GPU instead, if one's available.

## Two ways to run it

### 1. One-shot CLI — `summarise-yt-video.sh`

```
./summarise-yt-video.sh <youtube-video-id> [model]
```

Builds the full URL from the ID, ensures `data/` and `whisper-cache/` exist,
and runs `docker compose run --rm yt-summarize <url>`. Prints the summary to
stdout and exits — no server, no state kept in memory. The `yt-summarize`
service carries `profiles: ["tools"]` in `docker-compose.yaml` specifically
so `docker compose up` never starts it by accident.

### 2. Web UI — `web/`

```
docker compose up -d       # starts web + nginx (not yt-summarize)
open http://localhost:8090
```

A small Flask app (`web/app.py`) built on the *same* image family as the CLI
path — it has `yt-dlp` and `faster-whisper` installed directly, and invokes
`entrypoint.sh` **as a subprocess**, not via Docker-in-Docker. No new
containers get spun up per job; it's the same script, just called
in-process instead of as the container's `ENTRYPOINT`.

- **Submitting a job**: `POST /submit` takes a bare video ID or a full
  YouTube URL (watch/shorts/embed/live/`youtu.be` links all parsed), does a
  regex sanity check (11-character ID), and inserts a row into SQLite with
  `status = 'queued'`. Returns immediately — actual processing happens
  asynchronously.
- **Processing jobs — all run fully in parallel, only STT is capped**: a
  single background thread (`worker_loop`, started once at process startup)
  polls SQLite every 2s for the oldest `queued` row, flips it to `running`,
  and hands it to a fresh thread (`run_job`) rather than blocking on it — so
  it can immediately go claim the next queued row too. Concurrent
  submissions never double-claim the same row, since claiming (the
  `UPDATE ... status = 'running'`) always happens on `worker_loop`'s single
  thread before a job's `run_job` thread is even started. Each `run_job`
  thread spawns `entrypoint.sh` as its own subprocess with no throttling at
  the Python level at all — every job, `hetzner:`-prefixed or bare, runs at
  the same time as every other job, including several bare-model jobs
  hitting the same Ollama concurrently (Ollama itself queues/batches
  concurrent requests fine; nothing here needs to protect it). This
  guarantee still depends on there being exactly **one** `worker_loop`
  thread doing the claiming, which is why the container runs
  `gunicorn --workers 1` (see `web/Dockerfile`): more worker processes would
  each start their own polling thread and could double-claim the same row.
- **STT is capped at `STT_MAX_CONCURRENT` concurrent transcriptions
  (default 2), independent of how many jobs are running overall**: Whisper
  is CPU-bound on this box, so letting every job's transcription run at once
  would just have them all contend for the same cores. `entrypoint.sh`
  enforces the cap with a small semaphore
  (`acquire_stt_slot`/`release_stt_slot`) built from `flock` against
  `STT_MAX_CONCURRENT` fixed lock-file "slots" under
  `/tmp/yt-summarise-stt-locks/` — a job grabs the first free slot (retrying
  every second if none is free) before it calls `faster-whisper`, and
  releases it the moment transcription finishes, well before the
  summarize/translate steps. Applies identically to Hetzner and bare-model
  jobs; enforced by the OS (`flock`), not an in-process Python lock; tune it
  via the `STT_MAX_CONCURRENT` env var (`docker-compose.yaml`).
- **STT is deduplicated across jobs on the same video**: a *separate*,
  per-video-ID `flock` (`/tmp/yt-summarise-video-<video_id>.lock`) guards a
  transcript cache at `data/.transcript-cache/<video_id>/`. Before
  transcribing, a job checks whether that video's already cached (e.g. by an
  earlier job for the same video with a different model) and reuses it
  instead of running whisper again. Since the check and the transcription
  both happen inside the same per-video lock, whichever job gets there first
  does the real work and the rest just find it already cached once they're
  unblocked — no separate coordination needed. This lock only serializes
  jobs on the *same* video; jobs on different videos never contend on it,
  so it doesn't count against the `STT_MAX_CONCURRENT` cap above (that's a
  separate lock, only held around the actual whisper call). Never
  invalidated: a given video's audio doesn't change, so the cache is kept
  indefinitely.
- **Killing a job kills its whole process tree, not just entrypoint.sh**:
  `run_job` starts `entrypoint.sh` with `start_new_session=True`, putting it
  (and everything it spawns — `yt-dlp`, the whisper `python3` process, `curl`
  for the LLM call) in its own process group. Deleting a job sends `SIGTERM`
  to that whole group via `os.killpg`, not just to the top-level script —
  otherwise an in-progress whisper transcription would be orphaned and keep
  running (and keep holding the STT lock) after its job was deleted.
- **Reading results back**: on success, the worker re-reads
  `data/yt-summaries.jsonl` and pulls out the entry matching that job's
  video ID (title, summary, token count) to store alongside the job row. On
  failure, the last ~2000 chars of the subprocess's stderr are stored as the
  error.
- **The page** (`web/templates/index.html`) is a single view: a submit form
  plus a table of every job ever run, newest first, each with status,
  elapsed time, token count, and a "view"/"anzeigen" link per job. It
  auto-refreshes every 15s unconditionally — the result itself lives on a
  separate page (below), so there's nothing on the list page a refresh
  could interrupt mid-read.
- **Job detail page** (`GET /jobs/<id>`, `web/templates/job.html`): the full
  English/German summary (or error) for one job, opened in its own tab via
  the list page's links so it survives the list's own refresh cycle
  regardless of how long you spend reading. Only auto-refreshes itself
  while the job is still `queued`/`running` — a finished job's page is
  static.
- **Crash recovery**: if the container restarts mid-job (e.g. rebuilding
  after a code change), the in-flight subprocess dies but its DB row would
  otherwise stay `running` forever, since the worker loop only ever looks at
  `queued` rows. `init_db()` resets any leftover `running` row back to
  `queued` on startup, so it gets retried automatically.
- **nginx** (`web/nginx.conf`) just reverse-proxies `:80` → `web:8000`; it's
  there as the front door (TLS termination, etc. would go here later), not
  doing anything the Flask app couldn't do on its own yet.

## Storage

Nothing here is a "real" database beyond SQLite — this is a personal-scale
tool, not a service meant to handle load.

| Path | What | Written by |
|---|---|---|
| `data/yt-summaries.jsonl` | Every completed run, ever — `{timestamp, video_id, title, url, summary, summary_tokens}`. Despite the name, each record is **pretty-printed** (multi-line) JSON, not one compact object per line — entries are concatenated back-to-back, not newline-delimited. `web/app.py` parses this with a streaming JSON decoder (`iter_json_objects`) rather than `readlines()` for that reason. | `entrypoint.sh`, appended, never overwritten |
| `whisper-cache/` | Hugging Face cache (`~/.cache/huggingface` inside the container) — the downloaded Whisper model weights. Shared between the CLI and web paths via the same bind mount, so either one warms the cache for both. | `faster-whisper` |
| `data/.transcript-cache/<video_id>/` | `transcript.txt` + `language.txt` for one video, so a second job on the same video reuses the transcript instead of re-running whisper. Kept forever, never invalidated. | `entrypoint.sh` |
| `web-data/jobs.db` | SQLite DB, one `jobs` table: id, video_id, status (`queued`/`running`/`done`/`error`), submitted/started/finished timestamps, title, summary, summary_tokens, error. Used only by the web service — the CLI path doesn't know it exists. | `web/app.py` |

All are gitignored and created on demand (`mkdir -p` in the script, or
Docker auto-creating bind-mount targets) — nothing needs to be provisioned
before first run.

## Network dependencies

- **Outbound to the internet**: YouTube (audio + metadata), Hugging Face Hub
  (Whisper model weights, first run only), and — only when a `hetzner:`
  model is selected — `inference.hetzner.com`.
- **Outbound to the LAN**: `OLLAMA_HOST` for the Ollama summarization
  call (and `LAYA_HOST` for tagging, if set) — the only *local* GPU used anywhere in this pipeline.
- **No inbound dependencies** beyond nginx's published port (`8090`) for the
  web UI.

## Known limitations

- Video-ID validation is a plain 11-character regex — it doesn't confirm the
  video actually exists or is downloadable until `yt-dlp` tries.
- No auth on the web UI or on `/submit` — anyone who can reach port `8090`
  can queue jobs.
- Nothing caps how many jobs can be mid-*summarization*/*translation* at
  once, on either backend — only STT is capped (`STT_MAX_CONCURRENT`).
  Several bare-model jobs summarizing concurrently all hit the same Ollama
  at the same time; several Hetzner jobs are each rate-limited individually
  (10 requests/min per key) but not against each other here. Relies on
  Ollama/Hetzner themselves to queue or reject excess concurrent requests
  gracefully rather than fall over.

## License

[MIT](LICENSE)
