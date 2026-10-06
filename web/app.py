import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from markupsafe import Markup, escape

DB_PATH = "/web-data/jobs.db"
JSONL_PATH = "/data/yt-summaries.jsonl"
# Written by entrypoint.sh (one <video_id>/ directory per video, holding
# transcript.txt + language.txt) so a re-run of the same video can skip
# whisper. Read-only here — it doubles as the store for "original text".
TRANSCRIPT_CACHE_DIR = "/data/.transcript-cache"
ENTRYPOINT = "/app/entrypoint.sh"
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
# yt-dlp IDs from non-YouTube extractors are not 11-char YouTube IDs and can
# carry other punctuation; this is the loose "safe to use as a cache
# directory name" check (no separators, no leading dot) used wherever an ID
# reaches the filesystem.
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
YOUTUBE_URL_ID_RE = re.compile(
    r"(?:youtube(?:-nocookie)?\.com/(?:watch\?(?:[^ ]*&)?v=|shorts/|embed/|live/|v/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)
POLL_INTERVAL_SECONDS = 2

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
DEFAULT_MODEL = os.environ.get("SUMMARY_MODEL", "qwen3.8:27b")
# custodes hands out the GPU on this box and stops Ollama for stacks that
# need the card. Jobs take a lease there before calling Ollama, see
# OllamaLease. Empty CUSTODES_CONTAINER = no custodes, call Ollama directly.
CUSTODES_CONTAINER = os.environ.get("CUSTODES_CONTAINER", "")
OLLAMA_LEASE_MIB = int(os.environ.get("OLLAMA_LEASE_MIB", "11000"))
HETZNER_BASE_URL = os.environ.get("HETZNER_BASE_URL", "https://inference.hetzner.com/api/v1")
HETZNER_API_KEY = os.environ.get("HETZNER_INFERENCE_API_KEY", "")
# Fallback if the key is set but /models is unreachable or its response
# shape differs from what's expected — keeps the option present rather than
# silently vanishing. Mirrors the models actually configured for this key
# in the `assistant` stack's Open WebUI connector.
HETZNER_FALLBACK_MODELS = ["Qwen/Qwen3.6-35B-A3B-FP8", "Qwen3.8-27B"]

# Topic tagging with laya, a decision model served by Ollaya. Off when
# LAYA_HOST is empty. Experimental: accuracy is known to be mediocre (~3 in 4 right on this
# box's history), so every tag is shown with the numbers behind it.
LAYA_HOST = os.environ.get("LAYA_HOST", "")
LAYA_MODEL = os.environ.get("LAYA_MODEL", "laya")
# The same question, asked of TypeSafe's hosted Jev through Vercel's AI
# Gateway (TypeSafe-compatible API, so the request is identical), to compare
# the two side by side. Off when no key is configured. Billed per input token,
# about $0.00003 per video.
AI_GATEWAY_API_KEY = os.environ.get("AI_GATEWAY_API_KEY", "")
JEV_HOST = os.environ.get("JEV_HOST", "https://ai-gateway.vercel.sh/typesafe")
JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe-ai/jev")
# Title + channel + the start of the summary. The full summary made laya
# answer OTHER for almost everything; this cut is what tested best.
CATEGORY_SUMMARY_CHARS = 1500
CATEGORY_QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "Which topic is the YouTube video `video` about?",
        "criteria": {
            "IT": "software, programming, computers, AI, LLMs, IT infrastructure, "
                  "networking, hardware, tech industry",
            "POLITICS": "politics and current affairs: immigration, war in Ukraine, "
                        "AfD, Brandmauer, parties, elections, government, geopolitics",
            "OTHER": "anything else: science, history, health, entertainment, "
                     "economy, lifestyle",
        },
    }
}

app = Flask(__name__)
# The topic filter buttons on the index page, one per label the classifiers
# can answer with.
app.jinja_env.globals["topics"] = list(CATEGORY_QUESTIONS["category"]["criteria"])


def list_ollama_models():
    """Live model list from Ollama, so the dropdown reflects whatever's
    actually pulled rather than a hardcoded, driftable list."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3) as resp:
            data = json.load(resp)
        names = sorted(m["name"] for m in data.get("models", []))
        return names or [DEFAULT_MODEL]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return [DEFAULT_MODEL]


def list_hetzner_models():
    """Empty when no key is configured — mirrors the assistant stack's own
    convention of omitting the connector entirely rather than showing a
    dead option. The "hetzner:" prefix on each id is entrypoint.sh's
    routing signal (see call_llm there), stripped back off before the
    request goes to Hetzner."""
    if not HETZNER_API_KEY:
        return []
    try:
        req = urllib.request.Request(
            f"{HETZNER_BASE_URL}/models",
            headers={"Authorization": f"Bearer {HETZNER_API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.load(resp)
        ids = sorted(m["id"] for m in data.get("data", []))
        if not ids:
            raise ValueError("empty model list")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError):
        ids = HETZNER_FALLBACK_MODELS
    return [f"hetzner:{m}" for m in ids]


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            submitted_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            title TEXT,
            summary TEXT,
            error TEXT
        )
    """)
    for ddl in (
        "ALTER TABLE jobs ADD COLUMN summary_tokens INTEGER",
        "ALTER TABLE jobs ADD COLUMN model TEXT",
        "ALTER TABLE jobs ADD COLUMN summary_de TEXT",
        "ALTER TABLE jobs ADD COLUMN channel TEXT",
        "ALTER TABLE jobs ADD COLUMN stage TEXT",
        "ALTER TABLE jobs ADD COLUMN translate_tokens INTEGER",
        # NULL = never opened since it finished; see mark_viewed(). Existing
        # rows therefore all start out as unviewed, which is the honest
        # default — nothing recorded a view before this column existed.
        "ALTER TABLE jobs ADD COLUMN viewed_at TEXT",
        # The exact URL handed to yt-dlp. Jobs are no longer YouTube-only, so
        # the URL can't be rebuilt from video_id any more. NULL on rows
        # predating this column — those are all YouTube, hence the
        # watch?v= fallback in job_source_url().
        "ALTER TABLE jobs ADD COLUMN url TEXT",
        # laya's topic tag: the chosen label, its probability, and laya's own
        # confidence (a separate number, not the same as the probability).
        # category 'ERROR' = laya rejected the request; not retried.
        "ALTER TABLE jobs ADD COLUMN category TEXT",
        "ALTER TABLE jobs ADD COLUMN category_p REAL",
        "ALTER TABLE jobs ADD COLUMN category_confidence REAL",
        # The same three, from Jev.
        "ALTER TABLE jobs ADD COLUMN jev_category TEXT",
        "ALTER TABLE jobs ADD COLUMN jev_p REAL",
        "ALTER TABLE jobs ADD COLUMN jev_confidence REAL",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # already added by a previous run
    # A container restart (e.g. rebuilding after a code change) kills the
    # in-flight entrypoint.sh subprocess without ever updating its row, so
    # on startup any leftover 'running' job is orphaned — nothing would
    # otherwise pick it up again, since the worker loop only looks at
    # 'queued'. Requeue it so it's retried from scratch.
    conn.execute(
        "UPDATE jobs SET status = 'queued', started_at = NULL, stage = NULL WHERE status = 'running'"
    )
    conn.commit()
    conn.close()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def extract_video_id(raw):
    """The 11-char YouTube ID behind a bare ID or a full YouTube URL (with or
    without a scheme, watch/shorts/embed/live/youtu.be links, extra query
    params), or None when this isn't a recognisable YouTube video link."""
    raw = raw.strip()
    if VIDEO_ID_RE.match(raw):
        return raw
    m = YOUTUBE_URL_ID_RE.search(raw)
    return m.group(1) if m else None


def parse_submission(raw):
    """Turn whatever was pasted into the form into (url, video_id).

    Anything yt-dlp can download is fair game — entrypoint.sh hands the URL
    straight to yt-dlp and reads the title/ID/channel back out of its
    --write-info-json, so it was never actually YouTube-specific. Only the
    web form was, rejecting e.g. youtube.com/clip/…, youtube-nocookie.com
    and every non-YouTube site yt-dlp supports.

    video_id is None for anything but a plain YouTube video link; it is then
    filled in from yt-dlp's own metadata once the job has run (see run_job),
    which is the only thing that can know it. Returns (None, None) when the
    input is neither a video ID nor something URL-shaped."""
    raw = raw.strip()
    if not raw:
        return None, None

    video_id = extract_video_id(raw)
    if VIDEO_ID_RE.match(raw):
        # A bare ID is by definition a YouTube one — expand it to the URL
        # entrypoint.sh will be given.
        return f"https://www.youtube.com/watch?v={raw}", video_id

    url = raw if "://" in raw else f"https://{raw}"
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or "." not in parsed.netloc:
        return None, None
    return url, video_id


def iter_json_objects(text):
    """yt-summaries.jsonl holds pretty-printed (multi-line) JSON objects
    back-to-back, not one-object-per-line — a plain readlines()+json.loads
    would choke on that, so decode the stream positionally instead."""
    decoder = json.JSONDecoder()
    idx, length = 0, len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, end = decoder.raw_decode(text, idx)
        yield obj
        idx = end


RESULT_FIELDS = (
    "video_id",
    "title",
    "channel",
    "summary",
    "summary_de",
    "summary_tokens",
    "translate_tokens",
)


def read_latest_result(video_id, url=None):
    """The newest yt-summaries.jsonl entry for this job. Matched on video_id
    when we know it, else on the submitted URL — a non-YouTube job has no ID
    until yt-dlp reports one, and the entry records the URL it was given."""
    empty = {k: None for k in RESULT_FIELDS}
    try:
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return empty
    match = None
    for entry in iter_json_objects(text):
        if video_id:
            hit = entry.get("video_id") == video_id
        else:
            hit = url is not None and entry.get("url") == url
        if hit:
            match = entry
    if match is None:
        return empty
    return {k: match.get(k) for k in RESULT_FIELDS}


def read_transcript(video_id):
    """The whisper transcript the summary was made from, straight out of
    entrypoint.sh's cache — (text, detected_language), both None when the
    video was never transcribed on this box or the cache has since been
    cleared. The SAFE_ID_RE check keeps a hand-edited/legacy DB row (or an
    odd ID from a non-YouTube extractor) from turning into a path traversal
    out of the cache directory."""
    if not video_id or not SAFE_ID_RE.match(video_id):
        return None, None
    base = os.path.join(TRANSCRIPT_CACHE_DIR, video_id)
    try:
        with open(os.path.join(base, "transcript.txt"), encoding="utf-8") as f:
            text = f.read().strip()
    except OSError:
        return None, None
    if not text:
        return None, None
    try:
        with open(os.path.join(base, "language.txt"), encoding="utf-8") as f:
            language = f.read().strip() or None
    except OSError:
        language = None
    return text, language


def translation_is_duplicate(summary, summary_de):
    """True when the German "translation" is really just the summary again —
    the model echoing its input back, or a video that was already German.
    Compared on collapsed whitespace so a stray reflow/blank line doesn't
    hide the duplication."""
    if not summary or not summary_de:
        return False
    return " ".join(summary.split()) == " ".join(summary_de.split())


# The models write their summaries in markdown. Only **bold** is turned into
# HTML: the bullet "- " markers read fine as-is inside the <pre> the summary is
# rendered in, whereas a stray ** in the middle of a sentence does not.
# Deliberately not a markdown library — that would want to own the whole block
# (lists, paragraphs, line breaks) and lose the verbatim <pre> layout.
#
# \S on both inner edges is CommonMark's rule: "** x**" and "**x **" are not
# emphasis. The dot does not match newlines, so an unpaired ** can only ever
# swallow the rest of its own line, never the rest of the summary.
MARKDOWN_BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")


@app.template_filter("markdown_bold")
def markdown_bold(text):
    """Escape text for HTML, then re-introduce <strong> for **bold**.

    Escaping first is what makes this safe: by the time the bold pattern is
    applied, any < > & in the model's output is already an entity, so the only
    tags in the result are the ones added here.

    str() around the escaped text is load-bearing. re.sub() reassembles its
    result with the subject string's own join(), and Markup.join() escapes
    what it is given — run directly on the Markup, the <strong> tags added
    here would come back out as &lt;strong&gt;."""
    if not text:
        return ""
    escaped = str(escape(text))
    return Markup(MARKDOWN_BOLD_RE.sub(r"<strong>\1</strong>", escaped))


def elapsed_seconds(start_iso, end_iso):
    if not start_iso:
        return None
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso) if end_iso else datetime.now(timezone.utc)
    return int((end - start).total_seconds())


def format_elapsed(seconds):
    if seconds is None:
        return None
    minutes, seconds = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


# A job counts as viewable only once it has actually finished — a queued or
# running one has no result to have missed yet, so it is neither "new" nor
# markable as seen.
VIEWABLE_STATUSES = ("done", "error")


def job_source_url(job):
    """Where this job's video came from. Rows created before the url column
    existed are all YouTube, so rebuild theirs from the video ID."""
    if job["url"]:
        return job["url"]
    if job["video_id"]:
        return f"https://www.youtube.com/watch?v={job['video_id']}"
    return None


def with_elapsed(jobs):
    result = []
    for job in jobs:
        job = dict(job)
        job["source_url"] = job_source_url(job)
        secs = elapsed_seconds(job["started_at"], job["finished_at"])
        job["elapsed_seconds"] = secs
        job["elapsed"] = format_elapsed(secs)
        job["unread"] = job["status"] in VIEWABLE_STATUSES and not job["viewed_at"]
        result.append(job)
    return result


def mark_viewed(job):
    """Stamp a finished job the first time its detail page is opened. Only the
    first view is recorded (the timestamp then stays put), and only for
    finished jobs — opening a still-running one and leaving before it
    completes shouldn't hide the result you never saw."""
    if job["status"] not in VIEWABLE_STATUSES or job["viewed_at"]:
        return
    stamp = now()
    conn = get_db()
    conn.execute(
        "UPDATE jobs SET viewed_at = ? WHERE id = ? AND viewed_at IS NULL",
        (stamp, job["id"]),
    )
    conn.commit()
    conn.close()
    job["viewed_at"] = stamp
    job["unread"] = False


def load_job(job_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return with_elapsed([row])[0] if row is not None else None


@app.route("/jobs/<int:job_id>", methods=["GET"])
def job_detail(job_id):
    job = load_job(job_id)
    if job is None:
        return "Job not found", 404
    mark_viewed(job)
    transcript, _ = read_transcript(job["video_id"])
    return render_template(
        "job.html",
        job=job,
        transcript_chars=len(transcript) if transcript else None,
        translation_identical=translation_is_duplicate(job["summary"], job["summary_de"]),
    )


@app.route("/jobs/<int:job_id>/transcript", methods=["GET"])
def job_transcript(job_id):
    job = load_job(job_id)
    if job is None:
        return "Job not found", 404
    transcript, language = read_transcript(job["video_id"])
    if transcript is None:
        return "No transcript stored for this video", 404
    return render_template(
        "transcript.html", job=job, transcript=transcript, language=language
    )


@app.route("/", methods=["GET"])
def index():
    conn = get_db()
    jobs = conn.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
    conn.close()
    return render_template(
        "index.html",
        jobs=with_elapsed(jobs),
        error=None,
        local_models=list_ollama_models(),
        hetzner_models=list_hetzner_models(),
        default_model=DEFAULT_MODEL,
        ollama_host=OLLAMA_HOST,
    )


@app.route("/search", methods=["GET"])
def search():
    """IDs of the jobs whose summary (English or German) contains q, for the
    index page's text filter. Matched in Python rather than with SQL LIKE,
    which only folds case for ASCII (so "über" wouldn't find "Über")."""
    q = request.args.get("q", "").strip().casefold()
    conn = get_db()
    rows = conn.execute("SELECT id, summary, summary_de FROM jobs").fetchall()
    conn.close()
    return jsonify(ids=[
        r["id"] for r in rows
        if q in (r["summary"] or "").casefold() or q in (r["summary_de"] or "").casefold()
    ])


@app.route("/submit", methods=["POST"])
def submit():
    raw = request.form.get("video_id", "").strip()
    url, video_id = parse_submission(raw)
    model = request.form.get("model", "").strip() or DEFAULT_MODEL
    if url is None:
        conn = get_db()
        jobs = conn.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
        conn.close()
        error = (
            f"“{raw}” is neither a URL nor a YouTube video ID "
            "(expected 11 characters, e.g. dQw4w9WgXcQ)."
        )
        return render_template(
            "index.html",
            jobs=with_elapsed(jobs),
            error=error,
            local_models=list_ollama_models(),
            hetzner_models=list_hetzner_models(),
            default_model=DEFAULT_MODEL,
            ollama_host=OLLAMA_HOST,
        ), 400

    conn = get_db()
    conn.execute(
        "INSERT INTO jobs (video_id, url, status, submitted_at, model)"
        " VALUES (?, ?, 'queued', ?, ?)",
        (video_id or "", url, now(), model),
    )
    conn.commit()
    conn.close()
    return redirect(url_for("index"))


# Tracks the subprocess behind every currently-running job (any number can
# be running at once — every job, local or Hetzner, runs unrestricted; only
# the STT step inside entrypoint.sh caps its own concurrency, via
# STT_MAX_CONCURRENT/flock, independent of how many jobs are in flight), so
# deleting one can actually kill it instead of just hiding its row. Guarded
# by its own lock since multiple job threads touch it concurrently.
current_jobs = {}  # job_id -> Popen
current_jobs_lock = threading.Lock()


# Maps distinctive substrings of entrypoint.sh's own log() lines (stderr) to
# a short, human-readable stage — checked in order, first match wins. Lets
# the UI show what a 'running' job is actually doing right now instead of
# just "running", without entrypoint.sh needing to know about the DB at all.
STAGE_PATTERNS = [
    ("Downloading audio", "downloading"),
    ("waiting for STT lock", "waiting for STT"),
    ("Reusing cached transcript", "transcribing (cached)"),
    ("Acquired STT lock", "transcribing"),
    ("Waiting for the GPU", "waiting for GPU"),
    ("Summarizing with model", "summarizing"),
    ("Translating summary to German", "translating"),
    ("Logged to", "finishing"),
]


def parse_stage(line):
    for needle, stage in STAGE_PATTERNS:
        if needle in line:
            return stage
    return None


def set_stage(job_id, stage):
    conn = get_db()
    conn.execute("UPDATE jobs SET stage = ? WHERE id = ?", (stage, job_id))
    conn.commit()
    conn.close()


@app.route("/jobs/<int:job_id>/delete", methods=["POST"])
def delete_job(job_id):
    with current_jobs_lock:
        proc = current_jobs.get(job_id)
    if proc is not None:
        # proc.terminate() would only signal entrypoint.sh itself — the
        # actual whisper transcription runs as a grandchild (flock execs
        # into python3, still a child of the bash script), which would be
        # orphaned and keep running/keep holding the STT lock. Signaling the
        # whole process group (see start_new_session=True below) reaches it
        # too.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # already exited
    conn = get_db()
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("index"))


# Retrying is just putting a failed row back to 'queued' — worker_loop picks
# it up on its next poll and runs it from scratch, same as a fresh submission.
# Only 'error' rows are touched, so a double-click or a stale page can't
# requeue a job that's already running or done.
REQUEUE_SQL = (
    "UPDATE jobs SET status = 'queued', error = NULL, stage = NULL,"
    " started_at = NULL, finished_at = NULL WHERE status = 'error'"
)


@app.route("/jobs/<int:job_id>/retry", methods=["POST"])
def retry_job(job_id):
    conn = get_db()
    conn.execute(REQUEUE_SQL + " AND id = ?", (job_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("job_detail", job_id=job_id))


@app.route("/jobs/retry-failed", methods=["POST"])
def retry_failed():
    conn = get_db()
    conn.execute(REQUEUE_SQL)
    conn.commit()
    conn.close()
    return redirect(url_for("index"))


def log_line(text):
    return f"[{datetime.now().strftime('%H:%M:%S')}] {text}\n"


class OllamaLease:
    """One custodes lease, "yt-summarize", held while any job talks to Ollama.

    Taking it waits in custodes' queue until OLLAMA_LEASE_MIB fit beside the
    other leases, so no stack can evict Ollama while it is held. Then Ollama is
    woken in case custodes had stopped it. The lease is bound to this
    container's init process (pid as the host sees it), so it goes stale on its
    own if the container dies. A pid-only lease is never stopped for being idle.

    custodes runs inside its own container (`docker exec`), which has what it
    needs: pid: host, nvidia-smi, the docker socket and the g1 checkouts."""

    NAME = "yt-summarize"

    def __init__(self):
        self.cond = threading.Condition()
        self.users = 0
        self.state = "free"  # free | taking | held
        self.calls = threading.Lock()  # keeps a give from racing a later take
        self.host_pid = None

    def custodes(self, *args):
        r = subprocess.run(
            ["docker", "exec", CUSTODES_CONTAINER, "sh", "-c", 'exec "$CUSTODES" "$@"',
             "custodes", *args],
            capture_output=True, text=True,
        )
        out = (r.stdout + r.stderr).strip()
        return r.returncode, out

    def take(self):
        """Returns log lines for the job."""
        notes = []
        with self.calls:
            if self.host_pid is None:
                r = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Pid}}", os.uname().nodename],
                    capture_output=True, text=True,
                )
                self.host_pid = r.stdout.strip() if r.returncode == 0 else None
                if not self.host_pid:
                    return False, [log_line(f"custodes: cannot find own pid: {r.stderr.strip()}")]
            code, out = self.custodes(
                "take", self.NAME, str(OLLAMA_LEASE_MIB),
                "--no-evict", "--wait", "--pid", self.host_pid,
            )
            notes += [log_line(l) for l in out.splitlines()]
            if code != 0:
                return False, notes
            # fails harmlessly when Ollama runs, or was stopped by hand
            _, out = self.custodes("wake", "ollama")
            notes += [log_line(l) for l in out.splitlines()]
        return True, notes

    def give(self):
        with self.calls:
            code, out = self.custodes("give", self.NAME)
        if code != 0:
            print(f"custodes give {self.NAME}: {out}", flush=True)

    def acquire(self):
        """Blocks until Ollama may be used. Returns log lines for the job;
        if custodes fails the job goes ahead anyway, as it did without it."""
        with self.cond:
            self.users += 1
            while self.state == "taking":
                self.cond.wait()
            if self.state == "held":
                return []
            self.state = "taking"
        ok, notes = self.take()
        with self.cond:
            self.state = "held" if ok else "free"
            self.cond.notify_all()
        return notes

    def release(self):
        with self.cond:
            self.users -= 1
            if self.users or self.state != "held":
                return
            self.state = "free"
        self.give()


ollama_lease = OllamaLease() if CUSTODES_CONTAINER else None


def run_job(job_id, url, video_id, model):
    env = dict(os.environ)
    if ollama_lease:
        env["OLLAMA_GATE"] = "1"
    proc = subprocess.Popen(
        [ENTRYPOINT, url, model],
        stdin=subprocess.PIPE,  # the go-ahead after "Waiting for the GPU", see entrypoint.sh
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered, so stage updates land as each log() line is written
        start_new_session=True,  # own process group, so delete_job can kill the whole tree
        env=env,
    )
    with current_jobs_lock:
        current_jobs[job_id] = proc

    # entrypoint.sh's own log() lines (stderr) double as a live stage feed
    # — read them as they arrive rather than waiting for the process to
    # exit. stdout is drained concurrently on its own thread purely to
    # avoid a deadlock (the child could otherwise block writing to a full
    # stdout pipe while we're only reading stderr); its content isn't
    # used for anything but the error-tail fallback below.
    stdout_chunks = []
    stdout_thread = threading.Thread(
        target=lambda: stdout_chunks.append(proc.stdout.read()), daemon=True
    )
    stdout_thread.start()

    stderr_lines = []
    last_stage = None
    leased = False
    try:
        for line in proc.stderr:
            stderr_lines.append(line)
            stage = parse_stage(line)
            if stage and stage != last_stage:
                last_stage = stage
                set_stage(job_id, stage)
            if ollama_lease and "Waiting for the GPU" in line and not leased:
                leased = True
                stderr_lines += ollama_lease.acquire()
                try:
                    proc.stdin.write("\n")
                    proc.stdin.close()
                except OSError:
                    pass  # job was deleted while waiting
        proc.wait()
    finally:
        if leased:
            ollama_lease.release()
    stdout_thread.join()
    stdout = stdout_chunks[0] if stdout_chunks else ""
    stderr = "".join(stderr_lines)
    returncode = proc.returncode
    with current_jobs_lock:
        current_jobs.pop(job_id, None)

    conn = get_db()
    if returncode == 0:
        r = read_latest_result(video_id, url)
        # For a non-YouTube (or clip/embed) submission the ID was unknown at
        # submit time — yt-dlp's metadata is the first thing that knows it,
        # and it's what the transcript cache is keyed on, so store it now.
        conn.execute(
            """UPDATE jobs SET status = 'done', finished_at = ?, video_id = ?,
               title = ?, channel = ?, summary = ?, summary_de = ?,
               summary_tokens = ?, translate_tokens = ?
               WHERE id = ?""",
            (
                now(),
                video_id or r["video_id"] or "",
                r["title"],
                r["channel"],
                r["summary"],
                r["summary_de"],
                r["summary_tokens"],
                r["translate_tokens"],
                job_id,
            ),
        )
    else:
        err_tail = (stderr or stdout or "unknown error")[-2000:]
        conn.execute(
            "UPDATE jobs SET status = 'error', finished_at = ?, error = ? WHERE id = ?",
            (now(), err_tail, job_id),
        )
    conn.commit()
    conn.close()


# (name, base URL, model, API key, DB columns for label/probability/confidence)
CLASSIFIERS = []
if LAYA_HOST:
    CLASSIFIERS.append(
        ("laya", LAYA_HOST, LAYA_MODEL, "",
         ("category", "category_p", "category_confidence"))
    )
if AI_GATEWAY_API_KEY:
    CLASSIFIERS.append(
        ("jev", JEV_HOST, JEV_MODEL, AI_GATEWAY_API_KEY,
         ("jev_category", "jev_p", "jev_confidence"))
    )


# Statuses that say this particular request is wrong (400 bad request, 404,
# 422 failed validation), so retrying the same job can't help. Anything else —
# rate limits, overload, a bad key or an exhausted budget — is retried.
PERMANENT_HTTP_ERRORS = (400, 404, 422)


def classify(host, model, api_key, title, channel, summary):
    """(label, probability, confidence) for this video."""
    state = {
        "video": {
            "title": title or "",
            "channel": channel or "",
            "summary": (summary or "")[:CATEGORY_SUMMARY_CHARS],
        }
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{host}/v1/systemone",
        data=json.dumps(
            {"model": model, "state": state, "questions": CATEGORY_QUESTIONS}
        ).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        answer = json.load(resp)["answers"]["category"]
    label = answer["choice"]
    return label, answer["probabilities"][label], answer["confidence"]


def classify_loop(name, host, model, api_key, columns):
    """Tags every finished job this classifier hasn't tagged yet, newest
    first — new jobs as they finish, and the backlog from before tagging
    existed. One thread per classifier, kept out of run_job, so a slow,
    absent or rate-limited backend never holds up a summary or the other
    classifier."""
    label_col = columns[0]
    backoff = 5
    while True:
        conn = get_db()
        row = conn.execute(
            f"SELECT id, title, channel, summary FROM jobs"
            f" WHERE status = 'done' AND {label_col} IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row is None:
            time.sleep(5)
            continue
        try:
            result = classify(host, model, api_key, row["title"], row["channel"], row["summary"])
            backoff = 5
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            print(f"{name}: job {row['id']}: HTTP {exc.code}: {body}", flush=True)
            if exc.code not in PERMANENT_HTTP_ERRORS:
                # Jev's upstream answers 429 "high demand" (and 529 overload)
                # a lot; 401/402/403 mean the key or its budget, not this job.
                # Back off and retry the same job rather than marking it failed.
                try:
                    wait = max(float(exc.headers.get("retry-after") or 0), backoff)
                except ValueError:
                    wait = backoff
                time.sleep(wait)
                backoff = min(backoff * 2, 300)
                continue
            result = ("ERROR", None, None)  # this request is bad; not retried
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError):
            time.sleep(60)  # unreachable; try again later
            continue
        conn = get_db()
        conn.execute(
            f"UPDATE jobs SET {columns[0]} = ?, {columns[1]} = ?, {columns[2]} = ?"
            " WHERE id = ?",
            (*result, row["id"]),
        )
        conn.commit()
        conn.close()


def worker_loop():
    # This loop itself stays single-threaded and non-blocking: it claims the
    # oldest queued job (flips it to 'running' immediately, so it can't be
    # claimed twice) and hands it off to a fresh thread, then immediately
    # loops again rather than waiting for that job to finish. Jobs then run
    # fully in parallel with each other — the only concurrency limit left is
    # STT_MAX_CONCURRENT, enforced inside entrypoint.sh itself, not here.
    while True:
        conn = get_db()
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY id ASC LIMIT 1"
        ).fetchone()
        if row is not None:
            job_id, video_id = row["id"], row["video_id"]
            url = job_source_url(row)
            model = row["model"] or DEFAULT_MODEL
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                (now(), job_id),
            )
            conn.commit()
        conn.close()

        if row is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        threading.Thread(
            target=run_job, args=(job_id, url, video_id, model), daemon=True
        ).start()


init_db()
if ollama_lease:
    # A restart requeues running jobs, but the lease of the old process can
    # still stand (the container kept running): give it back.
    threading.Thread(target=ollama_lease.give, daemon=True).start()
threading.Thread(target=worker_loop, daemon=True).start()
for classifier in CLASSIFIERS:
    threading.Thread(target=classify_loop, args=classifier, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
