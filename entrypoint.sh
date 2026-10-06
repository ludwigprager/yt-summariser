#!/usr/bin/env bash
#
# Runs inside the yt-summarize container. Downloads a YouTube video's audio,
# transcribes it with faster-whisper, summarizes via Ollama, and appends the
# result to /data/yt-summaries.jsonl (mount /data from the host).
#
# Env vars (set via docker-compose or `-e`):
#   OLLAMA_HOST            default http://ollama:11434   (container-to-container name)
#   SUMMARY_MODEL          default qwen3.8:27b — prefix with "hetzner:" (e.g.
#                          "hetzner:Qwen3.8-27B") to route that job to Hetzner's
#                          hosted inference instead of local Ollama
#   HETZNER_INFERENCE_API_KEY   required only for "hetzner:"-prefixed models
#   WHISPER_MODEL_SIZE     default medium
#   WHISPER_DEVICE         default cuda
#   WHISPER_COMPUTE_TYPE   default int8_float16
#   STT_MAX_CONCURRENT     default 2 — how many whisper transcriptions may run
#                          at once across all entrypoint.sh subprocesses
#   OLLAMA_NUM_CTX         default 32768 — context window requested per Ollama
#                          request. Ignored for "hetzner:" models (that endpoint
#                          sizes its own context). See call_llm() for why the
#                          default is this large.

set -euo pipefail

URL="${1:?Usage: docker run ... <youtube-url> [model]}"
SUMMARY_MODEL="${2:-${SUMMARY_MODEL:-qwen3.8:27b}}"

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
HETZNER_BASE_URL="${HETZNER_BASE_URL:-https://inference.hetzner.com/api/v1}"
HETZNER_INFERENCE_API_KEY="${HETZNER_INFERENCE_API_KEY:-}"
WHISPER_MODEL_SIZE="${WHISPER_MODEL_SIZE:-medium}"
WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-int8_float16}"
STT_MAX_CONCURRENT="${STT_MAX_CONCURRENT:-2}"
OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-32768}"
DATA_DIR="/data"
LOG_FILE="${DATA_DIR}/yt-summaries.jsonl"
WORK_DIR="$(mktemp -d)"

cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }

# The web app's go-ahead for Ollama arrives on stdin (see step 3). Keep it on
# fd 3, away from yt-dlp, ffmpeg and whisper, which get /dev/null instead.
[[ -n "${OLLAMA_GATE:-}" ]] && exec 3<&0 </dev/null

# Calls whichever backend $SUMMARY_MODEL selects — a bare model name goes to
# local Ollama's native /api/generate, a "hetzner:"-prefixed one goes to
# Hetzner's OpenAI-compatible /chat/completions instead (different request
# and response shape, hence the branch). Sets LLM_TEXT and LLM_TOKENS.
#
# Nothing here may put the prompt on a command line. It carries a whole
# transcript, which for a long video runs past 128KB — MAX_ARG_STRLEN, the
# kernel's cap on any *single* argv string (32 pages; the overall ARG_MAX is
# not the binding limit). Passing it as `jq --arg prompt "$prompt"` made
# execve fail with E2BIG ("Argument list too long"), so jq never ran, the
# command substitution expanded to nothing, curl posted an empty body, and
# Ollama replied {"error":"missing request body"} — surfacing as "LLM backend
# returned no summary" two steps later. `curl -d "$body"` would then have hit
# the same wall on the assembled JSON. So the prompt reaches jq as a
# --rawfile and the body reaches curl as --data-binary @file; printf is a
# bash builtin, so writing the file has no argv limit of its own.
call_llm() {
  local prompt="$1" resp
  local prompt_file="${WORK_DIR}/llm-prompt.txt"
  local body_file="${WORK_DIR}/llm-body.json"

  printf '%s' "$prompt" > "$prompt_file"

  if [[ "$SUMMARY_MODEL" == hetzner:* ]]; then
    local remote_model="${SUMMARY_MODEL#hetzner:}"
    if [[ -z "$HETZNER_INFERENCE_API_KEY" ]]; then
      echo "SUMMARY_MODEL=$SUMMARY_MODEL needs HETZNER_INFERENCE_API_KEY, which is not set" >&2
      exit 1
    fi
    jq -n --arg model "$remote_model" --rawfile prompt "$prompt_file" \
      '{model: $model, messages: [{role: "user", content: $prompt}], stream: false}' \
      > "$body_file"
    resp=$(curl -sS "${HETZNER_BASE_URL}/chat/completions" \
      -H "Authorization: Bearer ${HETZNER_INFERENCE_API_KEY}" \
      -H "Content-Type: application/json" \
      --data-binary @"$body_file")
    LLM_TEXT=$(echo "$resp" | jq -r '.choices[0].message.content // empty')
    LLM_TOKENS=$(echo "$resp" | jq -r '.usage.completion_tokens // 0')
  else
    # num_ctx matters as much as the argv fix above, because overflowing it
    # fails silently rather than loudly. Ollama spends only *half* of num_ctx
    # on the prompt (measured against this server: prompt_eval_count comes
    # back as num_ctx/2 + 2 at every size tried) and, when the prompt is
    # longer than that, discards from the *front*. At the server's own 32768
    # default that left 16384 tokens for a ~38000-token transcript: most of
    # the video went missing, the instruction line at the top of the prompt
    # with it, and what came back was a confident summary of the tail end.
    # The default is a hardware compromise, not a comfortable fit. Covering
    # the longest transcript seen here (~52000 tokens) would need ~104000,
    # and an f16 KV cache for this model costs ~266KB/token — ~28GB at that
    # size, on top of 17.7GB of Q4_K_M weights. The GPU has 12GB. So 32768 is
    # the largest window that does not make Ollama trade away even more
    # weight layers to the CPU; the ~3% of transcripts too long for it are
    # better sent to a "hetzner:" model, which sizes its own context. The
    # prompt layout below puts the instruction last so that when a transcript
    # does overflow, truncation eats transcript rather than the task.
    jq -n --arg model "$SUMMARY_MODEL" --rawfile prompt "$prompt_file" \
      --argjson num_ctx "$OLLAMA_NUM_CTX" \
      '{model: $model, prompt: $prompt, stream: false, options: {num_ctx: $num_ctx}}' \
      > "$body_file"
    resp=$(curl -sS "${OLLAMA_HOST}/api/generate" --data-binary @"$body_file")
    LLM_TEXT=$(echo "$resp" | jq -r '.response // empty')
    LLM_TOKENS=$(echo "$resp" | jq -r '.eval_count // 0')
  fi
  LLM_RAW_RESPONSE="$resp"
}

mkdir -p "$DATA_DIR"

# --- 1. Download audio + metadata -------------------------------------------
log "Downloading audio for $URL"
# One job summarizes exactly one video: --no-playlist keeps a pasted
# watch?v=…&list=… URL from pulling the whole playlist, and --playlist-items 1
# does the same for a URL that is nothing but a playlist/channel (where
# --no-playlist has nothing to strip). Without them every entry would
# overwrite the same audio.%(ext)s and only the last one would be summarized.
yt-dlp \
  -x --audio-format mp3 \
  --no-playlist --playlist-items 1 \
  --write-info-json \
  -o "${WORK_DIR}/audio.%(ext)s" \
  "$URL"

AUDIO_FILE="${WORK_DIR}/audio.mp3"
INFO_JSON=$(ls "${WORK_DIR}"/*.info.json 2>/dev/null | head -n1)

VIDEO_TITLE="unknown title"
VIDEO_ID="unknown"
VIDEO_CHANNEL="unknown channel"
if [[ -n "$INFO_JSON" ]]; then
  VIDEO_TITLE=$(jq -r '.title // "unknown title"' "$INFO_JSON")
  VIDEO_ID=$(jq -r '.id // "unknown"' "$INFO_JSON")
  VIDEO_CHANNEL=$(jq -r '.uploader // .channel // "unknown channel"' "$INFO_JSON")
fi

[[ -f "$AUDIO_FILE" ]] || { echo "Audio download failed, no mp3 found" >&2; exit 1; }

# --- 2. Transcribe with faster-whisper ---------------------------------------
# The web UI runs every job's LLM step fully in parallel now (no per-backend
# serialization — see web/app.py), but whisper is still CPU-bound on this
# box, so unlimited concurrent transcriptions would just thrash each other.
# Two locks divide that concern:
#
#   1. A per-video-ID lock (VIDEO_LOCK_FILE) — guarantees correctness of the
#      transcript cache below: if two jobs target the same video (e.g. same
#      video, two different SUMMARY_MODELs), only the first actually runs
#      whisper; the second, unblocking right after, finds the first one's
#      cached output and reuses it. Safe to key on video ID alone (no cache
#      invalidation): a given video's audio track doesn't change. Jobs on
#      *different* videos never contend on this lock.
#   2. A small semaphore (STT_MAX_CONCURRENT slots, acquire_stt_slot/
#      release_stt_slot below) — caps how many *different* videos may be
#      transcribing at once across all entrypoint.sh subprocesses in this
#      container, regardless of how many jobs are running in total. Only
#      held around the actual whisper call, not the cache check/copy.
STT_LOCK_DIR="/tmp/yt-summarise-stt-locks"
mkdir -p "$STT_LOCK_DIR"

acquire_stt_slot() {
  while true; do
    local slot
    for slot in $(seq 1 "$STT_MAX_CONCURRENT"); do
      exec {STT_SLOT_FD}>"${STT_LOCK_DIR}/slot-${slot}.lock"
      if flock -n "$STT_SLOT_FD"; then
        return 0
      fi
      exec {STT_SLOT_FD}>&-
    done
    sleep 1
  done
}

release_stt_slot() {
  flock -u "$STT_SLOT_FD"
  exec {STT_SLOT_FD}>&-
}

VIDEO_LOCK_FILE="/tmp/yt-summarise-video-${VIDEO_ID}.lock"
TRANSCRIPT_CACHE_DIR="/data/.transcript-cache/${VIDEO_ID}"
TRANSCRIPT_FILE="${WORK_DIR}/transcript.txt"
LANGUAGE_FILE="${WORK_DIR}/language.txt"

log "Transcribing with faster-whisper (${WHISPER_MODEL_SIZE}, ${WHISPER_DEVICE}) — waiting for STT lock"
exec {VIDEO_LOCK_FD}>"$VIDEO_LOCK_FILE"
flock -x "$VIDEO_LOCK_FD"
if [[ -f "${TRANSCRIPT_CACHE_DIR}/transcript.txt" ]]; then
  log "Reusing cached transcript for video ${VIDEO_ID} (already transcribed by another job)"
  cp "${TRANSCRIPT_CACHE_DIR}/transcript.txt" "$TRANSCRIPT_FILE"
  cp "${TRANSCRIPT_CACHE_DIR}/language.txt" "$LANGUAGE_FILE"
else
  acquire_stt_slot
  log "Acquired STT lock — transcribing now"
  python3 - "$AUDIO_FILE" "$WHISPER_MODEL_SIZE" "$WHISPER_DEVICE" "$WHISPER_COMPUTE_TYPE" "$TRANSCRIPT_FILE" "$LANGUAGE_FILE" <<'PY'
import sys
from faster_whisper import WhisperModel

audio_path, model_size, device, compute_type, out_path, lang_path = sys.argv[1:7]

model = WhisperModel(model_size, device=device, compute_type=compute_type)
segments, info = model.transcribe(audio_path, vad_filter=True)

with open(out_path, "w", encoding="utf-8") as f:
    for seg in segments:
        f.write(seg.text.strip() + " ")

with open(lang_path, "w", encoding="utf-8") as f:
    f.write(info.language or "")
PY
  release_stt_slot
  mkdir -p "$TRANSCRIPT_CACHE_DIR"
  cp "$TRANSCRIPT_FILE" "${TRANSCRIPT_CACHE_DIR}/transcript.txt"
  cp "$LANGUAGE_FILE" "${TRANSCRIPT_CACHE_DIR}/language.txt"
fi
flock -u "$VIDEO_LOCK_FD"
exec {VIDEO_LOCK_FD}>&-

TRANSCRIPT=$(<"$TRANSCRIPT_FILE")
if [[ -z "${TRANSCRIPT// }" ]]; then
  echo "Transcription produced no text" >&2
  exit 1
fi
log "Transcript length: ${#TRANSCRIPT} chars"

DETECTED_LANGUAGE=$(<"$LANGUAGE_FILE")
log "Detected spoken language: ${DETECTED_LANGUAGE:-unknown}"

# --- 3. Summarize --------------------------------------------------------
BACKEND_DESC="Ollama at ${OLLAMA_HOST}"
[[ "$SUMMARY_MODEL" == hetzner:* ]] && BACKEND_DESC="Hetzner at ${HETZNER_BASE_URL}"
# Under the web UI, other g1 stacks can have stopped Ollama to use the GPU
# (g1-custodes). The web app waits for its lease there and wakes Ollama, then
# writes a line to our stdin; nothing else is ever written to it.
if [[ -n "${OLLAMA_GATE:-}" && "$SUMMARY_MODEL" != hetzner:* ]]; then
  log "Waiting for the GPU (custodes) before calling Ollama"
  read -r _ <&3 || true
fi
log "Summarizing with model ${SUMMARY_MODEL} (${BACKEND_DESC})"

if [[ "$DETECTED_LANGUAGE" == "de" ]]; then
  # Video is already German — summarize directly in German instead of
  # producing an English summary and then translating it back. This is why
  # SUMMARY stays empty: the web UI treats an empty English summary as "no
  # English view needed" and shows only the German ("anzeigen") result.
  # Instruction last, after the transcript — see call_llm(): Ollama drops the
  # front of an over-long prompt, so anything at the top is what gets lost.
  PROMPT=$(cat <<EOF
Titel: ${VIDEO_TITLE}

Transkript:
${TRANSCRIPT}

Fasse das vorstehende Transkript eines YouTube-Videos in 5-8 prägnanten
Stichpunkten auf Deutsch zusammen. Erfasse die zentralen Aussagen, Argumente
oder Kernbotschaften. Keine Füllwörter oder Wiederholungen.
EOF
)
  call_llm "$PROMPT"
  SUMMARY=""
  SUMMARY_DE="$LLM_TEXT"
  SUMMARY_TOKENS="$LLM_TOKENS"
  # This path summarizes straight into German, so there is no translate
  # stage and nothing to report for it — 0, not "unknown".
  TRANSLATE_TOKENS=0

  if [[ -z "$SUMMARY_DE" ]]; then
    echo "LLM backend returned no summary. Raw response:" >&2
    echo "$LLM_RAW_RESPONSE" >&2
    exit 1
  fi
else
  # Instruction last — same reason as the German branch above.
  PROMPT=$(cat <<EOF
Title: ${VIDEO_TITLE}

Transcript:
${TRANSCRIPT}

Summarize the YouTube video transcript above in 5-8 concise bullet points,
capturing the key claims, arguments, or takeaways. Do not pad with filler.
EOF
)

  call_llm "$PROMPT"
  SUMMARY="$LLM_TEXT"
  SUMMARY_TOKENS="$LLM_TOKENS"

  if [[ -z "$SUMMARY" ]]; then
    echo "LLM backend returned no summary. Raw response:" >&2
    echo "$LLM_RAW_RESPONSE" >&2
    exit 1
  fi

  # --- 3b. Translate the summary to German (best-effort) ---------------------
  log "Translating summary to German"

  TRANSLATE_PROMPT=$(cat <<EOF
Translate the following text into German. Respond with only the
translation, no preamble or commentary. Use standard German without gender-inclusive notation.

${SUMMARY}
EOF
)

  call_llm "$TRANSLATE_PROMPT"
  SUMMARY_DE="$LLM_TEXT"
  TRANSLATE_TOKENS="$LLM_TOKENS"

  if [[ -z "$SUMMARY_DE" ]]; then
    log "German translation failed, leaving it empty"
  fi
fi

# --- 4. Output + log ----------------------------------------------------------
echo "=== ${VIDEO_TITLE} ==="
echo "$URL"
echo
if [[ -n "$SUMMARY" ]]; then
  echo "$SUMMARY"
  echo
  echo "--- Deutsch ---"
fi
echo "$SUMMARY_DE"
echo

jq -n \
  --arg url "$URL" \
  --arg id "$VIDEO_ID" \
  --arg title "$VIDEO_TITLE" \
  --arg channel "$VIDEO_CHANNEL" \
  --arg summary "$SUMMARY" \
  --arg summary_de "$SUMMARY_DE" \
  --argjson summary_tokens "$SUMMARY_TOKENS" \
  --argjson translate_tokens "$TRANSLATE_TOKENS" \
  --arg model "$SUMMARY_MODEL" \
  --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{timestamp: $ts, video_id: $id, title: $title, channel: $channel, url: $url, summary: $summary, summary_de: $summary_de, summary_tokens: $summary_tokens, translate_tokens: $translate_tokens, model: $model}' \
  >> "$LOG_FILE"

log "Logged to ${LOG_FILE}"
