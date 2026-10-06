# Build context is the repo root (see docker-compose.yaml: build.context: .)
# so this can reuse entrypoint.sh unmodified instead of duplicating its logic.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg curl jq \
    && rm -rf /var/lib/apt/lists/*

# --timeout/--retries ride out a stalled read from the PyPI CDN: this layer
# pulls ~500 MB (ctranslate2, av, onnxruntime, numpy), and pip's 15s default
# turns a brief hiccup into a failed build that restarts the whole download.
RUN pip install --no-cache-dir --timeout 60 --retries 10 \
      faster-whisper \
      flask \
      gunicorn

# yt-dlp gets its own layer, after the heavy one, so refreshing it doesn't
# re-download everything else. YouTube now makes the downloader solve a
# JavaScript challenge; without a JS runtime every download ends in "HTTP
# Error 403: Forbidden". [default] brings yt-dlp-ejs (the challenge solver
# scripts), and the deno package is the runtime yt-dlp looks for. The ADD
# of PyPI's release info busts the build cache whenever a new yt-dlp is
# released, since YouTube breaks old versions and a cached layer would
# otherwise keep one forever.
ADD https://pypi.org/pypi/yt-dlp/json /tmp/yt-dlp-release.json
RUN pip install --no-cache-dir --timeout 60 --retries 10 --upgrade \
      "yt-dlp[default]" \
      deno

# docker CLI: jobs run g1-custodes (in its own container) via `docker exec`
# to get the GPU for Ollama, see OllamaLease in app.py
COPY --from=docker:28-cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

COPY web/app.py /app/app.py
COPY web/templates /app/templates

EXPOSE 8000
CMD ["gunicorn", "--worker-class", "gthread", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:8000", "app:app"]
