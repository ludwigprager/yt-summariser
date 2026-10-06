FROM python:3.11-slim

# ffmpeg: required by yt-dlp for audio extraction and by faster-whisper's
# decoding path. curl+jq: talk to Ollama's HTTP API and build/parse JSON.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg curl jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      faster-whisper

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

WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
