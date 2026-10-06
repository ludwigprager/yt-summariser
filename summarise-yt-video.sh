#!/usr/bin/env bash
#
# Usage: ./summarise-yt-video.sh <youtube-video-id> [model]
#
# Wraps `docker compose run` so you can pass a bare video ID instead of
# building the full URL yourself.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VIDEO_ID="${1:?Usage: $0 <youtube-video-id> [model]}"
URL="https://www.youtube.com/watch?v=${VIDEO_ID}"

mkdir -p data whisper-cache

args=("$URL")
[[ -n "${2:-}" ]] && args+=("$2")

exec docker compose run --rm yt-summarize "${args[@]}"
