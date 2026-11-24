#!/usr/bin/env bash
# Shared settings for remote/*.sh (same conventions as Dissertation/code/remote).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/remote/remote.env"
: "${DESKTOP:?}" "${REMOTE_DIR:?}"
rsh() { ssh -o ConnectTimeout=10 "$DESKTOP" "bash -lc $(printf '%q' "$*")"; }
