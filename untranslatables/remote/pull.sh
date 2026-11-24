#!/usr/bin/env bash
# Pull results and logs (not the activation cache) back to the laptop.
source "$(dirname "$0")/_common.sh"
for d in results logs data/pools; do mkdir -p "$REPO/$d"; rsync -azu --exclude "cache/" "$DESKTOP:$REMOTE_DIR/$d/" "$REPO/$d/"; done
echo "pulled"
