#!/usr/bin/env bash
# Push code + data to the desktop. Never deletes generated outputs there (results/ and cache/ are excluded).
source "$(dirname "$0")/_common.sh"
rsh "mkdir -p $REMOTE_DIR"
rsync -azu --delete --exclude '.git/' --exclude '.venv-cpu/' --exclude '.venv/' --exclude 'results/' --exclude 'cache/' --exclude 'logs/' \
  --exclude '__pycache__/' --exclude 'remote/remote.env' "$REPO/" "$DESKTOP:$REMOTE_DIR/"
echo "synced -> $DESKTOP:$REMOTE_DIR"
