#!/usr/bin/env bash
# Push code to the desktop (mirror), and push data/ without deleting anything generated there.
source "$(dirname "$0")/_common.sh"
rsh "mkdir -p $REMOTE_DIR/data"
rsync -az --delete --exclude '.venv/' --exclude 'data/' --exclude 'results/' --exclude 'out/' --exclude 'runs/' \
  --exclude 'logs/' --exclude '__pycache__/' --exclude 'remote/remote.env' --exclude '*.zip' \
  "$REPO/" "$DESKTOP:$REMOTE_DIR/"
rsync -azu "$REPO/data/" "$DESKTOP:$REMOTE_DIR/data/"
echo "synced -> $DESKTOP:$REMOTE_DIR"
