#!/usr/bin/env bash
# Pull generated data, results, lattices and logs back (never deletes locally).
source "$(dirname "$0")/_common.sh"
for d in data runs logs; do mkdir -p "$REPO/$d"; rsync -azu "$DESKTOP:$REMOTE_DIR/$d/" "$REPO/$d/"; done
echo "pulled"
