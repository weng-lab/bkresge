#!/bin/bash
set -euo pipefail

# === Config ===
SRC="/zata/public_html/users/kresgeb/psych_screen"
DEST="/zata/public_html/users/kresgeb/psych_screen_backup_2025_09_18"
# Dry-run flag (uncomment -n for dry-run)
# DRY_RUN="-n"
DRY_RUN=""

# Ensure destination exists
mkdir -p "$DEST"

# Flags for rsync
RSYNC_FLAGS="-aHAXx --inplace --numeric-ids --no-compress --ignore-existing --human-readable --mkpath $DRY_RUN --info=progress2"

echo "Starting rsync from $SRC to $DEST"

# Move into source directory
cd "$SRC"

# Copy all files and directories; rsync prints its own progress
rsync $RSYNC_FLAGS "$SRC"/ "$DEST"/

echo "Backup completed safely."
