#!/bin/bash
set -euo pipefail

# === Config ===
SRC="/zata/public_html/projects/downloads/psychscreen/spatial"
DEST="/data/zusers/kresgeb/psychscreen_spatial_backup_2025_10_08"
# Dry-run flag (uncomment -n for dry-run)
# DRY_RUN="-n"
DRY_RUN=""

# Ensure destination exists
mkdir -p "$DEST"

# Flags for rsync
RSYNC_FLAGS="-aHAXx --inplace --numeric-ids --compress --ignore-existing --human-readable --mkpath $DRY_RUN --info=progress2"

echo "Starting rsync from $SRC to $DEST"

# Move into source directory
cd "$SRC"

# Copy all files and directories; rsync prints its own progress
rsync $RSYNC_FLAGS "$SRC"/ "$DEST"/

echo "Backup completed safely."
