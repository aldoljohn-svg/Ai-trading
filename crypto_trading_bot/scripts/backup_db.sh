#!/usr/bin/env bash
# Back up the trading database.
#
#   ./scripts/backup_db.sh                 -> backups/trading_bot_<ts>.db.gz
#   ./scripts/backup_db.sh /mnt/backups
#
# Uses sqlite3's online .backup when available so a running bot is not
# interrupted and the copy is never torn mid-write.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/backups}"
DB="${DB_PATH:-$ROOT/data/trading_bot.db}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
KEEP="${KEEP_BACKUPS:-30}"

if [[ ! -f "$DB" ]]; then
  echo "no database at $DB" >&2
  exit 1
fi

mkdir -p "$DEST"
OUT="$DEST/trading_bot_${STAMP}.db"

if command -v sqlite3 >/dev/null 2>&1; then
  # Online backup: consistent even while the bot is writing.
  sqlite3 "$DB" ".backup '$OUT'"
else
  echo "sqlite3 not found - falling back to a file copy." >&2
  echo "Stop the bot first if you want a guaranteed-consistent copy." >&2
  cp "$DB" "$OUT"
fi

gzip -f "$OUT"
echo "backup written: ${OUT}.gz"

# Retention
COUNT=$(find "$DEST" -name 'trading_bot_*.db.gz' -type f | wc -l)
if (( COUNT > KEEP )); then
  find "$DEST" -name 'trading_bot_*.db.gz' -type f -printf '%T@ %p\n' \
    | sort -n | head -n "$(( COUNT - KEEP ))" | cut -d' ' -f2- \
    | xargs -r rm --
  echo "pruned old backups, keeping the newest $KEEP"
fi
