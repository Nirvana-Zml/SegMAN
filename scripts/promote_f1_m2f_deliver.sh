#!/usr/bin/env bash
# Legacy wrapper — use promote_dual_track_deliver.sh for V1.0.0 dual-track.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Redirecting to scripts/promote_dual_track_deliver.sh ..."
exec bash scripts/promote_dual_track_deliver.sh
