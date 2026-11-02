#!/usr/bin/env bash
# Read-only: print local working tree vs origin/main, to refresh docs/LOCAL_CHANGES_vs_REMOTE.md.
# Usage: bash scripts/diff_vs_remote.sh [--fetch]
# Never commits/pushes/resets; --fetch only runs `git fetch origin`.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "--fetch" ]; then
  git fetch origin || echo "[warn] git fetch failed; comparing against last fetched origin/main"
fi

echo "== snapshot: $(date '+%F %T %z')"
echo "== origin/main: $(git log -1 --format='%h %ci %s' origin/main)"
echo "== HEAD:        $(git log -1 --format='%h %ci %s' HEAD)"
echo "== ahead (origin/main..HEAD):"
git log --oneline origin/main..HEAD
echo "== behind (HEAD..origin/main):"
git log --oneline HEAD..origin/main

echo
echo "== tracked changes vs origin/main (CRLF ignored):"
git diff --ignore-cr-at-eol --stat origin/main
echo "== tracked files that differ only by line endings:"
for f in $(git diff --name-only origin/main); do
  if [ -z "$(git diff --ignore-cr-at-eol origin/main -- "$f")" ]; then echo "  $f"; fi
done

echo
echo "== untracked files (size bytes, mtime):"
git status --porcelain --untracked-files=all -z | tr '\0' '\n' | sed -n 's/^?? //p' | while IFS= read -r f; do
  printf '  %12s  %s  %s\n' "$(stat -c %s "$f" 2>/dev/null)" "$(date -r "$f" '+%F %T' 2>/dev/null)" "$f"
done
