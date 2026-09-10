#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BACKUP.tar.gz TARGET_DATA_DIR" >&2
  exit 64
fi
backup=$1
target=$2
[[ -f "$backup" ]] || { echo "backup not found: $backup" >&2; exit 66; }
mkdir -p "$target"
[[ -z "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
  echo "target must be empty: $target" >&2
  exit 73
}
tar -tzf "$backup" >/dev/null
staging="${target}.restore.$$"
rm -rf "$staging"
mkdir -p "$staging"
tar -xzf "$backup" -C "$staging"
# Refuse archives with path traversal or unexpected absolute entries.
if tar -tzf "$backup" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
  rm -rf "$staging"
  echo 'unsafe archive path' >&2
  exit 65
fi
shopt -s dotglob nullglob
items=("$staging"/*)
((${#items[@]} > 0)) || { rm -rf "$staging"; echo 'empty backup' >&2; exit 65; }
mv "${items[@]}" "$target"/
rmdir "$staging"
echo "restored into $target"
