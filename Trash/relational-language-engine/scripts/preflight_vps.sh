#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:-./runtime}
MIN_FREE_GB=${MIN_FREE_GB:-10}
MIN_FD=${MIN_FD:-4096}

fail=0
say() { printf '%-34s %s\n' "$1" "$2"; }

cpu=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf 0)
mem_kb=$(awk '/MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || printf 0)
mem_gb=$((mem_kb / 1024 / 1024))
mkdir -p "$DATA_DIR"
free_kb=$(df -Pk "$DATA_DIR" | awk 'NR==2 {print $4}')
free_gb=$((free_kb / 1024 / 1024))
fd=$(ulimit -n)

say 'Online vCPUs' "$cpu"
say 'Physical/virtual RAM (GiB)' "$mem_gb"
say 'Free data-disk space (GiB)' "$free_gb"
say 'Open-file soft limit' "$fd"
say 'Filesystem' "$(df -PT "$DATA_DIR" | awk 'NR==2 {print $2}')"
say 'Kernel' "$(uname -sr)"

if (( cpu < 4 )); then say 'CPU gate' 'WARN: fewer than 4 online vCPUs'; fi
if (( mem_gb < 7 )); then say 'RAM gate' 'WARN: less than ~8 GB installed'; fi
if (( free_gb < MIN_FREE_GB )); then say 'Disk gate' "FAIL: need at least ${MIN_FREE_GB} GiB free"; fail=1; fi
if [[ "$fd" != unlimited ]] && (( fd < MIN_FD )); then say 'FD gate' "FAIL: set LimitNOFILE >= ${MIN_FD}"; fail=1; fi

# Verify atomic replacement and fsync on the selected filesystem.
tmp="$DATA_DIR/.rlm-preflight.$$"
printf 'a' > "$tmp.a"
sync "$tmp.a" 2>/dev/null || true
mv "$tmp.a" "$tmp.b"
python3 - "$tmp.b" <<'PY'
import os, sys
p=sys.argv[1]
fd=os.open(p, os.O_RDONLY)
os.fsync(fd)
os.close(fd)
dfd=os.open(os.path.dirname(os.path.abspath(p)), os.O_RDONLY)
os.fsync(dfd)
os.close(dfd)
PY
rm -f "$tmp.b"
say 'rename/fsync gate' 'PASS'

exit "$fail"
