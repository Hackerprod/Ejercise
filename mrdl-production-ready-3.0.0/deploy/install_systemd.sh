#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo 'run as root: sudo deploy/install_systemd.sh' >&2
  exit 1
fi
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v /usr/local/bin/mrdl >/dev/null || {
  echo '/usr/local/bin/mrdl not found; install the release build first' >&2
  exit 1
}

getent group mrdl >/dev/null || groupadd --system mrdl
id -u mrdl >/dev/null 2>&1 || useradd --system --gid mrdl --home /var/lib/mrdl --shell /usr/sbin/nologin mrdl
install -d -o root -g mrdl -m 0750 /etc/mrdl
install -d -o mrdl -g mrdl -m 0750 /var/lib/mrdl/model /var/lib/mrdl/corpus /var/backups/mrdl
if [[ ! -e /etc/mrdl/mrdl.ini ]]; then
  install -o root -g mrdl -m 0640 "$ROOT_DIR/config/vps-system.ini" /etc/mrdl/mrdl.ini
fi
install -o root -g root -m 0755 "$ROOT_DIR/scripts/backup.sh" /usr/local/libexec/mrdl-backup
install -o root -g root -m 0644 "$ROOT_DIR"/deploy/mrdl-*.service "$ROOT_DIR"/deploy/mrdl-*.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable mrdl-audit.timer mrdl-gc.timer mrdl-backup.timer
printf 'Installed. Review /etc/mrdl/mrdl.ini, then start timers with:\n'
printf '  systemctl start mrdl-audit.timer mrdl-gc.timer mrdl-backup.timer\n'
