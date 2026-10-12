#!/bin/sh
# Runs as root, before the package's files are removed. Stops the guardian
# first: otherwise its unit file vanishes under a running process, which
# keeps reviving Focuser until the next reboot.
set -e

if command -v systemctl >/dev/null 2>&1; then
    systemctl disable --now focuser-guardian.service >/dev/null 2>&1 || true
fi

exit 0
