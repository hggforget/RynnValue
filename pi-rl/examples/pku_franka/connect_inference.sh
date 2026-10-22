#!/usr/bin/env bash
set -euo pipefail

# Run on the GPU server. The robot machine accesses its own loopback ports.
exec ssh -N -T \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -R 127.0.0.1:19080:127.0.0.1:8000 \
    -R 127.0.0.1:19081:127.0.0.1:8001 \
    "${ROBOT_HOST:-pku-franka-hdc}"
