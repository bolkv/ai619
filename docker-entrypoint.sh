#!/usr/bin/env bash
# Container entrypoint: ensure MONAI bundles are present, then hand off
# to whatever command the user passed (defaults to bash via CMD).
set -e

python /workspace/setup_bundles.py --bundles-only

exec "$@"
