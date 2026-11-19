#!/usr/bin/env bash
# Usage: GPU=0|1 [VAR=... ...] ./post_review/run_pr.sh post_review/<script>.py [args]; runs inside the mechinterp FHS like ../run.sh, on the chosen card.
cd /data/mechinterp/wdd || exit 1
. ./env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
FHS=$(grep -o '/nix/store/[^ "]*/bin/mechinterp-fhs' /nix/store/vf6dyqqyf4b4g0vld0cy8svrfi8asp2l-unit-script-mechinterp-start/bin/mechinterp-start | head -1)
[ -x "$FHS" ] || { echo "mechinterp-fhs not found"; exit 1; }
exec "$FHS" -c "/data/mechinterp/venv/bin/python $*"
