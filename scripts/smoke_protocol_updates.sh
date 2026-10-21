#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export IDP_SMOKE=1
OUT="$ROOT/data/raw/smoke"
FIG="$ROOT/figures/smoke"
DATA="$ROOT/figures/data/smoke"
mkdir -p "$OUT" "$FIG" "$DATA"
python "$ROOT/experiments/run_capacity.py" --config "$ROOT/configs/smoke/capacity.yml" --output "$OUT/capacity" --workers 1
python "$ROOT/plotting/plot_capacity.py" --input "$OUT/capacity" --output-dir "$FIG" --data-dir "$DATA"
python "$ROOT/experiments/run_correlation_phase_plane.py" --config "$ROOT/configs/smoke/correlation_phase_plane.yml" --output "$OUT/correlation_phase_plane"
python "$ROOT/plotting/plot_correlation_phase_plane.py" --input "$OUT/correlation_phase_plane" --output "$FIG/correlation_phase_plane" --data-dir "$DATA"
pytest -q "$ROOT/tests/test_capacity_protocols.py" "$ROOT/tests/test_correlation_dynamics.py"
python "$ROOT/scripts/check_outputs.py" "$OUT"
echo "Protocol-update smoke test completed."
