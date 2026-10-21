#!/usr/bin/env bash
#SBATCH --job-name=capacity_cycles
#SBATCH --output=logs/paper/capacity_cycles_%j.log
#SBATCH --error=logs/paper/capacity_cycles_%j.err
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --account=AIFAC_P02_1086
#SBATCH --partition=boost_usr_prod
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export IDP_CHECKPOINT_DIR="$PWD/.checkpoints/paper/capacity_cycles"
container_path=/leonardo/home/userexternal/sbetteti/Hopfield_Concentration/anaconda-wandb.sif
printf 'Started capacity-only rerun at %s\n' "$(date --iso-8601=seconds)"
srun singularity exec "$container_path" python experiments/run_capacity.py \
    --config configs/paper/capacity_cycles.yml \
    --output data/raw/paper/capacity_cycles --workers "$SLURM_CPUS_PER_TASK"
singularity exec "$container_path" python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
root=Path('data/raw/paper/capacity_cycles')
frame=pd.read_csv(root/'capacity_probability.csv')
assert set(frame.experiment)=={'scaled_cycle','fixed_cycle'}
for experiment, lo, hi in [('scaled_cycle',.5,1.5),('fixed_cycle',125.,225.)]:
    panel=frame[frame.experiment==experiment]
    assert panel.alpha_actual.between(lo,hi).all()
    assert set(panel.activation)=={'smoothf','tanh'}
    assert (panel.pattern_sets==10).all()
    assert np.isclose(panel.alpha_actual.min(),lo) and np.isclose(panel.alpha_actual.max(),hi)
assert len(frame[frame.experiment=='scaled_cycle'])==120
assert len(frame[frame.experiment=='fixed_cycle'])==40
assert (frame.groupby(['experiment','activation','N']).size()==20).all()
print('Validated both requested sweeps: 160 probability estimates.')
PY
srun singularity exec "$container_path" python plotting/plot_capacity.py \
    --input data/raw/paper/capacity_cycles --output-dir figures/paper --data-dir figures/data
srun singularity exec "$container_path" python plotting/plot_camera_ready_cycles.py \
    --input data/raw/paper/capacity_cycles/capacity_probability.csv \
    --output-dir figures/camera_ready
printf 'Completed capacity-only rerun and figure exports at %s\n' "$(date --iso-8601=seconds)"
