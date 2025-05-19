#!/bin/bash
#SBATCH --time=02:00:00 # Allow 2 hours wall clock time limit
#SBATCH --output=./slurm_log/slurm-%j.out  # redirect SLURM outputs
#SBATCH --error=./slurm_log/slurm-%j.err   # redirect SLURM errors

# Settings
DLB_NAME=alpha_study
PARAM_FILE="${DLB_NAME}_param_combos.txt"

if [ ! -f "$PARAM_FILE" ]; then
    echo "No parameter combinations text file found"
    exit
fi

TIME_STOP=600.0
TIME_STEP=1.0
NX=1024
NY=180
NZ=180
DY=1
DZ=1
HF=1

# Get parameters for this task
read WS TI SHEAR SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAM_FILE")

DX=$(echo "${WS}*${TIME_STOP}/${NX}" | bc -l)
TAG="${DLB_NAME}_ws_${WS}_ti_${TI}_shear_${SHEAR}_seed_"

python /home/s244289/py/dlb_generator/generate_box_alpha.py "$WS" "$TI" "$TIME_STEP" "$NX" "$NY" "$NZ" "$DX" "$DY" "$DZ" "$HF" "$TAG" "$SEED" "$SHEAR"
