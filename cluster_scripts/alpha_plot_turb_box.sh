#!/bin/bash
#SBATCH --time=02:00:00 # Allow 2 hours wall clock time limit
#SBATCH --output=./slurm_log/slurm-%j.out  # redirect SLURM outputs
#SBATCH --error=./slurm_log/slurm-%j.err   # redirect SLURM errors

if [ $# -ne 4 ]; then
  echo "Usage: $0 <WS> <TI> <SHEAR> <SEED>"
  exit 1
fi

# Step 0. Set environment variables etc.
. /home/s244289/set_env

# Settings
DLB_NAME=alpha_study

# Fixed parameters
TIME_STOP=600.0
TIME_STEP=1.0
NX=1024
NY=180
NZ=180
DY=1
DZ=1
HF=1

# Get parameters for this task from arguments
WS=$1
TI=$2
SHEAR=$3
SEED=$4

DX=$(echo "${WS}*${TIME_STOP}/${NX}" | bc -l)
TAG="${DLB_NAME}_ws_${WS}_ti_${TI}_shear_${SHEAR}_seed_"

python /home/s244289/py/dlb_generator/plot_box_alpha.py "$WS" "$TI" "$TIME_STEP" "$NX" "$NY" "$NZ" "$DX" "$DY" "$DZ" "$HF" "$TAG" "$SEED" "$SHEAR"
