#!/bin/bash
#SBATCH --time=02:00:00 # Allow 2 hours wall clock time limit

if [ $# -ne 5 ]; then
  echo "Usage: $0 <WS> <TI> <N> <A> <SEED>"
  exit 1
fi

# Step 0. Set environment variables etc.
. /home/s244289/set_env

# Settings
DLB_NAME=wave2_study

# Fixed parameters
TIME_STOP=600.0
TIME_STEP=1.0
NX=1024
NY=30
NZ=30
DY=6
DZ=6
HF=1

# Get parameters for this task from arguments
WS=$1
TI=$2
N=$3
A=$4
SEED=$5

DX=$(echo "${WS}*${TIME_STOP}/${NX}" | bc -l)
TAG="${DLB_NAME}_ws_${WS}_ti_${TI}_n_${N}_a_${A}_seed_"

python /home/s244289/py/dlb_generator/plot_box_wave.py "$WS" "$TI" "$TIME_STEP" "$NX" "$NY" "$NZ" "$DX" "$DY" "$DZ" "$HF" "$TAG" "$SEED" "$N" "$A"
