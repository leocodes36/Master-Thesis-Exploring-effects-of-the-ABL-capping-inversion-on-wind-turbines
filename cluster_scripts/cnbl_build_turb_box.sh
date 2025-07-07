#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --output=./slurm_log/slurm-%j.out
#SBATCH --error=./slurm_log/slurm-%j.err

# ----------- SETTINGS ------------------

# Fallbacks if not passed from sbatch
DLB_NAME=${DLB_NAME:-cnbl_study}
PARAM_FILE=${PARAM_FILE:-"${DLB_NAME}_param_combos.txt"}
OFFSET=${OFFSET:-0}

TIME_STOP=600.0
TIME_STEP=1.0
NX=1024
NY=180
NZ=180
DY=1
DZ=1
HF=1

# Set environment variables etc.
. /home/s244289/set_env

# ----------- GET PARAMS ----------------

if [ ! -f "$PARAM_FILE" ]; then
    echo "No parameter combinations text file found: $PARAM_FILE"
    exit 1
fi

# Get the actual index to read from the param file
IDX=$((SLURM_ARRAY_TASK_ID + OFFSET))

# Read parameter line at index (1-based line number for sed)
read WS TI H W S SEED < <(sed -n "$((IDX + 1))p" "$PARAM_FILE")

# Sanity check
if [ -z "$WS" ]; then
    echo "Failed to read parameters for index $IDX from $PARAM_FILE"
    exit 2
fi

# ----------- RUN PYTHON ----------------

DX=$(echo "${WS}*${TIME_STOP}/${NX}" | bc -l)
TAG="${DLB_NAME}_ws_${WS}_ti_${TI}_h_${H}_w_${W}_s_${S}_seed_"

python3 /home/s244289/py/dlb_generator/generate_box_cnbl.py \
    "$WS" "$TI" "$TIME_STEP" "$NX" "$NY" "$NZ" "$DX" "$DY" "$DZ" "$HF" "$TAG" "$SEED" "$H" "$W" "$S"
