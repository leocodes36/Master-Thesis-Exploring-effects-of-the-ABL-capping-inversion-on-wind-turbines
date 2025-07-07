#!/bin/bash

if [ $# -ne 5 ]; then
  echo "Usage: $0 <WS> <TI> <N> <A> <SEED>"
  exit 1
fi

. /home/s244289/set_env

# Fixed values
DLB_NAME=wave2_study
TIME_STOP=600.0
TIME_STEP=1.0
NX=1024
NY=30
NZ=30
DY=6
DZ=6
HF=1

# Inputs
WS=$1
TI=$2
N=$3
A=$4
SEED=$5

DX=$(echo "${WS}*${TIME_STOP}/${NX}" | bc -l)
TAG="${DLB_NAME}_ws_${WS}_ti_${TI}_n_${N}_a_${A}_seed_${SEED}"

# Path to bin directory
BIN_DIR="/home/s244289/dlbs/${DLB_NAME}/turb/"

# Call Python script
python /home/s244289/py/dlb_generator/replot_box_wave2.py \
  --tag "$TAG" \
  --bin_dir "$BIN_DIR" \
  --shape "$NX" "$NY" "$NZ" \
  --dx "$DX"