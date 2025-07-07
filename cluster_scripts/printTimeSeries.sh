#!/bin/bash
set -euo pipefail

# --- Configuration ---
. /home/s244289/set_env

if [ $# -le 3 ]; then
  echo "Usage: $0 <STUDY> <PARAMS...> <SEED>"
  echo "Example: $0 DLB ws 11.0 ti 0.16 n 0.01 a 2.0 h 162.0 w 12.0 1"
  exit 1
fi

STUDY=$1
PARAM_ARGS=("${@:2:$#-2}")
SEED="${@: -1}"

# Validate that we have an even number of param arguments
if (( ${#PARAM_ARGS[@]} % 2 != 0 )); then
  echo "Error: PARAMS should be provided in symbol-value pairs."
  exit 1
fi

# Build param string
PARAM_STRING=""
i=0
while [ $i -lt ${#PARAM_ARGS[@]} ]; do
  SYMBOL="${PARAM_ARGS[$i]}"
  VALUE="${PARAM_ARGS[$((i+1))]}"
  PARAM_STRING+="${SYMBOL}_${VALUE}_"
  i=$((i+2))
done

DLB_NAME="${STUDY}_study"
TAG="${DLB_NAME}_${PARAM_STRING}seed_${SEED}"
TAG="${TAG%_}"  # remove trailing underscore

RES_DIR="/home/s244289/dlbs/${DLB_NAME}/res"
SEL_FILE="${RES_DIR}/${TAG}.sel"
DAT_FILE="${RES_DIR}/${TAG}.dat"

SENS_LIST=(10 13 15 17 26 29 32 44 46 48)  # sensors to extract besides time (sensor 1 is time)

# --- Check if input files exist ---
if [[ ! -f "$SEL_FILE" || ! -f "$DAT_FILE" ]]; then
  echo "Error: Required files not found."
  [[ ! -f "$SEL_FILE" ]] && echo "  Missing: $SEL_FILE"
  [[ ! -f "$DAT_FILE" ]] && echo "  Missing: $DAT_FILE"
  echo ""
  echo "Usage: $0 <STUDY> <PARAMS...> <SEED>"
  echo "Example: $0 DLB ws 11.0 ti 0.16 n 0.01 a 2.0 h 162.0 w 12.0 1"
  exit 1
fi

# --- Step 1: Extract time column from sensor 1 ---
printTimeSeries "$SEL_FILE" "$DAT_FILE" 1 > tmp_time.txt

# --- Step 2: Extract each other sensor's data ---
HEADER=("time")
for SENS in "${SENS_LIST[@]}"; do
  printTimeSeries "$SEL_FILE" "$DAT_FILE" "$SENS" > "tmp_$SENS.txt"
  HEADER+=("$SENS")
done

# --- Step 3: Combine columns ---
paste tmp_time.txt $(for S in "${SENS_LIST[@]}"; do echo "tmp_$S.txt"; done) > "TimeSeries_${TAG}.txt"

# --- Step 4: Add header line ---
HEADER_LINE=$(IFS=$'\t'; echo "${HEADER[*]}")
sed -i "1i $HEADER_LINE" "TimeSeries_${TAG}.txt"

# --- Step 5: Move time series into directory ---
TS_DIR="timeseries"
mkdir -p "$TS_DIR"
mv "TimeSeries_${TAG}.txt" "$TS_DIR/"

echo "Time series written to $TS_DIR/TimeSeries_${TAG}.txt"

# --- Cleanup ---
rm -f tmp_time.txt tmp_*.txt