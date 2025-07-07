#!/bin/bash

# Settings
DLB_NAME=wave2_study
PARAM_FILE="${DLB_NAME}_param_combos.txt"
MAX_PARALLEL_JOBS=12 #adapt as needed
MAX_ARRAY_SIZE=1000  # cluster's limit

# Define parameter ranges as list or: start:step:stop
WS_RANGE="5.0 8.0 10.0 11.0 12.0 14.0"
TI_RANGE="0.16:0.02:0.18"
N_RANGE="0.01 0.03 0.06 0.1"
A_RANGE="2.0"
SEED_RANGE="1:1:10"

# Step 0. Set environment variables etc.
. /home/s244289/set_env

# Step 1: Build folder structure (if not already existing)
if [ -d "$DLB_NAME" ]; then
    echo "Folder structure for $DLB_NAME already exists. Skipping..."
else
    echo "Building folder structure..."
    build_dlb_folder.sh "$DLB_NAME"
fi

# Step 2: Build htc files (if not already existing)
HTC_DIR="$DLB_NAME/htc"

if ls "$HTC_DIR"/*.htc 1> /dev/null 2>&1; then
    echo "HTC files already exist in $HTC_DIR. Skipping..."
else
    echo "Submitting HTC file generation as batch job..."
    sbatch /home/s244289/dlbs/wave2_build_htc_files.sh
fi

# Step 3: Generate parameter combinations (if param file not already existing)
generate_sequence() {
    local range=$1
    IFS=':' read start step stop <<< "$range"
    python3 -c "start=$start; step=$step; stop=$stop; \
        n = int(round((stop - start) / step)) + 1 if step != 0 else 1; \
        print(' '.join(f'{start + i*step:.10g}' for i in range(n)))"
}

if [ -s "$PARAM_FILE" ]; then
    echo "Parameter file $PARAM_FILE already exists. Skipping generation..."

    NUM_TASKS=$(wc -l < "$PARAM_FILE")
    echo "Read $NUM_TASKS combinations."

    if [ "$NUM_TASKS" -le 0 ]; then
        echo "No tasks to submit. Exiting."
        exit 1
    fi

else
    echo "Generating parameter combinations..."
    > "$PARAM_FILE"

    for WS in $WS_RANGE; do
        for TI in $(generate_sequence "$TI_RANGE"); do
            for N in $N_RANGE; do
                for A in $A_RANGE; do
                    for SEED in $(generate_sequence "$SEED_RANGE"); do
                        echo "$WS $TI $N $A $SEED" >> "$PARAM_FILE"
                    done
                done
            done
        done
    done

    NUM_TASKS=$(wc -l < "$PARAM_FILE")
    echo "Generated $NUM_TASKS combinations."

    if [ "$NUM_TASKS" -le 0 ]; then
        echo "No tasks to submit. Exiting."
        exit 1
    fi
fi

# Step 4: Prepare chunks and ask user which chunk to submit
mkdir -p "slurm_log"
echo "Preparing job chunks (max $MAX_ARRAY_SIZE per chunk)..."

CHUNK_START=0
CHUNK_INDEX=0
CHUNK_FILES=()

while [ $CHUNK_START -lt $NUM_TASKS ]; do
    CHUNK_END=$((CHUNK_START + MAX_ARRAY_SIZE - 1))
    if [ $CHUNK_END -ge $NUM_TASKS ]; then
        CHUNK_END=$((NUM_TASKS - 1))
    fi

    CHUNK_SIZE=$((CHUNK_END - CHUNK_START + 1))
    CHUNK_FILES+=("$CHUNK_START $CHUNK_SIZE")  # store offset and size
    echo "Chunk $CHUNK_INDEX: $CHUNK_START-$CHUNK_END ($CHUNK_SIZE jobs)"

    CHUNK_START=$((CHUNK_END + 1))
    CHUNK_INDEX=$((CHUNK_INDEX + 1))
done

# Ask user which chunk to submit
read -p "Enter chunk index to submit (0 to $((${#CHUNK_FILES[@]} - 1))): " SELECTED_INDEX

SELECTED_INFO=(${CHUNK_FILES[$SELECTED_INDEX]})
CHUNK_OFFSET=${SELECTED_INFO[0]}
CHUNK_SIZE=${SELECTED_INFO[1]}

echo "Submitting chunk with offset $CHUNK_OFFSET, size $CHUNK_SIZE..."
JOB_ID=$(sbatch --array=0-$(($CHUNK_SIZE - 1))%$MAX_PARALLEL_JOBS \
    --export=DLB_NAME="$DLB_NAME",OFFSET="$CHUNK_OFFSET",PARAM_FILE="$PARAM_FILE" \
    --ntasks=1 \
    --mem=8G \
    wave2_build_turb_box.sh | awk '{print $4}')

# Step 5: Submit post-processing for this chunk only
echo "Submitting post-processing job with dependency on job $JOB_ID"
sbatch --dependency=afterok:$JOB_ID --wrap="bash move_turb_files.sh $DLB_NAME"
