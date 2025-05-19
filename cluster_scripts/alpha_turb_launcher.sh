#!/bin/bash

# Settings
DLB_NAME=alpha_study
PARAM_FILE="${DLB_NAME}_param_combos.txt"
MAX_PARALLEL_JOBS=10

# Parameter ranges
WS_RANGE="4.0:2.0:24.0"
TI_RANGE="0.16:0.0:0.16"
SHEAR_RANGE="0.1:0.05:0.5"
SEED_RANGE="1:1:10"

# Step 0. Set environment variables etc.
. /home/s244289/set_env

# Step 1: Build folder structure
build_dlb_folder.sh "$DLB_NAME"

# Step 2: Build htc files (submit as batch job)
echo "Submitting htc file generation as batch job..."
sbatch /home/s244289/dlbs/alpha_build_htc_files.sh

# Step 3: Generate parameter combinations
echo "Generating parameter combinations..."
> "$PARAM_FILE"  # Clears the file

generate_sequence() {
    local range=$1
    IFS=':' read start step stop <<< "$range"
    python3 -c "start=$start; step=$step; stop=$stop; \
        n = int(round((stop - start) / step)) + 1 if step != 0 else 1; \
        print(' '.join(f'{start + i*step:.10g}' for i in range(n)))"
}

for WS in $(generate_sequence "$WS_RANGE"); do
    for TI in $(generate_sequence "$TI_RANGE"); do
        for SHEAR in $(generate_sequence "$SHEAR_RANGE"); do
            for SEED in $(generate_sequence "$SEED_RANGE"); do
                echo "$WS $TI $SHEAR $SEED" >> "$PARAM_FILE"
            done
        done
    done
done

NUM_TASKS=$(wc -l < "$PARAM_FILE")

echo "Generated $NUM_TASKS combinations."

# Step 4: Submit job array
mkdir -p "slurm_log"
echo "Submitting $NUM_TASKS jobs with max $MAX_PARALLEL_JOBS parallel jobs..."
ARR_JOB_ID=$(sbatch --array=0-$(($NUM_TASKS - 1))%$MAX_PARALLEL_JOBS alpha_build_turb_box.sh | awk '{print $4}')

# Step 5: Submit Step 3 with dependency on successful completion of the job array
echo "Renaming and moving turb files after the job array completion..."
sbatch --dependency=afterok:$ARR_JOB_ID --wrap="bash move_turb_files.sh $DLB_NAME"
