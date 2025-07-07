#!/bin/bash
#SBATCH --time=10:00:00
# The name of the dlb
DLB_NAME=cnbl_study

# Parameter ranges (format: start:step:stop)
WS_RANGE="5.0 8.0 10.0 11.0 12.0 14.0"
TI_RANGE="0.16:0.0:0.16"
H_RANGE="150.0 200.0 250.0 300.0 400.0 500.0"
S_RANGE="4.0:2.0:10.0"
SEED_RANGE="1:1:10"

# Relevant HAWC2 sensors
SENS_LIST="10 12 15 17 26 29 32"

# Check to see if DLB folder exists
if [ ! -e ${DLB_NAME} ]; then
    echo "The study has not been created yet, run 'build_dlb_folder.sh ${DLB_NAME}' first"
    exit
fi

# Function to generate sequences
generate_sequence() {
    local range=$1
    IFS=':' read start step stop <<< "$range"
    python3 -c "start=$start; step=$step; stop=$stop; \
        n = int(round((stop - start) / step)) + 1 if step != 0 else 1; \
        print(' '.join(f'{start + i*step:.10g}' for i in range(n)))"
}

# Function to generate the W sequence based on H
generate_w_sequence() {
    local h=$1
    local w_range="0.02 0.05 0.10 0.15 0.20 0.25"

    python3 -c "
h = $h
w_range = list(map(float, '$w_range'.split()))
print(' '.join(str(round(h * w, 1)) for w in w_range))
"
}

# Step 0: Set environment variables etc.
echo "Setting environment..."
. /home/s244289/set_env

# Step 1: Generation of .stats files (no SENS loop needed)
echo "Generating .stats files..."
for WS in ${WS_RANGE}; do
    for TI in $(generate_sequence "$TI_RANGE"); do
        for H in ${H_RANGE}; do
            for W in $(generate_w_sequence "$H"); do
                for S in $(generate_sequence "$S_RANGE"); do

                    # Make .lst file for current params
                    RES_LIST="/home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_ws_${WS}_ti_${TI}_h_${H}_w_${W}_s_${S}.lst"
                    RES_LINE=""
                    RES_CNT=0

                    for SEED in $(generate_sequence "$SEED_RANGE"); do

                        # Generating .lst file entry 
                        TAG="${DLB_NAME}_ws_${WS}_ti_${TI}_h_${H}_w_${W}_s_${S}_seed_${SEED}"
                        RES_LINE="${RES_LINE} /home/s244289/dlbs/${DLB_NAME}/res/${TAG}.sel /home/s244289/dlbs/${DLB_NAME}/res/${TAG}.dat"
                        ((RES_CNT++))

                    done

                    # Compile the list file (all 10 seeds) and call getChannelStatistics
                    echo "${RES_CNT}${RES_LINE}" > "${RES_LIST}"
                    getChannelStatistics "${RES_LIST}" 7 ${SENS_LIST} -SQF > "/home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_ws_${WS}_ti_${TI}_h_${H}_w_${W}_s_${S}.stats"

                done
            done
        done
    done
done

# Step 2: Generation of .hist files
echo "Generating .hist files..."
for WS in ${WS_RANGE}; do
    for TI in $(generate_sequence "$TI_RANGE"); do
        for H in ${H_RANGE}; do
            for W in $(generate_w_sequence "$H"); do
                for S in $(generate_sequence "$S_RANGE"); do

                    # Make .hist file for param combo
                    HIST_FILE="/home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_ws_${WS}_ti_${TI}_h_${H}_w_${W}_s_${S}.hist"

                    for SENS in ${SENS_LIST}; do
                    
                        # Echo param combo into .hist file
                        echo "${DLB_NAME}_ws_${WS}_ti_${TI}_h_${H}_w_${W}_s_${S}_channel_${SENS}" >> "$HIST_FILE"
                        # Make a temporary file to store the time series
                        TMPFILE=$(mktemp)

                        for SEED in $(generate_sequence "$SEED_RANGE"); do

                            # Printing the time series into temporary file
                            TAG="${DLB_NAME}_ws_${WS}_ti_${TI}_h_${H}_w_${W}_s_${S}_seed_${SEED}"
                            HIST_LINE="/home/s244289/dlbs/${DLB_NAME}/res/${TAG}.sel /home/s244289/dlbs/${DLB_NAME}/res/${TAG}.dat"
                            printTimeSeries ${HIST_LINE} ${SENS} >> "$TMPFILE"

                        done

                        # Calculate the PDF data and append it to the .hist file; remove temporary file
                        freqStats pdf_data n_bin 30 <"$TMPFILE" >> "$HIST_FILE"
                        rm -f "$TMPFILE"

                    done
                done
            done
        done
    done
done

echo "Post-processing of ${DLB_NAME} complete!"