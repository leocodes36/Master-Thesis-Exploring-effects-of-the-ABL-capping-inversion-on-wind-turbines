#!/bin/bash

# The name of the dlb
DLB_NAME=unconstrained_study

# Parameter ranges (format: start:step:stop)
WS_RANGE="5.0:1.0:25.0"
TI_RANGE="0.16:0:0.16"
SEED_RANGE="1:1:10"

# Relevant HAWC2 sensors
SENS_LIST="17 26 29 32"

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

# Loop through wind speed levels — aggregate stats over matching WS values

for WS in $(generate_sequence "$WS_RANGE"); do
    RES_CNT=0
    RES_LINE=""
    RES_LIST="/home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_ws_${WS}.lst"

    for TI in $(generate_sequence "$TI_RANGE"); do
		for SEED in $(generate_sequence "$SEED_RANGE"); do

			TAG="${DLB_NAME}_ws_${WS}_ti_${TI}_seed_${SEED}"
			RES_LINE="${RES_LINE} /home/s244289/dlbs/${DLB_NAME}/res/${TAG}.sel /home/s244289/dlbs/${DLB_NAME}/res/${TAG}.dat"
			HIST_LINE="/home/s244289/dlbs/${DLB_NAME}/res/${TAG}.sel /home/s244289/dlbs/${DLB_NAME}/res/${TAG}.dat"

			for SENS in ${SENS_LIST}; do
				echo "${TAG}_channel_${SENS}" >> "/home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_ws_${WS}.hist"
				printTimeSeries ${HIST_LINE} ${SENS} | freqStats pdf_data >> "/home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_ws_${WS}.hist"
			done

			let RES_CNT=${RES_CNT}+1
		done
    done

    echo "${RES_CNT}${RES_LINE}" > "${RES_LIST}"
    getChannelStatistics "${RES_LIST}" 4 ${SENS_LIST} > "/home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_ws_${WS}.stats"
done

echo "Post-processing complete!"
