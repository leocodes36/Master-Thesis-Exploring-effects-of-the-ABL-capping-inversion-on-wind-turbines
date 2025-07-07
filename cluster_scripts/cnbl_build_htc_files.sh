#! /bin/bash

# The name of the dlb
DLB_NAME=cnbl_study
# The length of the simulation
TIME_STOP=600.0

# Define parameter ranges as: start:step:stop
WS_RANGE="5.0 8.0 10.0 11.0 12.0 14.0"
TI_RANGE="0.16:0.0:0.16"
H_RANGE="150.0 200.0 250.0 300.0 400.0 500.0"
S_RANGE="4.0:2.0:10.0"
SEED_RANGE="1:1:10"

# Number of points in the stream-wise direction [-]
NX=1024
# Number of points in the transverse direction [-]
NY=180
# Number of points in the vertical direction [-]
NZ=180
# Distance between points in the transverse direction [m]
DY=1
# Distance between points in the vertical direction [m]
DZ=1
# This is the template file
HTC_TEMPLATE="/home/s244289/DTU_10_MW_Reference_Wind_Turbine_v_9-1/htc/DTU_10MW_RWT_MIMC_TEMPLATE.htc"

# Check to see if there is a folder
if [ ! -e ${DLB_NAME} ]
then
    echo "The study has not been created yet, run 'build_dlb_folder.sh ${DLB_NAME}' first"
    exit
fi

# Function to generate a sequence (float support)
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

# Loop through all parameter combinations
for WS in $WS_RANGE; do
    for TI in $(generate_sequence $TI_RANGE); do
        for H in $H_RANGE; do
            for W in $(generate_w_sequence $H); do
                for S in $(generate_sequence $S_RANGE); do
                    for SEED in $(generate_sequence $SEED_RANGE); do
                        # Generate the TAG name including all parameters
                        TAG="${DLB_NAME}_ws_${WS}_ti_${TI}_h_${H}_w_${W}_s_${S}_seed_${SEED}"
                        # Calculate the DX
                        DX=$(echo "${WS}*${TIME_STOP}/${NX}" | bc -l)
                        # Copy over the template file
                        cp $HTC_TEMPLATE ${DLB_NAME}/htc/${TAG}.htc
                        # Replace the placeholders in the template
                        sed -i "s/MIMC_TEMPLATE_TAG_MIMC/${TAG}/g" ${DLB_NAME}/htc/${TAG}.htc
                        sed -i "s/MIMC_TEMPLATE_TIME_STOP_MIMC/${TIME_STOP}/g" ${DLB_NAME}/htc/${TAG}.htc
                        sed -i "s/MIMC_TEMPLATE_WIND_SPEED_MIMC/${WS}/g" ${DLB_NAME}/htc/${TAG}.htc
                        sed -i "s/MIMC_TEMPLATE_TURB_NX_MIMC/${NX}/g" ${DLB_NAME}/htc/${TAG}.htc
                        sed -i "s/MIMC_TEMPLATE_TURB_NY_MIMC/${NY}/g" ${DLB_NAME}/htc/${TAG}.htc
                        sed -i "s/MIMC_TEMPLATE_TURB_NZ_MIMC/${NZ}/g" ${DLB_NAME}/htc/${TAG}.htc
                        sed -i "s/MIMC_TEMPLATE_TURB_DX_MIMC/${DX}/g" ${DLB_NAME}/htc/${TAG}.htc
                        sed -i "s/MIMC_TEMPLATE_TURB_DY_MIMC/${DY}/g" ${DLB_NAME}/htc/${TAG}.htc
                        sed -i "s/MIMC_TEMPLATE_TURB_DZ_MIMC/${DZ}/g" ${DLB_NAME}/htc/${TAG}.htc
                    done
                done
            done
        done
    done
done
