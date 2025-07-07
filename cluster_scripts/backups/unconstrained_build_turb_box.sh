#! /bin/bash

# The name of the dlb
DLB_NAME=unconstrained_study
# The length of the simulation
TIME_STOP=600.0

# Define parameter ranges as: start:step:stop
WS_RANGE="5.0:1.0:25.0"
TI_RANGE="0.16:0:0.16"
SEED_RANGE="1:1:10"

# Time step in the simulation [s]
TIME_STEP=1.0
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
# High frequency compensation
HF=1

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

SEED_LIST=$(generate_sequence $SEED_RANGE)

# Loop through all parameter combinations
for WS in $(generate_sequence $WS_RANGE)
do
    for TI in $(generate_sequence $TI_RANGE)
    do
        # Nominal Wind Speed [m/s]
        NOMINAL_WIND_SPEED=$WS
        # Reference Turbulence Intensity [-]
        TURBULENCE_INTENSITY=$TI
        # Calculate DX
        DX=$(echo "${WS}*${TIME_STOP}/${NX}" | bc -l)
        # The base name that is given to the output files
        TAG="${DLB_NAME}_ws_${WS}_ti_${TI}_seed_"
        # Call python script to generate the turbulence files
        python \
            /home/s244289/py/dlb_generator/generate_box_unconstrained.py \
            "$NOMINAL_WIND_SPEED" \
            "$TURBULENCE_INTENSITY" \
            "$TIME_STEP" \
            "$NX" \
            "$NY" \
            "$NZ" \
            "$DX" \
            "$DY" \
            "$DZ" \
            "$HF" \
            "$TAG" \
            "$SEED_LIST"
        # Remove the stored spectral tensor
        rm mannsqrtphi_*
    done
done

# Rename .turb files to .bin
for file in tmp/*.turb; do
    mv "$file" "${file%.turb}.bin"
done

mv tmp/* /home/s244289/dlbs/${DLB_NAME}/turb/

echo "Done, enjoy turbulence!"