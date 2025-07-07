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

# Loop through windspeed list
for WS in ${WS_LIST}
do
    # Loop through turbulence intensity list
    for TI in ${TI_LIST}
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
            /Users/leo/python/thesis/python_utils/generate_box_unconstrained.py \
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
    done
done