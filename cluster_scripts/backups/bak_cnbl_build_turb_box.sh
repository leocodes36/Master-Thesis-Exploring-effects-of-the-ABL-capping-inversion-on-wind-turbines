#! /bin/bash

# The name of the dlb
DLB_NAME=cnbl_example_study
# The length of the simulation
TIME_STOP=200.0
# The different wind speeds considered in this study
WS_LIST="9.0 10.0"
# The different turbulence intensities
TI_LIST="0.16"
# The different seeds to be used in this study
SEED_LIST="1 2 3"
# Time step in the simulation [s]
TIME_STEP=1.0
# Number of points in the stream-wise direction [-]
NX=200
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
            /home/s244289/py/dlb_generator/generate_box_cnbl.py \
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

# Rename .turb files to .bin
for file in tmp/*.turb; do
    mv "$file" "${file%.turb}.bin"
done

mv tmp/* /home/s244289/dlbs/${DLB_NAME}/turb/

echo "Done, enjoy turbulence!"