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
# This is the template file
HTC_TEMPLATE="/home/s244289/DTU_10_MW_Reference_Wind_Turbine_v_9-1/htc/DTU_10MW_RWT_MIMC_TEMPLATE.htc"

# Check to see if there is a folder
if [ ! -e ${DLB_NAME} ]
then
	echo "The study has not been created yet, run 'build_dlb_folder.sh ${DLB_NAME}' first"
	exit
fi

# Loop through the wind speed parameters
for WS in ${WS_LIST}
do
	# Loop through the turbulence intensity parameters
	for TI in ${TI_LIST}
	do
		# Loop through the seeds
		for SEED in ${SEED_LIST}
		do
			# Generate the TAG name ... this must be the same in all scripts for it to work
			TAG="${DLB_NAME}_ws_${WS}_ti_${TI}_seed_${SEED}"
			# Calculate the DX
			# DX wind_speed*time_stop/NX
			DX=$(echo "${WS}*${TIME_STOP}/${NX}" | bc -l)
			# Copy over the template files
			cp $HTC_TEMPLATE ${DLB_NAME}/htc/${TAG}.htc
			# Substitute all the template tags with the real values with the 'sed' command
			sed "s/MIMC_TEMPLATE_TAG_MIMC/${TAG}/g" ${DLB_NAME}/htc/${TAG}.htc -i
			sed "s/MIMC_TEMPLATE_TIME_STOP_MIMC/${TIME_STOP}/g" ${DLB_NAME}/htc/${TAG}.htc -i
			sed "s/MIMC_TEMPLATE_WIND_SPEED_MIMC/${WS}/g" ${DLB_NAME}/htc/${TAG}.htc -i
			sed "s/MIMC_TEMPLATE_TURB_NX_MIMC/${NX}/g" ${DLB_NAME}/htc/${TAG}.htc -i
			sed "s/MIMC_TEMPLATE_TURB_NY_MIMC/${NY}/g" ${DLB_NAME}/htc/${TAG}.htc -i
			sed "s/MIMC_TEMPLATE_TURB_NZ_MIMC/${NZ}/g" ${DLB_NAME}/htc/${TAG}.htc -i
			sed "s/MIMC_TEMPLATE_TURB_DX_MIMC/${DX}/g" ${DLB_NAME}/htc/${TAG}.htc -i
			sed "s/MIMC_TEMPLATE_TURB_DY_MIMC/${DY}/g" ${DLB_NAME}/htc/${TAG}.htc -i
			sed "s/MIMC_TEMPLATE_TURB_DZ_MIMC/${DZ}/g" ${DLB_NAME}/htc/${TAG}.htc -i
		done
	done
done