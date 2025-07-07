#! /bin/bash

##########################################
# I alwasy like to put all the parameters for my study at the very beginning ...
# this way, I can create new studies, make adjustments very simply, 
# all without understanding the rest of the logic each and every time
##########################################

# The name of the dlb
DLB_NAME=example_study_leo
# The length of the simulation
TIME_STOP=200.0
# The different wind speeds considered in this study
WSP_LIST="8.0 9.0"
# The different turbulence amplitudes
AMP_LIST="0.1 0.2"
# The number of turb-box grid points in the x direction
NX=256
# This is the template file
HTC_TEMPLATE="/home/s244289/DTU_10_MW_Reference_Wind_Turbine_v_9-1/htc/DTU_10MW_RWT_MIMC_TEMPLATE.htc"

##########################################
# Then it's good practice to have a couple checks to see that everything is in place to run the script,
# that the arguments are correct, whatever, just always try to catch mistakes, 
# here it's super basic, but here as an example
##########################################

# Check to see if there is a folder
if [ ! -e ${DLB_NAME} ]
then
	echo "The study has not been created yet, run 'build_dlb_folder.sh ${DLB_NAME}' first"
	exit
fi

##########################################
# Finally, I generate the HTC files by looping over the parameters of my study and using sed
# There are better ways, for example, I have shared a set of python scripts that can do the same time
# It's really up to you ...
##########################################

# loop through the wind speed parameters
for W in ${WSP_LIST}
do
	# loop through the amplitude parameters
	for A in ${AMP_LIST}
	do
		# Generate the TAG name ... this must be the same in all scripts for it to work
		TAG=${DLB_NAME}_WSP_${W}_AMP_${A}
		# Calculate the DX
		# DX wind_speed*time_stop/NX
		DX=$(echo "${W}*${TIME_STOP}/${NX}" | bc -l)
		# Copy over the template files
		cp $HTC_TEMPLATE ${DLB_NAME}/htc/${TAG}.htc
		# Substitute all the template tags with the real values with the 'sed' command
		sed "s/MIMC_TEMPLATE_TAG_MIMC/${TAG}/g" ${DLB_NAME}/htc/${TAG}.htc -i
		sed "s/MIMC_TEMPLATE_TIME_STOP_MIMC/${TIME_STOP}/g" ${DLB_NAME}/htc/${TAG}.htc -i
		sed "s/MIMC_TEMPLATE_WIND_SPEED_MIMC/${W}/g" ${DLB_NAME}/htc/${TAG}.htc -i
		sed "s/MIMC_TEMPLATE_TURB_NX_MIMC/${NX}/g" ${DLB_NAME}/htc/${TAG}.htc -i
		sed "s/MIMC_TEMPLATE_TURB_NY_MIMC/32/g" ${DLB_NAME}/htc/${TAG}.htc -i
		sed "s/MIMC_TEMPLATE_TURB_NZ_MIMC/32/g" ${DLB_NAME}/htc/${TAG}.htc -i
		sed "s/MIMC_TEMPLATE_TURB_DX_MIMC/${DX}/g" ${DLB_NAME}/htc/${TAG}.htc -i
		sed "s/MIMC_TEMPLATE_TURB_DY_MIMC/7.5/g" ${DLB_NAME}/htc/${TAG}.htc -i
		sed "s/MIMC_TEMPLATE_TURB_DZ_MIMC/7.5/g" ${DLB_NAME}/htc/${TAG}.htc -i
	done
done

