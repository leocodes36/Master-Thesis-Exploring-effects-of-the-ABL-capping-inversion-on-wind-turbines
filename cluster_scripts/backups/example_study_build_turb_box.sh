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

##########################################
# Then it's good practice to have a couple checks to see that everything is in place to run the script,
# that the arguments are correct, whatever, just always try to catch mistakes, 
# here it's super basic, but here as an example
##########################################

if [ ! -e ${DLB_NAME} ]
then
	echo "The study has not been created yet, run 'build_dlb_folder.sh ${DLB_NAME}' first"
	exit
fi

##########################################
# Finally, I generate the turbulent boxes by looping over the parameters of the study
# To use hipersim, you would likely be calling a python script with arguments
# or create a python script that does exactly what this script does
# However, it is easier to parallelize a bash script ... so this is why I show this
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
		# In this example, I create my turb files with generateAnalyticTurbFile
		# Just run generateAnalyticTurbFile to get the help on how generateAnalyticTurbFile works
		generateAnalyticTurbFile ${DLB_NAME}/turb/${TAG}_u.bin ${NX} 32 32 ${DX} 7.5 7.5 ${W} "${A}*sin(0.5*t)"
		generateAnalyticTurbFile ${DLB_NAME}/turb/${TAG}_v.bin ${NX} 32 32 ${DX} 7.5 7.5 ${W} "0.0"
		generateAnalyticTurbFile ${DLB_NAME}/turb/${TAG}_w.bin ${NX} 32 32 ${DX} 7.5 7.5 ${W} "0.0"
	done
done

