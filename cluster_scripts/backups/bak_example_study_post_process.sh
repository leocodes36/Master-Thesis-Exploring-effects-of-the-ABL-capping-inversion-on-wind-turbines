#! /bin/bash

##########################################
# I alwasy like to put all the parameters for my study at the very beginning ...
# this way, I can create new studies, make adjustments very simply, 
# all without understanding the rest of the logic each and every time
##########################################

# The name of the dlb
DLB_NAME=example_study_leo
# The length of the simulation
# NOT USED HERE: TIME_STOP=200.0
# The different wind speeds considered in this study
WSP_LIST="8.0 9.0"
# The different turbulence amplitudes
AMP_LIST="0.1 0.2"
# The number of turb-box grid points in the x direction
# NOT USED HERE: NX=256
# This is the template file
# NOT USED HERE: HTC_TEMPLATE="/home/s244289/DTU_10_MW_Reference_Wind_Turbine_v_9-1/htc/DTU_10MW_RWT_MIMC_TEMPLATE.htc"

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
# Finally, I loop over all the wind speeds, and calculate all the statistics for that wind speed
# In this case, I am using my getChannelStatistics executable ... 
# Run getChannelStatistics without arguments to get the help screen
# you might want your own custom post-process script
# or you might want to have additional post-processing actions
##########################################

# loop through the different wind speed parameters
# In this case, all the simulations with the same wind speed are analyzed together, 
# so that's what the statistics will be based upon, the combined set of simulations with a common wind speed
for W in ${WSP_LIST}
do
	# So I create some variables for later writing the statistics file
	# The count
	RES_CNT=0
	# The text that will list out the result files
	RES_LINE=""
	# The file name where the list will be written
	RES_LIST=/home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_WSP_${W}.lst

	# Loop through the different amplitudes
	for A in ${AMP_LIST}
	do
		# Generate the TAG name ... this must be the same in all scripts for it to work
		TAG=${DLB_NAME}_WSP_${W}_AMP_${A}
		# increment the counter
		let RES_CNT=${RES_CNT}+1;
		# Append a new set of results
		RES_LINE="${RES_LINE} /home/s244289/dlbs/${DLB_NAME}/res/${TAG}.sel /home/s244289/dlbs/${DLB_NAME}/res/${TAG}.dat"
	done

	# writes the list file
	echo "${RES_CNT}${RES_LINE}" > ${RES_LIST}

	# Calls getChannelStatistics
	getChannelStatistics ${RES_LIST} 12  17 18 19  26 27 28  29 30 31  32 33 34 > /home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_WSP_${W}.stats
	#                     ^^^^^^^^^  ^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
	#                     list-file  nCh     list-of-channels                     File to store the results
done

