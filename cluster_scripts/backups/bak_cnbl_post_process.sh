#! /bin/bash

# The name of the dlb
DLB_NAME=cnbl_example_study
# The different wind speeds considered in this study
WS_LIST="9.0 10.0"
# The different turbulence intensities
TI_LIST="0.16"
# The different seeds to be used in this study
SEED_LIST="1 2 3"
# Relevant HAWC2 sensors
SENS_LIST="17 26 29 32"

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

# #Loop through the different wind speed parameters
# In this case, all the simulations with the same wind speed are analyzed together, 
# so that's what the statistics will be based upon, the combined set of simulations with a common wind speed
for WS in ${WS_LIST}
do
	# So I create some variables for later writing the statistics file
	# The count
	RES_CNT=0
	# The text that will list out the result files
	RES_LINE=""
	# The file name where the list will be written
	RES_LIST=/home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_ws_${WS}.lst

	# Loop through the different amplitudes
	for TI in ${TI_LIST}
	do
		for SEED in ${SEED_LIST}
		do
			# Generate the TAG name ... this must be the same in all scripts for it to work
			TAG="${DLB_NAME}_ws_${WS}_ti_${TI}_seed_${SEED}"
			# increment the counter
			let RES_CNT=${RES_CNT}+1;
			# Append a new set of results
			RES_LINE="${RES_LINE} /home/s244289/dlbs/${DLB_NAME}/res/${TAG}.sel /home/s244289/dlbs/${DLB_NAME}/res/${TAG}.dat"
			HIST_LINE="/home/s244289/dlbs/${DLB_NAME}/res/${TAG}.sel /home/s244289/dlbs/${DLB_NAME}/res/${TAG}.dat"			
			for SENS in ${SENS_LIST}
			do
				echo "${TAG}_channel_${SENS}" >> /home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_ws_${WS}.hist
				printTimeSeries ${HIST_LINE} ${SENS} | freqStats hist_data >> /home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_ws_${WS}.hist
			done
		done
	done

	# writes the list file
	echo "${RES_CNT}${RES_LINE}" > ${RES_LIST}

	# Calls getChannelStatistics
	getChannelStatistics ${RES_LIST} 4  ${SENS_LIST} > /home/s244289/dlbs/${DLB_NAME}/post/${DLB_NAME}_ws_${WS}.stats
	#                     ^^^^^^^^^  ^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
	#                     list-file  nCh     list-of-channels                     File to store the results
done

