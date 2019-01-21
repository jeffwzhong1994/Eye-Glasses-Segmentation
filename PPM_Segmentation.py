#Import packages
import os
import os.path
import time
import sys
import argparse

#Constant
INPUT_DIR =  "/Users/jeffwzhong/Desktop/VIPshop/data/Python/Converted_PPM/"
OUTPUT_DIR = "/Users/jeffwzhong/Desktop/VIPshop/data/Python/PPM_Segmented/"
EXE_SEGMENT_DIR = "/Users/jeffwzhong/Desktop/VIPshop/data/Python/segment/segment"
# Put the constant into command line
SIGMA = input('Sigma: ')
K = input('K Value for threshold: ')
MIN_COM = input('Minimum components:')


# MAIN FUNCTION
#Output the file to the correct Directory:
def output_segmented_images_to_directory(inputimg):

	#Change the output images to the correct path:
	outputimg = inputimg.replace(".PPM",".JPG")

	#Return output:
	return outputimg

# Input the image and change the output to correct path
def ppm_segmented_to_jpg_in_batch(image_dir):

	#get all the PPM images in the folder:
	for inputimg in os.listdir(image_dir):

		#Filter out all the None PPM image:
		if not inputimg.endswith(".PPM"):
			continue

		#Return output to correct functions:
		outputimg = output_segmented_images_to_directory(inputimg)

		#string append all the command line:
		command = EXE_SEGMENT_DIR + " " + str(SIGMA) + " " + str(K) + " " + str(MIN_COM) + " " + INPUT_DIR+ str(inputimg) + " " + OUTPUT_DIR + str(outputimg)

		print(command)
		#execute the program in Python
		os.system(command)


