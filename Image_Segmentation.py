#Importing other python files 
import PPM_converter as converter
import PPM_Segmentation as segmentation

#Importing regular packages
import os
import os.path
import sys
import argparse

#Get Constant
WEB = "https://www.online-utility.org/image/convert/to/PPM"
DL_DIRECTORY = "/Users/jeffwzhong/Downloads/"
SEG_DIR = "/Users/jeffwzhong/Desktop/VIPshop/data/Python/Converted_PPM/"


# MAIN FUNCTION
# INPUT
def image_segmentation(image_dir):

	#Call Convert JPG to PPM function:
	converter.convert_jpg_to_ppm_in_batch(image_dir)
	segmentation.ppm_segmented_to_jpg_in_batch(SEG_DIR)

#Definition of Parser:
def create_arg_parser():

	parser =argparse.ArgumentParser(description = 'Input your the directory.')
	parser.add_argument('inputDirectory', help = 'Path to the input directory',type = str)
	parser.add_argument('OutputDirectory', help = 'Path to the output that contains the images')
	return parser

#Main Function:
if __name__ == "__main__":

	arg_parser = create_arg_parser()
	parsed_args = arg_parser.parse_args(sys.argv[1:])
	print(parsed_args.inputDirectory)
	image_segmentation(parsed_args.inputDirectory)
