
#import packages
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os.path
import time
import sys
import argparse

# Get Constant 
WEB = "https://www.online-utility.org/image/convert/to/PPM"
GET_DIRECTORY = "/Users/jeffwzhong/Desktop/VIPshop/data/Python/Dapei_glasses/"
DL_DIRECTORY = "/Users/jeffwzhong/Downloads/"
OUTPUT_DIR = "/Users/jeffwzhong/Desktop/VIPshop/data/Python/Converted_PPM/"

# MAIN FUNCTION
# image_dir: (string) under which those images lie
def convert_jpg_to_ppm_in_batch(image_dir):
	# get all the images in the folder
	for image in os.listdir(image_dir):

		print(image_dir)
		#Correct/Incorrect Output path
		output =  OUTPUT_DIR +image.replace(".jpg",".PPM")
		wrongoutput = DL_DIRECTORY + image.replace(".jpg",".PPM")	

		# Exceptions: [When do I not need to download this file?]
		# 1. if this is not an image
		# 2. if corresponding PPM file exists under designated directory
		if not image.endswith(".jpg"):
			continue

		#Get directory:
		dir = image_dir + "/" + str(image)
		print(dir)
		# If the correct path does not have the files:
		if not os.path.exists(output):
			#If the wrong path has file:
			if os.path.exists(wrongoutput):
				#Put it in the correct path:
				os.rename(wrongoutput, output)

			else:	
				#If the output is not downloaded yet:
				# Import Webdriver:
				driver = webdriver.Safari()
				driver.get(WEB)
				
				# Find Element of choosing file	
				elem = driver.find_element_by_name("fname")
				elem = elem.send_keys(dir)

				#convert image and download
				driver.find_element_by_xpath("//input[@value='Convert and Download']").click()

				#Let the images fully downloaded
				while not os.path.exists(wrongoutput):
					time.sleep(1)

				#Jump out of the while loop:
				os.rename(wrongoutput, output)

				#Else, Quit;
				driver.quit()	
	


