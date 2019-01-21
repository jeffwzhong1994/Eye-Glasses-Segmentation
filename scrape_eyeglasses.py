import urllib
import urllib.error
import re
import os
from urllib.request import urlopen
from urllib.parse import quote
from django.utils.encoding import smart_str
from bs4 import BeautifulSoup, SoupStrainer


# PARAMETERS
KEYWORD= "眼镜"
KEYWORD = urllib.parse.quote(KEYWORD)
URL_PREFIX = "https://category.vip.com/suggest.php?keyword=" + KEYWORD
MAX_PAGE = 55
OUTPUT_DIR = "./image2/"

# UTILITY FUNCTIONS
def get_url_suffix(page):
	suffix = "&page="+ str(page)+ "&count=100&suggestType=brand#catPerPos"
	return suffix

# MAJOIR FUNCTIONALITIES

# extract html content form url
def getHtml(url):

	try:
		page = urlopen(url)
		html = page.read()

	except (urllib.error.HTTPError,http.client.IncompleteRead) as e:
		pass

	return html

# extract images from given html content
def getImgs(html):
	reg = r'small_image":"(.+?)"'
	reg1 = r'product_id":"(.+?)"'
	reg2 = r'brand_id":"(.+?)"'
	imgre = re.compile(reg)
	productre = re.compile(reg1)
	brandre = re.compile(reg2)
	# imgList = imgre.findall(str(html))
	productList = productre.findall(str(html))
	brandList = brandre.findall(str(html))
	path = 'image2'
	if not os.path.isdir(path):
		os.makedirs(path)

	#Parameters	
	count = 0	
	i = 0 

	#1.get Image URL
	for count in range(0,len(productList)):

		detailurl = "https://detail.vip.com/detail-" + brandList[count] + "-" + productList[count] + ".html"

		# 2.get HTML source
		try:
			detailpage = urlopen(detailurl)
			detailhtml = detailpage.read()
		except urllib.error.HTTPError as e:
			pass

		# 3.get image from HTML source using re

		reg3 = r'(//a.vpimg2.com/upload/merchandise/pdcvis/.+?)"' or r'(//a.vpimg3.com/upload/merchandise/pdcvis/.+?)"'
		try:
			detailImagere = re.compile(reg3)
			detailImageList = detailImagere.findall(str(detailhtml))
		except Exception as e:
			pass	
	
		# 4. replace the small image with big image and fix some bugs, output the images to directory
		for i in range(0,len(detailImageList)):
			# lastUnderscore = detailImageList[i].find("_")
			# detailImageList[i] = detailImageList[i][0:lastUnderscore] + ".jpg"
			# detailImageList[i] = detailImageList[i].replace("//", "")
			detailImageList[i] = detailImageList[i] .replace("_420x420_90", "")
			detailImageList[i] = detailImageList[i] .replace("_64x64_90", "")
			detailImageList[i] = detailImageList[i] .replace("_54x69_100", "")
			detailImageList[i] = detailImageList[i] .replace("jp.jpg", "jpg")
			lastSlash = detailImageList[i].rfind("/")
			imgName1 = detailImageList[i][lastSlash+1:]
			detailImageList[i] = "http:" + detailImageList[i]
			urllib.request.urlretrieve(detailImageList[i], OUTPUT_DIR + imgName1)
			i+=1

		# return detailImageList
			
	count += 1

# REAL STUFF STARTS HERE

# 1. iterate through pages
for page in range(1, MAX_PAGE+1):
	print ("On page " + str(page) + "/" + str(MAX_PAGE))
	# 2. construct url
	url = URL_PREFIX + get_url_suffix(page)
	print ("Url: " + url)
	html = getHtml(url)
	# "驼峰命名法"
	# 3. download images on current page
	ImageList = getImgs(html)
