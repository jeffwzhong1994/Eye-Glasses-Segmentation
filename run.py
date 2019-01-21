# *************************************
# Eyeglass Segmentation Toolbox       *
#                                     * 
# Rong Yuan (rong01.yuan@vipshop.com) *
#                                     *
# Dec 04, 2018                        *
# *************************************
import numpy as np
import cv2

import EyeglassUtils

rgbImgPath = "./data/image/01.jpg"
segImgPath = "./data/segmentation/01.jpg"

rgbImg = cv2.imread(rgbImgPath)
segImg = cv2.imread(segImgPath)
height, width, channels = rgbImg.shape

hsvImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2HSV)


# lenses mask is a binary image with lenses area set white
lensesMask = EyeglassUtils.find_symmetric_lenses(rgbImg, segImg)

# eyeglass mask is a binary image with whole eyeglass area set white
eyeglassMask = EyeglassUtils.find_eyeglass_mask(rgbImg)

# set background and lenses area to transparent
rgbImgLensesBgdRemoved = np.zeros(shape=(height, width, 3))

# build alpha channel
alpha = eyeglassMask.astype(np.uint8)
notLensesMask = cv2.bitwise_not(lensesMask).astype(np.uint8)
alpha = cv2.bitwise_and(alpha, notLensesMask)

# create rgba img ready for output
rgbaImg = np.zeros(shape=(height, width, 4))
rgbaImg[:, :, 0:3] = rgbImg.copy()
rgbaImg[:, :, 3] = alpha.astype(np.uint8)

# write rgba image out as a png file [jpg doesn't allow transparency]
cv2.imwrite("outputimg.png", rgbaImg)
cv2.waitKey(0)

