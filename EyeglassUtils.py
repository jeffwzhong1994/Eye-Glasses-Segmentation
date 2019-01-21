# *************************************
# Eyeglass Segmentation Toolbox		  *
# 									  *	
# Rong Yuan (rong01.yuan@vipshop.com) *
#									  *
# Dec 04, 2018						  *
# *************************************

import numpy as np
from scipy.signal import savgol_filter
import cv2

from skimage.segmentation import active_contour

try:
    set
except NameError:
    from sets import Set as set
# from sets import Set

# **** USED ONLY FOR SEGMENTATION RELATED TASKS ****
class Segment:
	def __init__(self, _pixels):
		self.pixels = _pixels
		self.area = 0
	def draw(self, canvas, mask = None):
		self.area = 0
		for pixel in self.pixels:
			x = pixel[0]
			y = pixel[1]
			if mask is None or mask[y, x] == 255:
				canvas[y, x] = 255
				self.area += 1

# -----------------------------------
# Find foreground mask from edge map
# -----------------------------------
def find_eyeglass_mask(img):
	if len(img.shape) == 2:
		edgeMap = img.copy()
	else:
		grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		edgeMap = cv2.Canny(grayImg, 100, 200)
	height, width = edgeMap.shape
	mask = np.zeros(shape=(height, width))
	for j in range(width):
		if np.count_nonzero(edgeMap[:, j]) == 0:
			continue
		top = -1
		bot = -1
		for i in range(height):
			if edgeMap[i, j] == 255:
				top = i
				break
		for i in reversed(range(height)):
			if edgeMap[i, j] == 255:
				bot = i
				break
		mask[top: bot+1, j] = 255
	return mask.copy()

# ---------------------------------------------------------
# Find all segments from color-specified segmentation image
# ---------------------------------------------------------
def find_all_segments(segmentation):
	height, width, channels = segmentation.shape
	outputSegments = []
	nSegments = 0
	segmentMap = {}
	pixelMap = {}
	output = np.zeros(shape=(height, width))
	for i in range(height):
		for j in range(width):
			color = (segmentation[i, j, 0], segmentation[i, j, 1], segmentation[i, j, 2])
			if color not in segmentMap:
				segmentMap[color] = nSegments
				nSegments += 1
			segmentId = segmentMap[color]
			if segmentId not in pixelMap:
				pixelMap[segmentId] = []
			pixelMap[segmentId].append([j, i])

	for segmentId, pixels in pixelMap.items():
		s = Segment(pixels)
		outputSegments.append(s)

	return outputSegments, segmentMap, pixelMap

# ---------------------------------------------------------
# Find smooth boundary pixels in order from a binary image
# using Savitzky-Golay Smoothing Filters 
#
# "Chapter 14. Statistical Description of Data"
#
# [Input binary image should mask targeted area white]
# ---------------------------------------------------------
def find_smooth_boundary(mask):
	lowerBorder = []
	upperBorder = []
	for j in range(mask.shape[1]):
		if np.count_nonzero(mask[:, j]) == 0:
			continue
		for i in range(mask.shape[0]):
			if mask[i, j] == 255:
				upperBorder.append([j, i])
				break
	for j in range(mask.shape[1]):
		if np.count_nonzero(mask[:, j]) == 0:
			continue
		for i in reversed(range(mask.shape[0])):
			if mask[i, j] == 255:
				lowerBorder.append([j, i])
				break

	lowerBorder = np.array(lowerBorder)
	upperBorder = np.array(upperBorder)

	window = 21
	order = 3

	lowerY = lowerBorder[:, 1]
	UpperY = upperBorder[:, 1]

	filteredLowerY = savgol_filter(lowerY, window, order)
	filteredUpperY = savgol_filter(UpperY, window, order)

	lowerBorder[:, 1] = filteredLowerY.copy()
	upperBorder[:, 1] = filteredUpperY.copy()

	nVertices = lowerBorder.shape[0] + upperBorder.shape[0]
	contour = np.zeros(shape=(nVertices, 2))
	idx = 0
	for pixel in lowerBorder:
		contour[idx, :] = pixel
		idx += 1
	for pixel in reversed(upperBorder):
		contour[idx, :] = pixel
		idx += 1

	nVertices = lowerBorder.shape[0] + upperBorder.shape[0]
	contour = np.zeros(shape=(nVertices, 2))
	idx = 0
	for pixel in lowerBorder:
		contour[idx, :] = pixel
		idx += 1
	for pixel in reversed(upperBorder):
		contour[idx, :] = pixel
		idx += 1

	return contour.astype(int)

def converge_boundary(rgbImg, contour):
	snake = active_contour(rgbImg, contour, max_px_move=0.1, max_iterations=50)

	return snake.astype(np.int)
	# 1. find center of the contour to start with
	# center = np.sum(contour, axis = 0)
	# center = center / contour.shape[0]

	# canvas = np.zeros(shape=edge.shape)
	# for p in contour:
	# 	canvas[p[1], p[0]] = 255

	# # 2. find converged boundary
	# convergedContour = np.zeros(shape=contour.shape)
	# x0, y0 = center
	# step = 0.5
	# for i in range(contour.shape[0]):
	# 	x, y = contour[i, :]
	# 	rho = (x - x0) * (x - x0) + (y - y0) * (y - y0)
	# 	rho = rho ** 0.5
	# 	costheta = (x - x0) / rho
	# 	sintheta = (y - y0) / rho
	# 	while edge[y, x] == 0:
	# 		rho += step
	# 		x = int(round(costheta * rho + x0))
	# 		y = int(round(sintheta * rho + y0))
	# 		if x < 0 or y < 0 or x >= edge.shape[1] or y >= edge.shape[0]:
	# 			x, y = contour[i, :]
	# 			break
	# 	convergedContour[i, 0] = x
	# 	convergedContour[i, 1] = y

	# return convergedContour.copy()

def similarity_between_areas(area1, area2):
	# 1. align left end
	left1 = -1
	left2 = -1
	height, width = area1.shape

	for j in range(width):
		if np.count_nonzero(area1[:, j]) > 0:
			left1 = j
			break
	for j in range(width):
		if np.count_nonzero(area2[:, j]) > 0:
			left2 = j
			break
	diff = left1 - left2
	shifted2 = np.roll(area2, diff, axis = 1)

	# 2. compute similarity
	intersectionCount = np.count_nonzero(cv2.bitwise_and(area1, shifted2))
	unionCount = np.count_nonzero(cv2.bitwise_or(area1, shifted2))
	similarity = intersectionCount * 1.0 / unionCount

	return similarity

# ---------------------
# Find convexity defect 
# ---------------------
def find_defects(mask):
	# find main contour
	Gx = cv2.Sobel(mask.astype(np.uint8), cv2.CV_64F, 1, 0)
	Gy = cv2.Sobel(mask.astype(np.uint8), cv2.CV_64F, 0, 1)
	G = np.square(Gx) + np.square(Gy)
	G = np.sqrt(G)
	_, edge = cv2.threshold(G, 0, 255, cv2.THRESH_BINARY)
	edge = edge.astype(np.uint8)
	contours = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contours = contours[1]
	maxContour = None
	maxArea = -1
	for idx in range(len(contours)):
		contour = contours[idx]
		area = cv2.contourArea(contour)
		if area > maxArea:
			maxArea = area
			maxContour = contour
	# find max defect
	hull = cv2.convexHull(maxContour, returnPoints=False)
	defects = cv2.convexityDefects(maxContour, hull)
	maxDefectDepth = 0.0
	maxDefect = None
	for defect in defects:
		depth = defect[0][3] / 256
		if depth > maxDefectDepth:
			maxDefectDepth = depth
			maxDefect = defect[0]
			
	# return only large defect
	MIN_DEFECT_THRESHOLD = 10.0
	if maxDefectDepth > MIN_DEFECT_THRESHOLD:
		# return this area as contour
		start = maxDefect[0]
		end = maxDefect[1]
		convexity = maxDefect[2]
		convexityDefect = []
		if start < end:
			for i in range(start, end + 1):
				convexityDefect.append(maxContour[i][0])
		else:
			for i in range(start, len(maxContour)):
				convexityDefect.append(maxContour[i][0])
			for i in range(end + 1):
				convexityDefect.append(maxContour[i][0])
		return np.array(convexityDefect)
	return None
			
# -------------------------------------------------------------------
# Find mask of both lenses
# from given eyeglass image along with its segmentation
# 
# rgbImg
# segmentation: 2D matrix where different segments are in diverse colors	
# leftLens: 2D binary matrix that masks left lens area with 255
# rightLens: similar to leftLens
# -------------------------------------------------------------------
def find_symmetric_lenses(rgbImg, segmentation):
	# 1. find x-symmetry line
	grayImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)
	edgeMap = cv2.Canny(grayImg, 100, 200)
	height, width  = edgeMap.shape
	foregroundMask = find_eyeglass_mask(edgeMap)

	left = -1
	right = -1
	for col in range(width):
		if np.count_nonzero(edgeMap[:, col]) > 0:
			left = col
			break
	for col in reversed(range(width)):
		if np.count_nonzero(edgeMap[:, col]) > 0:
			right = col
			break
	mid = int((left + right) / 2)
	glassWidth = mid - left + 1

	# 2. split the segmentation
	leftSegmentation = segmentation[:, left: left + glassWidth, :].copy()
	rightSegmentation = segmentation[:, mid: mid + glassWidth, :].copy()
	leftForeground = foregroundMask[:, left: left + glassWidth].copy()
	rightForeground = foregroundMask[:, mid: mid + glassWidth].copy()

	# 3. find all segments from both segmentation map
	leftSegments, leftSegmentDict, leftPixelDict = find_all_segments(leftSegmentation)
	rightSegments, rightSegmentDict, rightPixelDict = find_all_segments(rightSegmentation)

	leftSegmentationMap = np.zeros(shape=leftSegmentation.shape[0:2])
	rightSegmentationMap = np.zeros(shape=rightSegmentation.shape[0:2])

	# 4. start on both sides from the largest segment
	maxLeftSegmentArea = -1
	maxLeftIndex = -1
	maxRightSegmentArea = -1
	maxRightIndex = -1
	for i in range(len(leftSegments)):
		leftSegmentationMap = np.zeros(shape=leftSegmentation.shape)
		leftSegments[i].draw(leftSegmentationMap, leftForeground)
		if leftSegments[i].area > maxLeftSegmentArea:
			maxLeftSegmentArea = leftSegments[i].area
			maxLeftIndex = i
	for i in range(len(rightSegments)):
		leftSegmentationMap = np.zeros(shape=rightSegmentation.shape)
		rightSegments[i].draw(rightSegmentationMap, rightForeground)
		if rightSegments[i].area > maxRightSegmentArea:
			maxRightSegmentArea = rightSegments[i].area
			maxRightIndex = i

	leftSegmentationMap = np.zeros(shape=leftSegmentation.shape[0:2])
	rightSegmentationMap = np.zeros(shape=rightSegmentation.shape[0:2])
	leftSegments[maxLeftIndex].draw(leftSegmentationMap)
	rightSegments[maxRightIndex].draw(rightSegmentationMap)

	leftDefect = find_defects(leftSegmentationMap)
	rightDefect = find_defects(rightSegmentationMap)
	leftDefectMap = np.zeros(shape=(height, width))
	rightDefectMap = np.zeros(shape=(height, width))
	cv2.drawContours(leftDefectMap, [leftDefect], 0, 255, cv2.FILLED)
	cv2.drawContours(rightDefectMap, [rightDefect], 0, 255, cv2.FILLED)

	leftDefectSegments = {}
	rightDefectSegments = {}

	for i in range(leftSegmentation.shape[0]):
		for j in range(leftSegmentation.shape[1]):
			color = tuple(leftSegmentation[i, j, :])
			leftSegmentId = leftSegmentDict[color]
			if leftDefectMap[i, j] == 255:
				if leftSegmentId not in leftDefectSegments:
					leftDefectSegments[leftSegmentId] = 0
				leftDefectSegments[leftSegmentId] += 1

	for i in range(rightSegmentation.shape[0]):
		for j in range(rightSegmentation.shape[1]):
			color = tuple(rightSegmentation[i, j, :])
			rightSegmentId = rightSegmentDict[color]
			if rightDefectMap[i, j] == 255:
				if rightSegmentId not in rightDefectSegments:
					rightDefectSegments[rightSegmentId] = 0
				rightDefectSegments[rightSegmentId] += 1

	maxApps = 0
	maxId = -1
	for id, apps in leftDefectSegments.items():
		if apps > maxApps:
			maxApps = apps
			maxId = id
	leftSegments[maxId].draw(leftSegmentationMap)

	maxApps = 0
	maxId = -1
	for id, apps in rightDefectSegments.items():
		if apps > maxApps:
			maxApps = apps
			maxId = id
	rightSegments[maxId].draw(rightSegmentationMap)

	# 5. iteratively accumulate both segments unitl convergence
	rightFlipped = np.fliplr(rightSegmentationMap)
	intersectionMap = cv2.bitwise_and(leftSegmentationMap, rightFlipped)
	unionMap = cv2.bitwise_or(leftSegmentationMap, rightFlipped)
	intersectionCount = np.count_nonzero(intersectionMap)
	unionCount = np.count_nonzero(unionMap)
	similarity = intersectionCount * 1.0 / unionCount
	updated = True
	while updated:
		leftArea = np.count_nonzero(leftSegmentationMap)
		rightArea = np.count_nonzero(rightSegmentationMap)
		updated = False
		if leftArea < rightArea:
			bestComplement = -1
			bestSimilarity = 0.0
			for i in range(len(leftSegments)):
				s = leftSegments[i]
				testAccumulatedMap = leftSegmentationMap.copy()
				s.draw(testAccumulatedMap)
				intersectionMap = cv2.bitwise_and(testAccumulatedMap, rightFlipped)
				unionMap = cv2.bitwise_or(testAccumulatedMap, rightFlipped)
				intersectionCount = np.count_nonzero(intersectionMap)
				unionCount = np.count_nonzero(unionMap)
				newSimilarity = similarity_between_areas(testAccumulatedMap, rightFlipped)
				if newSimilarity > similarity:
					if newSimilarity > bestSimilarity:
						bestSimilarity = newSimilarity
						bestComplement = i
			if bestComplement != -1:
				s = leftSegments[bestComplement]
				s.draw(leftSegmentationMap)
				similarity = bestSimilarity
				updated = True
		else:
			bestComplement = -1
			bestSimilarity = 0.0
			for i in range(len(rightSegments)):
				s = rightSegments[i]
				testAccumulatedMap = rightSegmentationMap.copy()
				s.draw(testAccumulatedMap)
				rightFlipped = np.fliplr(testAccumulatedMap)
				intersectionMap = cv2.bitwise_and(leftSegmentationMap, rightFlipped)
				unionMap = cv2.bitwise_or(leftSegmentationMap, rightFlipped)
				intersectionCount = np.count_nonzero(intersectionMap)
				unionCount = np.count_nonzero(unionMap)
				newSimilarity = similarity_between_areas(leftSegmentationMap, rightFlipped)
				if newSimilarity > similarity:
					if newSimilarity > bestSimilarity:
						bestSimilarity = newSimilarity
						bestComplement = i
			if bestComplement != -1:
				s = rightSegments[bestComplement]
				s.draw(rightSegmentationMap)
				similarity = bestSimilarity
				updated = True


	leftMask = np.zeros(shape=(height, width))
	rightMask = np.zeros(shape=(height, width))

	leftMask[:, left: left + glassWidth] = leftSegmentationMap.copy()
	rightMask[:, mid: mid + glassWidth] = rightSegmentationMap.copy()

	
	leftContour = find_smooth_boundary(leftMask)
	rightContour = find_smooth_boundary(rightMask)

	for p in leftContour:
		x = p[0]
		y = p[1]
		# cv2.circle(rgbImg, (x, y), 3, (0, 0, 255))

	Gx = cv2.Sobel(grayImg, cv2.CV_8U, 1, 0)
	Gy = cv2.Sobel(grayImg, cv2.CV_8U, 0, 1)
	Gx = Gx.astype(np.float64)
	Gy = Gy.astype(np.float64)
	G = np.square(Gx) + np.square(Gy)
	G = np.sqrt(G)

	leftContour = converge_boundary(rgbImg, leftContour.astype(np.int))
	rightContour = converge_boundary(rgbImg, rightContour.astype(np.int))

	lensesMask = np.zeros(shape=(height, width)).astype(np.uint8)

	cv2.drawContours(lensesMask, [leftContour, rightContour], -1, 255, cv2.FILLED)

	# lensesMask = cv2.bitwise_or(leftMask, rightMask)

	return lensesMask.astype(np.uint8)


def find_shadow(rgbImg, segImg):

	# 1. find eyeglass mask
	grayImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)
	edgeMap = cv2.Canny(grayImg, 100, 200)
	height, width  = edgeMap.shape
	foregroundMask = find_eyeglass_mask(edgeMap)

	# 2. find all segments
	segments, segmentMap, pixelMap = find_all_segments(segImg)
	# 3. find all shadow segments
	shadowSegment = Set()
	#    3.1 from left
	for row in range(height):
		if np.count_nonzero(foregroundMask[row, :]) == 0:
			continue
		for col in range(width):
			if foregroundMask[row, col] == 255:
				color = tuple(segImg[row, col, :])
				segmentId = segmentMap[color]
				shadowSegment.add(segmentId)
				break
	#    3.2 from bot
	for col in range(width):
		if np.count_nonzero(foregroundMask[:, col]) == 0:
			continue
		for row in reversed(range(height)):
			if foregroundMask[row, col] == 255:
				color = tuple(segImg[row, col, :])
				segmentId = segmentMap[color]
				shadowSegment.add(segmentId)
				break
	#    3.3 from right
	for row in range(height):
		if np.count_nonzero(foregroundMask[row, :]) == 0:
			continue
		for col in reversed(range(width)):
			if foregroundMask[row, col] == 255:
				color = tuple(segImg[row, col, :])
				segmentId = segmentMap[color]
				shadowSegment.add(segmentId)
				break
	#    3.4 from top
	for col in range(width):
		if np.count_nonzero(foregroundMask[:, col]) == 0:
			continue
		for row in range(height):
			if foregroundMask[row, col] == 255:
				color = tuple(segImg[row, col, :])
				segmentId = segmentMap[color]
				shadowSegment.add(segmentId)
				break

	for segmentId in shadowSegment:
		for pixel in pixelMap[segmentId]:
			rgbImg[pixel[1], pixel[0], :] = [255, 255, 255]


