import cv2
import numpy as np
import pickle


'''
Function to add squares to the live stream
'''
def display_squares(img):
	# Image parametes
	x_len, y_len, w_len, h_len, d = 420, 140, 10, 10, 10
	new_img = None
	
	# 10 rows of pixels
	for i in range(10):
		# 5 columns of pixels
		for j in range(5):

			# Precautionary new_img variable
			if np.any(new_img == None):
				new_img = img[y_len:y_len+h_len, x_len:x_len+w_len]
			else:
				new_img = np.vstack((new_img, img[y_len:y_len+h_len, x_len:x_len+w_len]))
			# display individual pixel squares
			cv2.rectangle(img, (x_len,y_len), (x_len+w_len, y_len+h_len), (0,255,0), 2)
			# move to next pixel in the row
			x_len+=w_len+d

		# reset row value	
		x_len = 420
		# move to next column of pixels
		y_len += h_len +d

	# return the image with the squares
	return new_img

'''
Function that reads the live stream and detecs key presses
'''
def main():
	# live_stream parameters
	x, y, w, h = 300, 100, 300, 300
	
	# Flags for key presses
	captureFlag, saveFlag = False, False
	
	# Use webcam at VideoCapture(0)
	live_stream = cv2.VideoCapture(0)
	
	# Use webcam at VideoCapture(1) if webcam at VideoCapture(0) isn't working
	#live_stream = cv2.VideoCapture(1)

	# Loop for real-time video feed
	while True:

		# Read individual frames
		img = live_stream.read()[1]
		img = cv2.flip(img, 1)
		
		# Convert the BGR image to HSV (Hue Saturation Value)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
		# variable to detect key presses
		keypress = cv2.waitKey(1)

		# When 'c' is pressed
		if keypress == ord('c'):
			# Get HSV of the image
			hsvCrop = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			# set capture flag to true
			captureFlag = True
			# Plot histogram using HSV
			histogram = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
			cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
		
		# When 's' is pressed
		elif keypress == ord('s'):
			# set the save flag to true
			saveFlag = True	
			# Stop waiting for more key presses
			break
		
		# if captureFlag is HIGH
		if captureFlag:	
			# Create a back projection
			dst = cv2.calcBackProject([hsv], [0, 1], histogram, [0, 180, 0, 256], 1)
			disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
			cv2.filter2D(dst,-1,disc,dst)
			# Apply blur functions
			blur = cv2.GaussianBlur(dst, (11,11), 0)
			blur = cv2.medianBlur(blur, 15)
			# Get threshold values
			ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			# Merge if value is above threshold
			thresh = cv2.merge((thresh,thresh,thresh))
			res = cv2.bitwise_and(img,thresh)
			# Display the Thresholded image
			cv2.imshow("Threshold", thresh)
		
		# When the saveFlag is LOW
		if not saveFlag:
			# Continue showing the 50 pixel squares
			imgCrop = display_squares(img)
		# Display the live feed with the histogram
		cv2.imshow("Hand Histogram", img)
	
	# Stop using camers
	live_stream.release()
	# Destroy windows created by OpenCV
	cv2.destroyAllWindows()
	
	# Save the histogram as a pickle
	with open("hist", "wb") as f:
		pickle.dump(histogram, f)


if (__name__ == "__main__"):
    main()