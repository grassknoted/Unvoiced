import cv2
import numpy as np
import pickle
from pandas import DataFrame
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import load_model
from keras import losses
from gtts import gTTS
import os 

# CHANGE THE FILE NAME TO DIFFERENT HAAR CASCADES TO FIND THE BEST ONE
haar_cascade = 'hand_detection_cascade.xml'
image_no = 0

def speak(ip_text):
	tts = gTTS(text=ip_text, lang='en')
	tts.save("pcvoice.mp3")
	os.system("mpg321 pcvoice.mp3")

def convert_label(x):
	if x == '0':
		return 'A'
	elif x == '1':
		return 'B'
	elif x == '2':
		return 'C'
	elif x == '3':
		return 'D'
	elif x == '4':
		return 'E'
	elif x == '5':
		return 'F'
	elif x == '6':
		return 'G'
	elif x == '7':
		return 'H'
	elif x == '8':
		return 'I'
	elif x == '9':
		return 'J'
	elif x == '10':
		return 'K'
	elif x == '11':
		return 'L'
	elif x == '12':
		return 'M'
	elif x == '13':
		return 'N'
	elif x == '14':
		return 'O'
	elif x == '15':
		return 'P'
	elif x == '16':
		return 'Q'
	elif x == '17':
		return 'R'
	elif x == '18':
		return 'S'
	elif x == '19':
		return 'T'
	elif x == '20':
		return 'U'
	elif x == '21':
		return 'V'
	elif x == '22':
		return 'W'
	elif x == '23':
		return 'X'
	elif x == '24':
		return 'Y'
	elif x == '25':
		return 'Z'

def main():

	train_data = pd.read_csv('/media/akash/This is Storage/Sem VI/ML Lab Project/sign_mnist_train.csv', sep=',', header = None, low_memory=False)
	#print(train_data.head(n=5))
	test_data = pd.read_csv('/media/akash/This is Storage/Sem VI/ML Lab Project/sign_mnist_test.csv', sep=',', header = None, low_memory=False)

	train_data = train_data[1:]
	test_data = test_data[1:]

	observed_train_values = train_data
	observed_train_values = observed_train_values.drop(observed_train_values.columns[1:], axis=1)
	only_train_pixels = train_data
	only_train_pixels = only_train_pixels.drop(only_train_pixels.columns[0], axis=1)

	observed_train_values = list(observed_train_values.values.flatten())
	only_train_pixels = only_train_pixels.values.tolist()

	observed_test_values = test_data
	observed_test_values = observed_test_values.drop(observed_test_values.columns[1:], axis=1)
	only_test_pixels = test_data
	only_test_pixels = only_test_pixels.drop(only_test_pixels.columns[0], axis=1)

	observed_test_values = list(observed_test_values.values.flatten())
	only_test_pixels = only_test_pixels.values.tolist()

	train_data['real_label'] = train_data.apply(lambda row: convert_label(row[0]), axis=1)
	test_data['real_label'] = test_data.apply(lambda row: convert_label(row[0]), axis=1)
	#print(observed_train_values)
	print("L:",len(only_test_pixels),"IL:", len(only_test_pixels[2]), "Vals:", len(observed_test_values), len(observed_test_values[0]) )


	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(25, activation='softmax')) 

	model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])

	model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])

	#print(only_train_pixels)
	print("Train shape: ", len(only_train_pixels), "x", len(only_train_pixels[0]))
	print("Train labels shape: ", len(observed_train_values), "x", len(observed_train_values[0]))

	train_pixels = []
	for image in only_train_pixels:
		image = np.reshape(image, (28, 28))
		train_pixels.append(image)

	print("Training Shape: ", train_pixels[0][0].shape[1:])

	#model.fit(np.array(only_train_pixels), np.asarray(observed_train_values), epochs=1)

	model = load_model('cnn_model.h5', custom_objects={'loss_categorical_crossentropy': losses.categorical_crossentropy})

	#print(only_train_pixels[0])
	#print(only_train_pixels)
	#print(tr_data.head())
	'''
	print(only_train_pixels.head())
	for index, row in only_train_pixels.iterrows():
		for value in range(785):
			only_train_pixels.at[index, value] = int(only_train_pixels.at[index, value])/255

	print("AFTER\n")
	print(only_train_pixels.head())
	'''
	x, y, w, h = 300, 100, 300, 300
    # Flags for key presses
	captureFlag, saveFlag, escapeFlag = False, False, False
    
    # Use webcam at VideoCapture(0)
	live_stream = cv2.VideoCapture(0)

	global image_no
    
    # Use webcam at VideoCapture(1) if webcam at VideoCapture(0) isn't working
    #live_stream = cv2.VideoCapture(1)

    # Loop for real-time video feed
	while True:

        # Detect keypresses
		keypress = cv2.waitKey(1)

		# Read individual frames
		img = live_stream.read()[1]

		# Laterally invert the frame
		img = cv2.flip(img, 1)

		# Conver to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# HSV Filtered Image
		hsvCrop = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# Getting and normalizing the Histogram
		histogram = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
		cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)

		# What image to scan?
		img_to_use = img

		# The Haar cascade used to detect a hand in the frame
		hand_cascade = cv2.CascadeClassifier(haar_cascade)       
		hands = hand_cascade.detectMultiScale(img_to_use, 1.3, 5)

		for (x,y,w,h) in hands:
			cv2.rectangle(img_to_use, (x,y), (x+w,y+h), (0,0,255), 2)

		# If 'c' is pressed
		if keypress == ord('c'):
			captureFlag = True

		if captureFlag:
			captureFlag = False
			resized_image = cv2.resize(gray, (28, 28))
			print(model.summary())
			print("Resized Image:",type(resized_image), len(resized_image))
			print(type(resized_image[0]), len(resized_image[0]))
			print(type(resized_image[0][0]))
			resized_image = np.reshape(resized_image,[1,28,28,1])
			cv2.imwrite( "./saved/image"+str(image_no)+".jpg", resized_image );
			prediction = model.predict(resized_image)
			
			for vec in range(25):
				if(prediction[0][vec] == 1):
					predicted_letter = convert_label(vec)
					print(predicted_letter)
					#speak(predicted_letter)
			

			#speak('c')
			image_no = image_no + 1
			'''
            # Create a back projection
            dst = cv2.calcBackProject([hsvCrop], [0, 1], histogram, [0, 180, 0, 256], 1)
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
			
			'''

        # If ESC is pressed
		if keypress == 27:
			exit(0)

		cv2.imshow("Hand Histogram", img_to_use)

    
    # Stop using camers
	live_stream.release()

    # Destroy windows created by OpenCV
	cv2.destroyAllWindows()
    
    # Save the histogram as a pickle
	with open("hist", "wb") as f:
		pickle.dump(histogram, f)


if (__name__ == "__main__"):
	main()