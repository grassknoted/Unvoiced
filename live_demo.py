import cv2				# Import OpenCV for image processing
import sys				# Import for time
import os				# Import for reading files
import threading		# Import for separate thread for image classification
import numpy as np 		# Import for converting vectors
from gtts import gTTS   # Import Google Text to Speech

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf # Import tensorflow for Inception Net's backend

# Language in which you want to convert
language = 'en'

# Get a live stream from the webcam
live_stream = cv2.VideoCapture(0)

# Load training labels file
label_lines = [line.rstrip() for line in tf.gfile.GFile("training_set_labels.txt")]

# Load trained model's graph
with tf.gfile.FastGFile("trained_model_graph.pb", 'rb') as f:
	# Define a tensorflow graph
    graph_def = tf.GraphDef()

    # Read and import line by line from the trained model's graph
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def predict(image_data):

	# Focus on Region of Interest (Image within the bounding box)
	resized_image = image_data[70:350, 70:350]
	
	# Resize to 200 x 200
	resized_image = cv2.resize(resized_image, (200, 200))
	
	image_data = cv2.imencode('.jpg', resized_image)[1].tostring()

	predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

	# Sort to show labels of first prediction in order of confidence
	top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

	max_score = 0.0
	res = ''
	for node_id in top_k:
		human_string = label_lines[node_id]
		score = predictions[0][node_id]
		if score > max_score:	
			max_score = score
			res = human_string
	return res, max_score

# STILL WORKING ON THIS
# Function to classify images in real time
def classify_image(image_array):

	# New tensorflow session
	with tf.Session() as sess:

		# Convert the image array into a tensorflow readable format
		image_array = cv2.resize(image_array,dsize=(299,299), interpolation = cv2.INTER_CUBIC)

		# Convert the image array into a numpy array
		np_image_data = np.asarray(image_array)

		# Insert an extra dimension to work for the current model's shape
		np_final = np.expand_dims(np_image_data, axis=0)

		# Feed the image data to the graph of the trained model and get first prediction
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

		# DEBUG THIS
		# Initial version : predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
		predictions = sess.run(softmax_tensor,{'Mul:0': np_final})

		# Sort to show labels of first prediction in order of confidence
		sorted_predictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

		print("\n\nPredicted Letter: ", str(label_lines[sorted_predictions[0]]).upper(), "\tScore: ", predictions[0][sorted_predictions[0]], "\n\n")

		speak_letter(str(label_lines[sorted_predictions[0]]).upper())
		# Display the letters and the score for each prediction
		'''for letter_prediction in sorted_predictions:
			letter = label_lines[letter_prediction]
			score = predictions[0][letter_prediction]
			print('%s (score = %.5f)' % (letter, score))'''

def speak_letter(letter):
	# Create the text to be spoken
    prediction_text = letter
    
    # Create a speech object from text to be spoken
    speech_object = gTTS(text=prediction_text, lang=language, slow=False)

    # Save the speech object in a file called 'prediction.mp3'
    speech_object.save("prediction.mp3")
 
    # Playing the speech using mpg321
    os.system("mpg321 prediction.mp3")

# TEST FUNCTION
def printit():
	img = live_stream.read()[1]
	print("Hello World", len(img))

with tf.Session() as sess:
	# Feed the image_data as input to the graph and get first prediction
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

	# Global variable to keep track of time
	time_counter = 0

	# Flag to check if 'c' is pressed
	captureFlag = False
	
	# Infinite loop
	while True:

		# Display live feed until ESC key is pressed
		# Press ESC to exit
		keypress = cv2.waitKey(1)

		# Flip the image laterally
		#img = cv2.flip(img, 1)
		
		# TESTING:
		#threading.Timer(5.0, printit).start()
		
		# Read a single frame from the live feed
		img = live_stream.read()[1]

		# Set a region of interest
		cv2.rectangle(img, (70, 70), (350, 350), (0,255,0), 2)

		# Show the live stream
		cv2.imshow("Live Stream", img)
		
		# To get time intervals
		#if time_counter % 20 == 0:

			#classify_image(img)

		# 'C' is pressed
		if keypress == ord('c'):
			captureFlag = True

		if captureFlag:
			captureFlag = False

			# Show the image considered for classification
			# Just for Debugging
			cv2.imshow("Resized Image", resized_image)

			# Get the letter and the score
			letter, score = predict(image_data)
			print("Letter: ",letter.upper(), " Score: ", score)

			# Say the letter out loud
			speak_letter(letter)

		# If ESC is pressed
		if keypress == 27:
			exit(0)	

		# Update time
		time_counter = time_counter + 1

# Stop using camers
live_stream.release()

# Destroy windows created by OpenCV
cv2.destroyAllWindows()