import cv2				# Import OpenCV for image processing
import sys				# Import for time
import os				# Import for reading files
import threading		# Import for separate thread for image classification
import numpy as np 		# Import for converting vectors

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf # Import tensorflow for Inception Net's backend

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

		# Display the letters and the score for each prediction
		for letter_prediction in sorted_predictions:
			letter = label_lines[letter_prediction]
			score = predictions[0][letter_prediction]
			print('%s (score = %.5f)' % (letter, score))

# TEST FUNCTION
def printit():
	img = live_stream.read()[1]
	print("Hello World", len(img))

# Global variable to keep track of time
time_counter = 0

# Infinite loop
while True:

	# Display live feed until ESC key is pressed
	# Press ESC to exit
	keypress = cv2.waitKey(1)
	
	# TESTING:
	#threading.Timer(5.0, printit).start()
	
	# Read a single frame from the live feed
	img = live_stream.read()[1]
	
	# To get time intervals
	if time_counter % 15 == 0:
		classify_image(img)
	
	# Update time
	time_counter = time_counter + 1

	# If ESC is pressed
	if keypress == 27:
		exit(0)