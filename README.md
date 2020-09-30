# Unvoiced
Application that converts American Sign Language to Speech.

This application uses transfer learning with an Inception V3 architecture that can be found at: https://github.com/xuetsing/image-classification-tensorflow


## Requirements
To install the necessary requirements to run the command:  

`sudo sh install_requirements.sh`


## Files
1. `old_model.py` The first CNN model that was tried. Scrapped because it didn't give good accuracy on real time test images. (Not used anymore)

2. `live_demo.py` prediction of the sign language alphabet that is shown by the speaker on live stream.

3. `query_classification.py` classification of a given test image.

## Dataset
The dataset used for this project was created by the owner of this repository. It is available on Kaggle as the [ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet) Dataset.
https://www.kaggle.com/grassknoted/asl-alphabet

#### query_classification.py
To run this file:
`python3 query_classification.py ./Test\ Images/<Letter>_test.jpg`

##### Example:

Running `python3 query_classification.py ./Test\ Images/L_test.jpg` should classify the image and predict the letter _L_

##### How it works:

This file generates a letter prediction for the sign in the query image using the trained model in the file `trained_model_graph.pb` which is a _PureBasic_ file that stores the model trained to classify ASL Alphabets.
 This file also uses `training_set_lables.txt` for the order in which the training was done.

The prediction is spoken using Google's Text to Speech API. This is the classification which will finally be applied to the live stream model. 


#### live_demo.py

By default, it is works in real time. To change to _capture mode_ press `C`
In capture mode, classification is done on the region of interest only when `C` is pressed.

Pressing `R` goes back to real time mode.

Pressing `ESC` closes the live stream and exits the program.
