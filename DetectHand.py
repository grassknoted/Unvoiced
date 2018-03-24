import cv2
import numpy as np
import pickle

# CHANGE THE FILE NAME TO DIFFERENT HAAR CASCADES TO FIND THE BEST ONE
haar_cascade = 'hand_detection_cascade.xml'

def main():
    # live_stream parameters
    x, y, w, h = 300, 100, 300, 300
    
    # Flags for key presses
    captureFlag, saveFlag, escapeFlag = False, False, False
    
    # Use webcam at VideoCapture(0)
    live_stream = cv2.VideoCapture(0)
    
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

        # The Haar cascade used to detect a hand in the frame
        hand_cascade = cv2.CascadeClassifier(haar_cascade)       
        hands = hand_cascade.detectMultiScale(hsvCrop, 1.3, 5)

        for (x,y,w,h) in hands:
            cv2.rectangle(hsvCrop,(x,y),(x+w,y+h),(0,0,255),2)

        # If 'c' is pressed
        if keypress == ord('c'):
            captureFlag = True

        if captureFlag:
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

        # If ESC is pressed
        if keypress == 27:
            exit(0)

        cv2.imshow("Hand Histogram", hsvCrop)

    
    # Stop using camers
    live_stream.release()

    # Destroy windows created by OpenCV
    cv2.destroyAllWindows()
    
    # Save the histogram as a pickle
    with open("hist", "wb") as f:
        pickle.dump(histogram, f)


if (__name__ == "__main__"):
    main()