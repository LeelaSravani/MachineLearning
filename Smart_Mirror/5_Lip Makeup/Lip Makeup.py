# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:23:42 2019

@author: user
"""

import cv2

import numpy as np

from keras.models import load_model




# Load the saved deep learning model for lip makeup detection
        
model = load_model('Lip Makeup.h5')

# Read the image

test_image = cv2.imread(r'F:\Smart mirror\Lipmakeup test\test7.jpg')

# Display the image

cv2.imshow('image', test_image)

# Haarcascade Classifier for frontal face detection

face_cascade = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

# Convert the color image to grayscale

gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image

face = face_cascade.detectMultiScale(gray, 1.2, 5)

# Loop over each face

for (x, y, w, h) in face:
    
    # Crop the face in the image
    
    test_image = test_image[y:y + h, x:x + w]
    
# If face is present 

if (len(test_image)!=0):
    
    # resize the image
    
    test_image = cv2.resize(test_image, (130, 150))
    
    # Convert the image to array format
    
    test_image = np.array(test_image)
    
    # Convert the datatype o float
    
    test_image = test_image.astype('float32')
    
    # Normalize the image
    
    test_image /= 255
    
    # Adding an extra dimension as tensorflow needs it
    
    test_image = np.expand_dims(test_image, axis=0)
    
    # Predicting the class probabilities of the image
    
    probs = model.predict(test_image)

    # Predicting the test image
    
    print(probs)
    
    #print(model.predict_classes(test_image))
    
    #print('Predicted:', names[model.predict_classes(test_image)[0]])
    
    # Converting the probalities into percentages

    for lis in probs:
        #per = lis[clas[0]]*100
        lips = round(lis[0]*100,3)
        nolip = round(lis[1]*100,3)
        
    # Print the predictions
    
    print('Prediction: \nLip Makeup {0}% \nNo Lip makeup {1}%'.format(lips,nolip))
    
    # Print the comment based on the lip makeup percentage
    if lips < 25:
        
        print('Looks like there\'s no lip makeup')

    elif lips < 50:
        
        print('You can even improve your lip makeup')

    elif lips < 75:
        
        print('Looking good!')

    else:
        
        print('Almost perfect!')



cv2.waitKey(0)

cv2.destroyAllWindows()
