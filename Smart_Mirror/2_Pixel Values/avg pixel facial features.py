import cv2
from PIL import Image
import numpy as np

# Initialise the haarcascades frontal face classifier

face_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

# Initialise the haarcascades eye classifier

eyes_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_eye.xml")

# Initialise the nose classifier

nose_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\nose.xml")

# Initialise the mouth classifier

mouth_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\mouth.xml")

# load an image and resize it
        
image = cv2.imread(r'F:/Smart mirror/Lipmakeup test/test3.jpg')

image = cv2.resize(image,(400,400))

# Convert color image to gray image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

def calcAvgPixel(im):

    # Extracting all the pixels in the image in the form of list
    
    pixels = list(im.getdata())

    pixel_list = [x for sets in pixels for x in sets]

    # Separating red, blue and green pixels

    pixel_red = pixel_list[::3]

    pixel_green = pixel_list[1::3]

    pixel_blue = pixel_list[2::3]
    
    # Calculating the mean of each channel

    red = int(np.mean(np.array(pixel_red)))

    green = int(np.mean(np.array(pixel_green)))

    blue = int(np.mean(np.array(pixel_blue)))
    
    # Returning the rgb values
    
    return (red, green, blue);

# Detect faces in the image

face = face_detect.detectMultiScale(gray, 1.2, 5)

d={}

# If face is detected

if len(face)!=0:

    # Make a copy of original image

    image1 = image.copy()

    # Loop over the face

    for (x, y, w, h) in face:

        # Draw rectangle around face

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the image upto face

        roi_gray = gray[y:y + h, x:x + w]

        roi_color = image[y:y+h,x:x+w]

        roi_color_f = image1[y:y + h, x:x + w]

        # Save the face image

        cv2.imwrite("face.jpg", roi_color_f)

        # Read back the face image

        im_f = Image.open(r"face.jpg")
        
        # Calculating Avg pixel for face area
        
        print("Avg pixel intensity for face:",calcAvgPixel(im_f))

        # Detect eyes in the image

        eyes = eyes_detect.detectMultiScale(roi_gray, 1.5,5)

        # Counter for no.of eyes detected

        i=1

        # If eyes detected

        if len(eyes)!=0:

            # Loop over each eye detection

            for (ex, ey, ew, eh) in eyes:

                # Draw rectangle around eyes

                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Crop eyes portion in  the face image

                roi_color_e = roi_color_f[ey:ey+eh,ex:ex+ew]

                # Save the eyes image

                cv2.imwrite("eyes.jpg", roi_color_e)

                # Read back the eyes image

                im_e = Image.open(r"eyes.jpg")
                
                # Calculating Avg pixel for face area

                print("Avg pixel intensity for eye",i,":", calcAvgPixel(im_e))

                # Increment the counter after each detection

                i=i+1

                # Break the loop after both the eyes are detected

                if i>2:

                    break

                        # Detect nose in the image

        nose = nose_detect.detectMultiScale(roi_gray,1.1)

        # Counter

        j=1

        # If nose detected

        if len(nose)!=0:

            # Loop over the nose

            for (nx, ny, nw, nh) in nose:

                # Draw rectangle around nose

                cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 255, 0), 2)

                # Crop the nose area from the face

                roi_color_n = roi_color_f[ny:ny + nh, nx:nx + nw]

                # Save the nose image

                cv2.imwrite("nose.jpg", roi_color_n)

                # Read back the nose image

                im_n = Image.open(r"nose.jpg")

                # Calculate the avg pixel in the nose area
                
                print("Avg pixel intensity for nose:", calcAvgPixel(im_n))
                
                # Increment the counter

                j=j+1

                # Break the loop after nose detected

                if j>1:

                    break
        
        # Detect mouth in the image

        mouth = mouth_detect.detectMultiScale(roi_gray, 1.2,45)

        # Counter

        k=1

        # If mouth detected

        if len(mouth)!=0:

            # Loop over the mouth

            for (mx, my, mw, mh) in mouth:

                # Draw rectangle around mouth

                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 255), 2)

                # Crop the mouth portion from the face

                roi_color_m = roi_color_f[my:my + mh, mx:mx + mw]

                # Save the mouth image

                cv2.imwrite("mouth.jpg", roi_color_m)

                # Read back the mouth image

                im_m = Image.open(r"mouth.jpg")

                # Calculate the avg pixel in the mouth area
                
                print("Avg pixel intensity for mouth:",calcAvgPixel(im_m))
                
                # Increment the counter

                k=k+1

                # Break the loop after mouth is detected

                if k>1:

                    break

# Display the image in a window
                    
cv2.imshow("Detected facial feaures", image)

cv2.waitKey(0)

cv2.destroyAllWindows()
