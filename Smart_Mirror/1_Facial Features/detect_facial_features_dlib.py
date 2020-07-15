import cv2
import dlib

# initialize dlib's face detector (HOG-based) and then create

face_detect = dlib.get_frontal_face_detector()

# the facial landmark predictor

predictor = dlib.shape_predictor("F:\shape_predictor_68_face_landmarks.dat")

# load the input image, resize it, and convert it to grayscale

image = cv2.imread(r'F:/Smart mirror/Lipmakeup test/test1.jpg')

image = cv2.resize(image,(400,400))

grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image

rects = face_detect(grayimage,1)

# loop over the face detections

for (i,rect) in enumerate(rects):

    # determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array

    shape = predictor(grayimage,rect)

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box

    shape = face_utils.shape_to_np(shape)

    (x,y,w,h) = face_utils.rect_to_bb(rect)

    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)

    # show the face number

    cv2.putText(image, "Face {}".format(i+1), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)


    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image

    for (x,y) in shape:

        cv2.circle(image,(x,y),1,(0,255,0),2)

# Dispaly the image in a window
        
cv2.imshow("Image with facial marks detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
