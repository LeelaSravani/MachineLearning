import cv2
import numpy as np
from PIL import Image
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# Load the image and resize it
img = cv2.imread(r'F:/Smart mirror/Lipmakeup test/test3.jpg')

img = cv2.resize(img,(400,400))
#img1=img.copy()

# Convert the image from bgr to hsv colorspace

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Set the range of skin color pixel values in rgb

lower = np.array([0, 48, 80], dtype=np.uint8)

upper = np.array([20, 255, 255], dtype=np.uint8)

# Apply the Gaussian blur and get mask of pixels

img_hsv = cv2.GaussianBlur(img_hsv, (3,3), 0)

mask = cv2.inRange(img_hsv, lower, upper)

# convert single channel mask back into 3 channels

mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

# perform bitwise and on mask to obtain cut-out image that is not skin

masked_img = cv2.bitwise_and(img, mask_rgb)

# Display masked image

cv2.imshow(" masked image",masked_img)

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


# Initialise the haarcascades frontal face classifier

face_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

# Convert color image to gray scale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image

face = face_detect.detectMultiScale(gray, 1.2, 5)

# Initialise left and right face vars

left_face=0

right_half=0

# Create blank images for left and right half faces

left_half=np.zeros(img.shape)

right_half=np.zeros(img.shape)

# Loop over the face

for (x,y,w,h) in face:

    #masked_img_face = masked_img[y:y+h, x:x+w]

    # Calculate the coordinates for left half face

    x_left_start = x

    y_left_start = y

    x_left_end = int(x+w/2)

    y_left_end = int(y+h)

    # Calculate the coordinates for right half face

    x_right_start = x_left_end

    y_right_start = y

    x_right_end = x+w

    y_right_end = y+h

    # Crop the left and right half faces from the masked image

    left_half = masked_img[y_left_start:y_left_end, x_left_start:x_left_end]

    right_half = masked_img[y_right_start:y_right_end, x_right_start:x_right_end]

# Display faces
    
cv2.imshow("face",masked_img_face)

cv2.imshow("left face",left_half)

cv2.imshow("right face",right_half)

# Save the masked left and right half faces

cv2.imwrite("left face.jpg",left_half)

cv2.imwrite("right face.jpg",right_half)

# Read back the left and right face images

left_face = Image.open("left face.jpg")

right_face = Image.open("right face.jpg")

# Print pixel intensities

print("Avg pixel intensity on left face : " ,calcAvgPixel(left_face))

print("Avg pixel intensity on right face : " ,calcAvgPixel(right_face))

# Calculate the avg pixel of each half face

r1,g1,b1= calcAvgPixel(left_face)

r2,g2,b2 = calcAvgPixel(right_face)

color1_rgb = sRGBColor(r1, g1, b1);

color2_rgb = sRGBColor(r2, g2, b2);


# Convert from RGB to Lab Color Space

color1_lab = convert_color(color1_rgb, LabColor);

color2_lab = convert_color(color2_rgb, LabColor);

# Find the color difference

delta_e = delta_e_cie2000(color1_lab, color2_lab)

print ("The difference between the 2 color = ", delta_e)

# Impose a threshold either from user or hard-coded


# threshold = int(input("Enter a threshold: "))
threshold = 25
if delta_e > threshold:
    print("Asymmetry exists!")
else:
    print("Everything is fine!")

cv2.waitKey(0)
cv2.destroyAllWindows()