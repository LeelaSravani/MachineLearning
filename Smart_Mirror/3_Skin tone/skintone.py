import cv2
import numpy as np

# Load the image and resize it

img = cv2.imread(r'F:/Smart mirror/Lipmakeup test/test3.jpg')

img = cv2.resize(img,(400,400))

# Convert the image from bgr to hsv colorspace

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Set the range of skin color pixel values

lower = np.array([0, 48, 80], dtype=np.uint8)

upper = np.array([20, 255, 255], dtype=np.uint8)

# Apply the Gaussian blur and get mask of pixels

img_hsv = cv2.GaussianBlur(img_hsv, (3,3), 0)

mask = cv2.inRange(img_hsv, lower, upper)

# convert single channel mask back into 3 channels

mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

# perform bitwise and on mask to obtain cut-out image that is not skin

masked_img = cv2.bitwise_and(img, mask_rgb)

# Display the image in a window

cv2.imshow(" masked image",masked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


        

        