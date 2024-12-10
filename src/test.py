import cv2

# Load the grayscale image
gray_image = cv2.imread('./bin/test.png', cv2.IMREAD_GRAYSCALE)

# Convert the grayscale image to RGB
rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

# Save or display the RGB image
cv2.imwrite('./bin/test_rgb.png', rgb_image)
cv2.imshow('RGB Image', rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()