import cv2
import numpy as np

# Load the dictionary for ArUco markers (e.g., DICT_4X4_50)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Generate a marker with ID 42 from the dictionary
marker_id = 42
marker_size = 200  # Size of the marker in pixels (e.g., 200x200)

# Create the marker image
marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# Save the marker image to a file
cv2.imwrite(f'aruco_marker_{marker_id}.png', marker_img)

# Display the marker
cv2.imshow('ArUco Marker', marker_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
