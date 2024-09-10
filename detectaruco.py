import cv2
import numpy as np
import pickle 

# Load camera calibration data
with open('calibration.pkl', 'rb') as f:
    calibration = pickle.load(f)

with open('cameraMatrix.pkl', 'rb') as f:
    camera_matrix = pickle.load(f)


with open('dist.pkl', 'rb') as f:
    dist_coeffs = pickle.load(f)

#calibration = np.load('calibration_data.npz')
#camera_matrix = calibration['camera_matrix']
#dist_coeffs = calibration['dist_coeffs']

# Load the image containing ArUco markers
#image = cv2.imread('aruco_marker_42.png')
cap = cv2.VideoCapture(0)

while 1:
    #ret,image = cap.read()
    image = cv2.imread("aruco.png")
    # Load the dictionary of markers (e.g., DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    # Detect the markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(image)

    # If markers are detected
    if ids is not None:
    # Define the size of the marker (in meters, for example, 0.05 means 5cm marker side length)
        marker_size = 0.05  # Change this to the actual size of your printed marker

        # Define 3D points for the marker corners in the marker's coordinate system
        # Assume the marker is on the XY plane (z = 0) with a size of marker_size x marker_size
        object_points = np.array([
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0]
        ], dtype=np.float32)

        # Estimate pose for each detected marker
        for i in range(len(ids)):
            # Get the 2D image points of the detected marker corners
            image_points = corners[i][0]

            # Estimate the pose using solvePnP
            ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

            if ret:
                # Draw the axis for the marker
                cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, marker_size / 2)

                # Compute the distance from the camera to the marker
                distance = np.linalg.norm(tvec)
                print(f"Distance from camera to marker {ids[i][0]}: {distance} meters")

                # Draw the detected marker on the image
                cv2.aruco.drawDetectedMarkers(image, corners)
    # if ids is not None:
    #     # Estimate the pose of each marker
    #     rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        
    #     # Draw the markers and axes on the image
    #     for i in range(len(ids)):
    #         cv2.aruco.drawDetectedMarkers(image, corners)
    #         cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

    #         # Get the translation vector (tvecs) for the distance
    #         print(f"Distance from camera to marker {ids[i][0]}: {np.linalg.norm(tvecs[i])} meters")

    # Display the image with detected markers
    cv2.imshow('Aruco Markers', image)
    ch = cv2.waitKey(1)
    if ch == ord("q"):
        break
cv2.destroyAllWindows()