import cv2
import numpy as np

# Get the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# Create detector parameters
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


# Initialize video capture
cap = cv2.VideoCapture("/dev/video4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejectedCandidates = detector.detectMarkers(gray)
    print("Detected markers:", ids)
    # Draw detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Display the frame
    cv2.imshow('ArUco Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()