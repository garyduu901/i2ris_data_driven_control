import cv2
import numpy as np

# === ArUco Setup ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# === Load Input Video ===
video_path = "marker_video.mp4"  # Replace with your file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

# === Get Video Properties ===
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = "aruco_traced_output.mp4"

# === Setup Video Writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# === Track Center Positions in Pixel Space ===
pixel_trajectory = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for marker_corners in corners:
            pts = marker_corners[0]  # shape: (4, 2)
            center = np.mean(pts, axis=0).astype(int)
            pixel_trajectory.append(tuple(center))

    # Draw trajectory
    for point in pixel_trajectory:
        cv2.circle(frame, point, radius=3, color=(0, 0, 255), thickness=-1)

    # Write the frame to output
    out.write(frame)

    # Optional display
    cv2.imshow("Aruco Pixel Trace", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved traced video to: {output_path}")
