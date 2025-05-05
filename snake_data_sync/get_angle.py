import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from math import atan2, asin, degrees

# File directories
filename = "snake_data_sync" # Main directory (change as needed)
csv_path = filename + "/sync_data/out.csv" # encoder+image name csv location
img_path = filename + "/video_cap_script/cap_img/" # Captured images location

# Converting rotational vector ro euler angles
# rvec: rotational vector
def rvec_to_euler_angles(rvec):
    # Convert rvec to a rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Extract Euler angles from the rotation matrix
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6  # Check for singularity

    if not singular:
        x = atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # Roll
        y = asin(-rotation_matrix[2, 0])                        # Pitch
        z = atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Yaw
    else:
        x = atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = asin(-rotation_matrix[2, 0])
        z = 0

    # Convert radians to degrees for readability
    roll = degrees(x)
    pitch = degrees(y)
    yaw = degrees(z)
    
    return roll, pitch , yaw

# Extract euler angles from ArUco labels
'''
Input Parameters:
frame: Current frame to be processed
roll_list, pitch_list, yaw_list: list for recording angles
roll_ref = 0, pitch_ref = 0, yaw_ref = 0: reference angle at home
'''
def get_euler_angles(frame, roll_list, pitch_list, yaw_list, cam, img_array, roll_ref = 0, pitch_ref = 0, yaw_ref = 0, aid = 0):
    # Convert to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejectedCandidates = detector.detectMarkers(frame)

    markerLength = 0.03

    # Calibrated camera matrix
    # Top
    if cam == "side":
        camera_matrix = np.array([[1.73248470e+04, 0.00000000e+00, 7.81146835e+02],
                                [0.00000000e+00, 1.63636632e+04, 3.44176153e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        dist_coeffs = np.array([[5.32260993e+01, -3.79225987e+04, -9.12305448e-02, -1.55317628e-01, -8.23481304e+01]])

    # Side
    if cam == "top":
        camera_matrix = np.array([[1.05185452e+04, 0.00000000e+00, 7.04149471e+02],
                                [0.00000000e+00, 1.07492107e+04, 2.40750193e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        dist_coeffs = np.array([[7.58013879e+00, -2.04446767e+03, -3.19411398e-02,  4.27681822e-03, -2.90622118e+00]])

    object_points = np.array([
            [-markerLength / 2, -markerLength / 2, 0],
            [ markerLength / 2, -markerLength / 2, 0],
            [ markerLength / 2,  markerLength / 2, 0],
            [-markerLength / 2,  markerLength / 2, 0],
        ], dtype=np.float32)

    if ids is not None:
        # Draw markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        arcuo_ids = [int(array[0]) for array in ids]
        contains_id = any(aid in arcuo_ids for aid in arcuo_ids)
        if contains_id:
            # print(corners)
            # Estimate pose of each marker
            # rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
            success, rvec, tvec = cv2.solvePnP(object_points, corners[0], camera_matrix, dist_coeffs)
            
            # Draw respective axis on the frame, x = red, y = green, z = blue
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

            # Append image to a storing array for visualization/video
            img_array.append(frame)

            # Convert the rotation vector to euler angles
            roll, pitch, yaw = rvec_to_euler_angles(rvec)

            # Calculate the angles with respective to the reference (initial angles)
            roll -= roll_ref
            pitch -= pitch_ref
            yaw -= yaw_ref

        else:
            # If the no marker recognized, NaN
            roll = np.nan
            pitch = np.nan
            yaw = np.nan

    else:
        # If the no marker recognized, NaN
        roll = np.nan
        pitch = np.nan
        yaw = np.nan

    # Record the values to their respective list
    roll_list.append(roll)
    pitch_list.append(pitch)
    yaw_list.append(yaw)

    # Display the resulting frame
    cv2.imshow('ArUco Detection', frame)

    
def get_angle_list(df, img_path, cam, aid):
    rolls_head = []
    pitches_head = []
    yaws_head = []

    rolls = []
    pitches = []
    yaws = []

    img_array = []
    img_path = img_path + cam + "/"

    # Record the first 9 values as the reference angle
    # for i in range(1, 30):
    #     frame = cv2.imread(img_path + df.iloc[i, 1])
    #     get_euler_angles(frame, rolls_head, pitches_head, yaws_head, cam, img_array)

    for i in range(5, 30):
        frame = cv2.imread(img_path + "Image_" + str(i) + ".jpg")
        get_euler_angles(frame, rolls_head, pitches_head, yaws_head, cam, img_array)

    # Calculate the average of the recorded reference angle
    roll_ref =  np.nanpercentile(sorted(rolls_head), 55)
    pitch_ref = np.nanpercentile(sorted(pitches_head), 75)
    yaw_ref = np.nanpercentile(sorted(yaws_head), 55)  

    # Calculate angles for all captured images
    for i in range(df["Img name"].shape[0]):
        frame = cv2.imread(img_path + df.iloc[i, 1])
        if frame is None:
            print(img_path + df.iloc[i, 1])
            frame = cv2.imread(img_path + df.iloc[0, 1])
            get_euler_angles(frame, rolls, pitches, yaws, cam, img_array, roll_ref, pitch_ref, yaw_ref, aid)
        else:
            get_euler_angles(frame, rolls, pitches, yaws, cam, img_array, roll_ref, pitch_ref, yaw_ref, aid)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cv2.destroyAllWindows()
    height, width, layers = frame.shape
    size = (width,height)
    out = cv2.VideoWriter(filename + '/track_output_' + cam + '.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    # Saving and plotting
    rolls = np.array(rolls)
    pitches = np.array(pitches)
    yaws = np.array(yaws)

    # SIDE CAM
    if cam == "side":
        df['pitches'] = yaws
    
    # TOP CAM
    if cam == "top":
        df['yaws'] = yaws

    # plt.plot()
    # plt.title("Angles v.s. Encoder Counts")
    # colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']
    # plt.scatter(df["Encoder counts"], yaws, s = 3)
    # plt.xlabel("Encoder Counts")
    # plt.ylabel("Angle in Degrees")

    # plt.gca().invert_yaxis()
    # plt.show()
    # plt.savefig(filename + "/plot_" + cam +".png", dpi=600)  # Save as PNG with high resolution
    # plt.close()

    cv2.destroyAllWindows()

if __name__=="__main__":
    # Initiate the lists and data frames
    df = pd.read_csv(csv_path)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters =  cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    get_angle_list(df, img_path, "top", 9)
    get_angle_list(df, img_path, "side", 14)
    # df["pitches"] = df["pitches"] + 10
    df["yaws"] = df["yaws"] -3.7

    

    df = df.drop(columns=["Img name"])
    df.to_csv(filename + '/output_w_angles.csv', index=False)
