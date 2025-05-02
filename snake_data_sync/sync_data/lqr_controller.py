import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

import rospy
import cv2, sys

from std_msgs.msg import String
from std_msgs.msg import Float32, Int32
from std_msgs.msg import Time
import matplotlib.pyplot as plt
import message_filters
import time
import math

import pandas as pd


encoder_list0= []
encoder_list1= []
img_list0 = []
# img_list1 = []
img_ts_list = []
encoder_ts_list = []

# === System Parameters ===
theta_estimated = [45, 13, 45, 13]
m1 = m2 = 1.0
k1, c1, k2, c2 = theta_estimated
beta = 45 * np.pi / 180

A = np.array([
    [0, 1, 0, 0],
    [-k1/m1, -c1/m1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, -k2/m2, -c2/m2]
])

B = np.array([
    [0.0, 0.0],
    [np.cos(beta), -np.sin(beta)],
    [0.0, 0.0],
    [np.sin(beta), np.cos(beta)]
])

# Control constraints
u_min_pitch, u_max_pitch = -1000, 1000
u_min_yaw, u_max_yaw = -1000, 1000


# === LQR Parameters ===
Q_lqr = np.diag([3.8, 0.01, 3.8, 0.01])
R_lqr = np.eye(2) * 0.0001

# Solve the LQR problem
P = scipy.linalg.solve_continuous_are(A, B, Q_lqr, R_lqr)
K_lqr = np.linalg.inv(R_lqr) @ B.T @ P
print(K_lqr)

x = np.array([0, 0, 0, 0])
model_output = [x]
control_actions = []
desired_traj = []

# RK4 Integrator
def rk4_step(x, u, dt):
    def dynamics(xi, ui):
        return A @ xi + B @ ui
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Desired Trajectory Generator (Circle)
def desired_trajectory(t):
    radius_pitch = 10
    radius_yaw = 10
    omega = 0.5
    pitch = radius_pitch * np.cos(omega * t)
    yaw = radius_yaw * np.sin(omega * t)
    pitch_velocity = -radius_pitch * omega * np.sin(omega * t)
    yaw_velocity = radius_yaw * omega * np.cos(omega * t)
    return np.array([pitch, pitch_velocity, yaw, yaw_velocity])

# === Simulation ===


# def callback_main(img_name, encoder_cnt0, encoder_cnt1):
def callback_main(img_ts, img_name, encoder_cnt1, encoder_cnt0):
    global img_list0, encoder_list, img_ts_list, encoder_ts_list
    rospy.loginfo("Received from img_pub: %s, from encoder_pub1: %d, from encoder_pub0: %d", img_name.data, encoder_cnt1.data, encoder_cnt0.data)
    img_list0.append(img_name.data)
    # encoder_list0.append(encoder_cnt0.data)
    encoder_list1.append(encoder_cnt1.data)
    encoder_list0.append(encoder_cnt0.data)
    img_ts_list.append(img_ts.data)
    encoder_ts_list.append(time.time())
    
def main():
    global encoder_counts, img_name_st, model_output, control_actions, desired_traj
    dt = 0.2
    N = 500
    
    init_time = time.time()
    rospy.init_node('Position_ROS_example', anonymous=True)
    pub0 = rospy.Publisher('/Request_Position_USB0', Int32, queue_size=10)
    pub1 = rospy.Publisher('/Request_Position_USB1', Int32, queue_size=10)    
    
    # encoder_sub0 = message_filters.Subscriber("/Current_Position_USB0", Int32)
    img_sub0 = message_filters.Subscriber("Publish_Image_top", String)
    encoder_sub1 = message_filters.Subscriber("/Current_Position_USB1", Int32)
    encoder_sub0 = message_filters.Subscriber("/Current_Position_USB0", Int32)
    
    img_ts_sub = message_filters.Subscriber("img_ts", String)
    
    ts = message_filters.ApproximateTimeSynchronizer([img_ts_sub, img_sub0, encoder_sub1, encoder_sub0], 10,  slop=0.05, allow_headerless=True)
    # ts = message_filters.ApproximateTimeSynchroniarrayzer([img_sub 0, encoder_sub0, encoder_sub1], 10,  slop=0.05, allow_headerless=True)
    ts.registerCallback(callback_main)

    rate = rospy.Rate(10) 

    zeros = [0] * 100
    x = np.array([0, 0, 0, 0])

    
    while not rospy.is_shutdown():
        position_request1 = Int32()
        position_request0 = Int32()
        
        for i in range(N):
            current_time = time.time() - init_time
            
            x_ref = desired_trajectory(current_time)
            desired_traj.append(x_ref)

            u = -K_lqr @ (x - x_ref)

            u[0] = np.clip(u[0], u_min_pitch, u_max_pitch)
            u[1] = np.clip(u[1], u_min_yaw, u_max_yaw)
            
            pos0 = int(u[0])
            pos1 = int(u[1])
            position_request1.data = pos1
            position_request0.data = pos0
            rospy.loginfo(f"--------------Publishing Request_Position1: {position_request1.data},Publishing Request_Position0: {position_request0.data}-------------")
            pub0.publish(position_request0)
            pub1.publish(position_request1)
            rate.sleep()

            x = rk4_step(x, u, dt)

            model_output.append(x)
            control_actions.append(u)
            dt = (time.time() - init_time) - current_time
            print(dt)
            
            if dt < 0.05:
                rospy.loginfo(f"Run Video Capture Code First: videocapture_1219.py") 
                break

            
        break

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    df = pd.DataFrame({"Img TS": img_ts_list, "Img name": img_list0, "Enc TS": encoder_ts_list, "Encoder counts 1": encoder_list1, "Encoder counts 0": encoder_list0})
    # df = pd.DataFrame({"TS": time_stamps, "Img name": img_list0, "Encoder counts 0": encoder_list0, "Encoder counts 1": encoder_list1})
    df.to_csv('out.csv', index=False)
    
    captures = [cv2.VideoCapture("/dev/video5"),
                cv2.VideoCapture("/dev/video0")]
    captures[0].release()  # If using OpenCV for video capture
    captures[1].release()  # If using OpenCV for video capture

    model_output = np.array(model_output)
    control_actions = np.array(control_actions)
    desired_traj = np.array(desired_traj)

    # === Visualization ===
    # time = np.arange(len(model_output)) * dt
    pitch_angle_model = model_output[1:, 0]
    pitch_velocity_model = model_output[1:, 1]
    yaw_angle_model = model_output[1:, 2]
    yaw_velocity_model = model_output[1:, 3]
    
    print(pitch_angle_model.shape)
    
    ref_pitch = desired_traj[:, 0]
    ref_pitch_vel = desired_traj[:, 1]
    ref_yaw = desired_traj[:, 2]
    ref_yaw_vel = desired_traj[:, 3]
    
    pitch_input = control_actions[:, 0]
    yaw_input = control_actions[:, 1]
    
    # Combine arrays as before
    combined_data = np.column_stack((
        pitch_angle_model, pitch_velocity_model, 
        yaw_angle_model, yaw_velocity_model,
        ref_pitch, ref_pitch_vel,
        ref_yaw, ref_yaw_vel,
        pitch_input, yaw_input
    ))

    # Define column names
    columns = [
        "pitch_angle_model", "pitch_velocity_model", 
        "yaw_angle_model", "yaw_velocity_model", 
        "ref_pitch", "ref_pitch_vel", 
        "ref_yaw", "ref_yaw_vel", 
        "pitch_input", "yaw_input"
    ]

    # Create DataFrame
    df = pd.DataFrame(combined_data, columns=columns)

    # Save to CSV
    df.to_csv("combined_data.csv", index=False)
    # ref_pitch = [desired_trajectory(t)[0] for t in time]
    # ref_pitch_vel = [desired_trajectory(t)[1] for t in time]
    # ref_yaw = [desired_trajectory(t)[2] for t in time]
    # ref_yaw_vel = [desired_trajectory(t)[3] for t in time]

    # plt.figure(figsize=(12, 6))
    # plt.plot(time, pitch_angle, color="blue",  label='Pitch Angle')
    # plt.plot(time, ref_pitch, '--', color="blue", label='Pitch Ref')
    # plt.plot(time, yaw_angle, color="red", label='Yaw Angle')
    # plt.plot(time, ref_yaw, '--', color="red", label='Yaw Ref')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Angle (deg)')
    # plt.title('Pitch and Yaw Angle Tracking with LQR')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    sys.exit(0)

# plt.figure(figsize=(12, 6))
# plt.plot(time, pitch_velocity, color="blue",  label='Pitch velocity')
# plt.plot(time, ref_pitch_vel, '--', color="blue", label='Pitch velocity Ref')
# plt.plot(time, yaw_velocity, color="red", label='Yaw velocity')
# plt.plot(time, ref_yaw_vel, '--', color="red", label='Yaw velocity Ref')
# plt.xlabel('Time (s)')
# plt.ylabel('Angular Velocity (deg/s)')
# plt.title('Pitch and Yaw Velocity Tracking with LQR')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# plt.figure(figsize=(12, 6))
# plt.plot(time[:-1], control_actions[:, 0], label='Pitch Motor Input')
# plt.plot(time[:-1], control_actions[:, 1], label='Yaw Motor Input')
# plt.xlabel('Time (s)')
# plt.ylabel('Motor Command')
# plt.title('LQR Control Inputs Over Time')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
