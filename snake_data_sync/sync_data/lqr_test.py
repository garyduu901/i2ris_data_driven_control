import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

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
u_min_pitch, u_max_pitch = -1200, 1200
u_min_yaw, u_max_yaw = -1200, 1200


# === LQR Parameters ===
Q_lqr = np.diag([3.8, 0.01, 3.8, 0.01])
R_lqr = np.eye(2) * 0.0001

# Solve the LQR problem
P = scipy.linalg.solve_continuous_are(A, B, Q_lqr, R_lqr)
K_lqr = np.linalg.inv(R_lqr) @ B.T @ P
print(K_lqr)

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
x = np.array([0, 0, 0, 0])
dt = 0.1
N = 500
trajectory = [x]
control_actions = []

for step in range(N):
    current_time = step * dt
    x_ref = desired_trajectory(current_time)

    u = -K_lqr @ (x - x_ref)

    u[0] = np.clip(u[0], u_min_pitch, u_max_pitch)
    u[1] = np.clip(u[1], u_min_yaw, u_max_yaw)

    x = rk4_step(x, u, dt)

    trajectory.append(x)
    control_actions.append(u)

trajectory = np.array(trajectory)
control_actions = np.array(control_actions)

# === Visualization ===
time = np.arange(len(trajectory)) * dt
pitch_angle = trajectory[:, 0]
pitch_velocity = trajectory[:, 1]
yaw_angle = trajectory[:, 2]
yaw_velocity = trajectory[:, 3]

ref_pitch = [desired_trajectory(t)[0] for t in time]
ref_pitch_vel = [desired_trajectory(t)[1] for t in time]
ref_yaw = [desired_trajectory(t)[2] for t in time]
ref_yaw_vel = [desired_trajectory(t)[3] for t in time]

plt.figure(figsize=(12, 6))
plt.plot(time, pitch_angle, color="blue",  label='Pitch Angle')
plt.plot(time, ref_pitch, '--', color="blue", label='Pitch Ref')
plt.plot(time, yaw_angle, color="red", label='Yaw Angle')
plt.plot(time, ref_yaw, '--', color="red", label='Yaw Ref')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.title('Pitch and Yaw Angle Tracking with LQR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 6))
plt.plot(time, pitch_velocity, color="blue",  label='Pitch velocity')
plt.plot(time, ref_pitch_vel, '--', color="blue", label='Pitch velocity Ref')
plt.plot(time, yaw_velocity, color="red", label='Yaw velocity')
plt.plot(time, ref_yaw_vel, '--', color="red", label='Yaw velocity Ref')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (deg/s)')
plt.title('Pitch and Yaw Velocity Tracking with LQR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 6))
plt.plot(time[:-1], control_actions[:, 0], label='Pitch Motor Input')
plt.plot(time[:-1], control_actions[:, 1], label='Yaw Motor Input')
plt.xlabel('Time (s)')
plt.ylabel('Motor Command')
plt.title('LQR Control Inputs Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
