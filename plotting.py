import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("output_cleaned.csv")
df = pd.read_csv("snake_data_sync\output_w_angles.csv")
model_data = pd.read_csv("snake_data_sync\sync_data\combined_data.csv")

# plt.scatter(df['pitches'], df['yaws'], label='col1 vs col2', s=5)
# plt.plot(df['Enc TS'], df['pitches'], label='pitches')
# plt.plot(df['Enc TS'], df['yaws'], label='yaws')
# plt.ylim(-30, 30)  # Set y-axis from 0 to 40


yaw_ref = model_data['ref_yaw'].values
yaw_robot = df['yaws'].values
time = df['Enc TS'].values
time = time - time[0]  # Normalize time to start from 0
end_time = time[-1]

time_for_ref = np.linspace(0, end_time, len(yaw_ref))
time_for_robot = np.linspace(0, end_time, len(yaw_robot))

plt.plot(time_for_ref, yaw_ref,  label='Ref Yaw')
plt.plot(time_for_robot, yaw_robot + 10 , label='Robot Yaw')


# pitch_ref = model_data['ref_pitch'].values
# pitch_robot = df['pitches'].values
# time = df['Img TS'].values
# time = time - time[0]  # Normalize time to start from 0
# end_time = time[-1]

# time_for_ref = np.linspace(0, end_time, len(pitch_ref))
# time_for_robot = np.linspace(0, end_time, len(pitch_robot))

# plt.plot(time_for_ref, pitch_ref,  label='Ref Pitch')
# plt.plot(time_for_robot, pitch_robot + 5, label='Robot Pitch')

# plt.plot(df['Enc TS'], df['yaws'], label='yaws')
# plt.ylim(-30, 30)  # Set y-axis from 0 to 40


plt.xlabel('Time')
plt.ylabel('Angles (degrees)')
plt.title('Yaw Reference vs Robot Pitch')
plt.legend(loc='upper right')
plt.show()