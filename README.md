# Control of Eye Snake Robot with Data-Driven Models

This README provides step-by-step instructions for data collection and model validation using the **Improved Integrated Robotics Intraocular Snake (I2RIS)**.

---

## System Overview

The I2RIS system captures the bending angles of the snake robot by applying various encoder counts to its motors. The setup includes:

- Two microscopes (for pitch and yaw angle measurement)
- MAXON motors with EPOS controllers
- A workstation running ROS Noetic and image processing tools

Using angle calculation algorithms and data-driven models, the system enables **automated data acquisition and validation** of robotic bending behavior.

---

## Getting Started

### Pre-Startup Checklist

1. Ensure all physical connections are secure:
   - Microscopes ↔ Workstation
   - EPOS controllers ↔ Motors
   - Controller ↔ Workstation (with EPOS library installed)

2. Start communication with the controllers:
   - Open **two terminal windows** and execute the following in each:

     ```bash
     source /opt/ros/noetic/setup.bash
     cd /epos_ros/devel/lib/epos_ros
      ./epos_ros_node –p USBX –n X 
     ```

3. Open a third terminal window and run:

   ```bash
   source /opt/ros/noetic/setup.bash
   roscore
   ```

---

## Data Collection Procedure

Navigate to your workspace:

   ```bash
   cd /path/to/snake_data_sync
   ```
<!--
2. Open the `sync_data` subfolder in **Visual Studio Code (VS Code)**:

   * Open either `sync_data_xxxx_one_motor.py` or `sync_data_xxxx_two_motor.py`.

3. Choose a waypoint generation strategy:

   * **Sawtooth pattern**:

     ```python
     gen_waypoints(cycle, max_range, interval)
     ```
   * **Decaying sine waves**:

     ```python
     gen_sin_waypoints(q_max, f_signal, num_command, a, q_offset)
     ```
   * **CSV-based waypoints**
     ```python
     test_data = pd.read_csv('input.csv')
      waypoints = test_data.iloc[:, 0].tolist()
     ```

4. Set homing position (if needed):

   ```python
   waypoints = np.zeros(...)
   ```

5. To control a single motor:

   * Select the appropriate publisher and subscriber:

     * Pitch (USB0): `Request_Position_USB0`, `Current_Position_USB0`
     * Yaw (USB1): `Request_Position_USB1`, `Current_Position_USB1`

---
-->
## Video Capture

1. In another VS Code window, run:

   ```bash
   videocapture_1219.py
   ```

2. Identify microscope device ports:

   ```bash
   v4l2-ctl --list-devices
   ```

   (Typically, devices are `/dev/video0`, `/dev/video4`, and `/dev/video6`)

3. Confirm real-time image feed windows open for both microscopes.
4. Start the front camera by running `cap_traj.py` in a new VS Code window.
---

## Running Data Collection (Updated)

1. Run one of the controller script:
- `mppi_controller.py`
- `mppi_controller_kim_fwd_kin.py`
- `lqr_controller_kim_fwd_kin.py`
- `lqr_controller.py`
2. Select desired trajectory by modifying parameters and attributes in function `desired_trajectory(t, desired_traj='')`:
-  `'circle'`: trajectory of sine and cose wave inputs, modifying `omega, phase_shift` gives different trajectories.
-  `'rect'`: rectangular trajectory, modifying `width_ratio, r` gives different size of rectangles.
-  `'square'`: rectangular trajectory, modifying `r` gives different size of squares.
3. Once the robot finishes running, press `'q'` in the front camera feed window to save the recorded video.
4. Run `aruco_marker_track.py` to track the trajectory.
5. Run the PCC modeling scripts to map the trajectory in the X-Z plane. *(To be added)*
6. Stop video capture manually:
   * Press `q` in the real-time video window for all windows showing camera feed.
7. Home the robot after finish the data collection by running `sync_data_0221_two_motor.py`.
8. Run `aruco_marker_track.py` to get the video with trajectory.
---

## Emergency Stop

* **Preferred**: Terminate the thread via VS Code terminal or stop communication with motors.
* **Avoid**: Power shutdown. This may cause the motors to treat the last position as the home (encoder = 0), breaking homing routines.

---

## Angle Calculation

Run the following script to compute angles:

```bash
get_angle.py
```

Make sure paths are adjusted relative to your main folder.
Outputs are saved as `output_w_angles.csv`
---

## Scripts and Folder Overview

### Controlling the Snake (in `sync_data/`)

| File                          | Description                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------- |
| `combined_data.csv`           | Output of `lqr_controller.py`, includes reference, simulated, and actual states |
| `input.csv`                   | Sawtooth input used for model training                                          |
| `lqr_controller.py`           | An implementation of LQR controller to control I2RIS motion                     |
| ~~`lqr_test.py`~~             | ~~Simulates inputs, reference signals, and results~~                            |
| `out.csv`                     | Output from data collection                                                     |
| `sync_data_0221_two_motor.py` | Controls both motors simultaneously                                             |
| `sync_data_0221_one_motor.py` | Controls a single motor                                                         |
| `sync_data.py`                | Original script, **deprecated**                                                 |
| `waypoints.py`                | Generates waypoints (currently unused)                                          |

---

### Image Capture and Calibration

| File                                          | Description                                         |
| --------------------------------------------- | --------------------------------------------------- |
| `cap_img/`                                    | Collected images for angle calculation              |
| `aruco1.py`                                   | ArUco marker detection and ID display               |
| `cameraCalibration.py`                        | Generates camera matrix and distortion coefficients |
| `saveChessBoradImagesForCameraCalibration.py` | Captures checkerboard images for calibration        |

---

### Miscellaneous

| File                   | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| `app.py`, `index.html` | GUI (currently not used)                                    |
| `output_w_angles.csv`  | Final data output; clean accordingly (remove NaNs, check sync) |

## Update 1 (May 17th, 2025)

The following controlling scripts have been added to the repo, with some updates to their functionality:

### Controlling the Snake (`sync_data/`)

| File                             | Description                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| `mppi_controller.py`             | MPPI controller implementation for controlling the I2RIS robot                       |
| `mppi_controller_kim_fwd_kin.py` | MPPI controller with a forward kinematics model by Dr. Jin Seob Kim                  |
| `lqr_controller_kim_fwd_kin.py`  | LQR controller with a forward kinematics model by Dr. Jin Seob Kim                   |
| `i2ris_kinematics.py`            | Kinematics class based on Dr. Jin Seob Kim’s model (only forward kinematics is used) |
| `sync_data_0221_two_motor.py`    | Script dedicated to homing the position of the I2RIS robot                           |

### Image Capture and Calibration

| File                    | Description                                                                |
| ----------------------- | -------------------------------------------------------------------------- |
| `cap_traj.py`           | Launches the front camera and starts recording; press `q` to exit and save |
| `aruco_marker_track.py` | Tracks ArUco markers; used to check I2RIS trajectory after control scripts |
| `tracked_output.mp4`    | Sample motion trajectory tracked using ArUco markers                       |
| `traj.mp4`              | Video recorded via `cap_traj.py`                                           |

---

## Notes

* Always ensure synchronization between motor commands and video data for accurate analysis by checking the time stamps.
* Clean output CSV files before using them for training or evaluation.

---

## Contact

For contributing and troubleshooting, contact pdu5@jh.edu.

