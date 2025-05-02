import cv2
import rospy
from std_msgs.msg import String, Int32
from std_msgs.msg import Time
import time, multiprocessing
import pandas as pd

import os
import shutil


def delete_and_create_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Delete the folder and its contents
        shutil.rmtree(folder_path)  # Use os.rmdir(folder_path) if the folder is empty
        print(f"Deleted folder: {folder_path}")

    # Create the new folder
    os.mkdir(folder_path)
    print(f"Created folder: {folder_path}")


delete_and_create_folder('video_cap_script/cap_img/top/')
delete_and_create_folder('video_cap_script/cap_img/side/')



rospy.init_node('img', anonymous=True)
pub_top = rospy.Publisher('Publish_Image_top', String , queue_size=10)
pub_side = rospy.Publisher('Publish_Image_side', String , queue_size=10)
img_ts_pub_top = rospy.Publisher('img_ts_top', String, queue_size=10)
img_ts_pub_side = rospy.Publisher('img_ts_side', String, queue_size=10)
rate = rospy.Rate(10)         


# Initialize the video capture object
# cap = cv2.VideoCapture("/dev/video2")
# cap.set(cv2.CAP_PROP_FPS, 10)
# fps = cap.get(cv2.CAP_PROP_FPS)

# cap2 = cv2.VideoCapture("/dev/video4")
captures = [cv2.VideoCapture("/dev/video2"),
            cv2.VideoCapture("/dev/video4")]
captures[0].set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Width in pixels
captures[0].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

captures[1].set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Width in pixels
captures[1].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the video capture is opened successfully
# if not cap.isOpened(): 
#     print("Error: Could not open video.")
#     exit()
    
# if not cap2.isOpened():
#     print("Error: Could not open video side.")
#     exit()

# def retrieve_data(return_queue):
    
#     data_1, data_2 = return_queue.get()  # Retrieve the two sets of data
#     df_top = pd.DataFrame({"Top Img TS": data_2, "Top Img name": data_1})
#     df_top.to_csv('top_img_out.csv', index=False)
    
#     data_1, data_2 = return_queue.get()
#     df_side = pd.DataFrame({"Side Img TS": data_2, "Side Img name": data_1})
#     df_side.to_csv('side_img_out.csv', index=False) 
    

def capture_video_top(return_queue):
    global captures
    img_list_top = []
    img_ts_list_top = []
    count = 0
    while True:
        # Capture frame-by-frameencoder_ts
        # cam = captures[camera]
        ret, frame = captures[0].read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display the resulting frame
        cv2.imshow('Video', frame)
        count += 1
        img_name = "Image_%d.jpg" % count
        
        
        path = "video_cap_script/cap_img/top/"
    
        cv2.imwrite(os.path.join(path , img_name), frame) 

        # rospy.loginfo(img_name)
        # img_publisher.publish(img_name)
        img_list_top.append(img_name)
        
        timestamp = time.time()
        timestamp_str = str(timestamp)
        print(timestamp_str)    
        # ts_publisher.publish(timestamp_str)
        img_ts_list_top.append(timestamp_str)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return_queue.put((img_list_top, img_ts_list_top))
            captures[0].release()  # If using OpenCV for video capture
            cv2.destroyAllWindows()
            break
    
        rate.sleep()
        
        
def capture_video_side(return_queue):
    global captures
    img_list_side = []
    img_ts_list_side = []
    count = 0
    while True:
        # Capture frame-by-frameencoder_ts
        # cam = captures[camera]
        ret, frame = captures[1].read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display the resulting frame
        cv2.imshow('Video', frame)
        count += 1
        img_name = "Image_%d.jpg" % count
        path = "video_cap_script/cap_img/side/"
    
        cv2.imwrite(os.path.join(path , img_name), frame) 

        # rospy.loginfo(img_name)
        # img_publisher.publish(img_name)
        img_list_side.append(img_name)
        
        timestamp = time.time()
        timestamp_str = str(timestamp)
        print(timestamp_str)    
        # ts_publisher.publish(timestamp_str)
        img_ts_list_side.append(timestamp_str)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return_queue.put((img_list_side, img_ts_list_side))
            captures[1].release()  # If using OpenCV for video capture
            cv2.destroyAllWindows()
            break
    
        rate.sleep()
        
# def monitor_key_press(processes):
#     # Wait for a specific key to press (e.g., 'q')
#     global img_list_side, img_list_top, img_ts_list_side, img_ts_list_top
#     print("Press 'q' to stop both processes.")
#     while True:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Key pressed! Stopping processes.")
#             for p in processes:
#                 p.terminate()  # Terminate the processes
#             break  # Exit the monitoring loop  
    

# Create threads for two cameras
return_queue = multiprocessing.Queue()

process1 = multiprocessing.Process(target=capture_video_top, args=(return_queue,))
process2 = multiprocessing.Process(target=capture_video_side, args=(return_queue,))

process1.start()
process2.start()

# retrieve_data(return_queue)




process1.join()
process2.join()

