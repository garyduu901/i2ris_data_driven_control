import cv2
import rospy
from std_msgs.msg import String, Int32
from std_msgs.msg import Time
import time


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


counttop = 0
countside = 0
rospy.init_node('img', anonymous=True)
pub = rospy.Publisher('Publish_Image', String , queue_size=10)
img_ts_pub = rospy.Publisher('img_ts', String, queue_size=10)
# rate = rospy.Rate(60)       
rate = rospy.Rate(20)     


# Initialize the video capture object
# cap = cv2.VideoCapture("/dev/video2")
# cap.set(cv2.CAP_PROP_FPS, 10)
# fps = cap.get(cv2.CAP_PROP_FPS)

# cap2 = cv2.VideoCapture("/dev/video4")
captures = [cv2.VideoCapture("/dev/video4"),
            cv2.VideoCapture("/dev/video0")]
captures[0].set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Width in pixels 1920*1080
captures[0].set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

captures[1].set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Width in pixels
captures[1].set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Check if the video capture is opened successfully
# if not cap.isOpened(): 
#     print("Error: Could not open video.")
#     exit()
    
# if not cap2.isOpened():
#     print("Error: Could not open video side.")
#     exit()

while True:
    for i in range(len(captures)):
        # Capture frame-by-frameencoder_ts
       
        ret, frame = captures[i].read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display the resulting frame
        # cv2.imshow('Video', frame)

        if i == 0:
            counttop += 1
            path = "video_cap_script/cap_img/top/"
            img_name = "Image_%d.jpg" % counttop
            cv2.imwrite(os.path.join(path , img_name), frame) 
        
        elif i == 1:
            countside += 1
            path = "video_cap_script/cap_img/side/"
            img_name = "Image_%d.jpg" % countside
            cv2.imwrite(os.path.join(path , img_name), frame) 

        # Break the loop on 'q' key press
        if 0xFF == ord('q'):
            captures[0].release()  # If using OpenCV for video capture
            captures[1].release() 
            break

    rospy.loginfo(img_name)
    pub.publish(img_name)
    
    timestamp = time.time()
    timestamp_str = str(timestamp)
    print(timestamp_str)    
    img_ts_pub.publish(timestamp_str)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        captures[0].release()  # If using OpenCV for video capture
        captures[1].release()  # If using OpenCV for video capture
        break
    

