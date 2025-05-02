import rospy
import cv2, sys
import numpy as np

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
pos_i = 0
   
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
       

# def callback_encoder_USB0(data):
#     global encoder_counts
#     rospy.loginfo("Receiving encoder: %d", data.data)
#     encoder_list.append(data)

def gen_waypoints(cycle, max_range, interval):
    wayspoints = list(range(0, max_range + 1, interval))
    for i in range(cycle - 1):
        wayspoints += list(range(max_range, -max_range - 1, -interval))
        wayspoints += list(range(-max_range, max_range + 1, interval))
    wayspoints += list(range(max_range, -max_range - 1, -interval))
    wayspoints += list(range(-max_range, 1, interval))
    return wayspoints

def gen_sin_waypoints(q_max, f_signal, num_command, a, q_offset):
    f_command = 1
    f_signal = f_signal/100
    # t = np.linspace(0, int(num_command / f_command), num_command) 
    t = np.linspace(0, int(num_command / f_command), 800+1) # 0203 test
    # q_input = (q_max * np.exp(-f_signal * 0.13 * t)) * (np.sin(2 * np.pi * f_signal * t) + a) + q_offset
    q_input = (q_max * np.exp(-f_signal * 0.2 * t)) * (np.sin(2 * np.pi * f_signal * t) + a) + q_offset
    q_input = np.round(q_input).astype(int)
    
    if q_input[-1]!=0:
        # no matter what let it going back to zero at the end 0205
        last_value = q_input[-1]
        transition_length = 20
        transition = np.linspace(last_value,0,transition_length)
        q_input = np.concatenate((q_input, transition[1:]))
        
    
    # print('q_input',q_input)
    plt.plot(q_input,'o', markersize = 3, color = 'r', linewidth =1)
    plt.grid(True)
    plt.show()
    return q_input

def main():
    global encoder_counts, img_name_str, pos_i
    rospy.init_node('Position_ROS_example', anonymous=True)
    pub0 = rospy.Publisher('/Request_Position_USB0', Int32, queue_size=10)
    pub1 = rospy.Publisher('/Request_Position_USB1', Int32, queue_size=10)    
    
    # encoder_sub0 = message_filters.Subscriber("/Current_Position_USB0", Int32)
    img_sub0 = message_filters.Subscriber("Publish_Image_top", String)
    encoder_sub1 = message_filters.Subscriber("/Current_Position_USB1", Int32)
    encoder_sub0 = message_filters.Subscriber("/Current_Position_USB0", Int32)
    
    img_ts_sub = message_filters.Subscriber("img_ts", String)
    
    ts = message_filters.ApproximateTimeSynchronizer([img_ts_sub, img_sub0, encoder_sub1, encoder_sub0], 10,  slop=0.05, allow_headerless=True)
    # ts = message_filters.ApproximateTimeSynchronizer([img_sub 0, encoder_sub0, encoder_sub1], 10,  slop=0.05, allow_headerless=True)
    ts.registerCallback(callback_main)
    
    # if len(sys.argv) > 4:
    #     cycle = int(sys.argv[1]) 
    #     max_range = int(sys.argv[2]) 
    #     step = int(sys.argv[3]) 
    #     freq = float(sys.argv[4])
    #     waypoints = gen_waypoints(cycle, max_range, step)
    # else:
    #     max_range = int(sys.argv[1]) 
    #     pts = int(sys.argv[2]) 
    #     freq = float(sys.argv[3])
    #     waypoints = gen_sin_waypoints(max_range, freq, pts)
    # rate = rospy.Rate(freq) 
    
    # loop_num = 0
    # cycle_num = 1
    # pos = 0
    # step = 100
    # range_max = 2000
    rate = rospy.Rate(10) 

    # waypoints = gen_waypoints(cycle_num, range_max, step)
    zeros = [0] * 100
    
    #
    test_data = pd.read_csv('input.csv')
    waypoints = test_data.iloc[:, 0].tolist()
    # waypoints = gen_waypoints(1, 1100, 5)
    # waypoints = gen_waypoints(1, 1300, 7)
    # waypoints = list(range(0,1301,10)) + list(range(1296,-1,-10)) 
    # waypoints = list(range(0,-1301,-10)) + list(range(-1296,1,10))
    # waypoints = 3 * waypoints
    
    print(len(waypoints))
    print(waypoints)
    
    # waypoints = gen_sin_waypoints(1800, 2, 250, 1, -1800)
    
    #0208
    # test_data = pd.read_csv('test3.csv')
    # waypoints =0 test_data.iloc[:,1].values
    # print(waypoints)



    # waypoints = gen_sin_waypoints(1800, 2.5, 400)  #1800, 1.5/2/2.5, 400
    waypoints = zeros
    # print(waypoints)
    # print(np.diff(waypoints))
    
    while not rospy.is_shutdown():
        if pos_i > len(waypoints) - 1:
            break
            
        # Rotate the robot base for any phi angle
        position_request1 = Int32()
        position_request0 = Int32()
        pos1 = int(waypoints[pos_i]) # change to zeros if needed
        tan_phi = math.tan(math.pi * (-15)/180)
        pos0 = int(tan_phi * waypoints[pos_i])
        position_request1.data = pos1
        position_request0.data = pos0
        pos_i += 1
        
        rospy.loginfo(f"--------------Publishing Request_Position1: {position_request1.data},Publishing Request_Position0: {position_request0.data}-------------")
        pub0.publish(position_request0)
        pub1.publish(position_request1)
        rate.sleep()
        
        # Go back to zero position
        # position_request = Int32()
        # pos = int(waypoints[pos_i]) # change to zeros if needed
        # position_request.data = pos
        # pos_i += 1
        
        # rospy.loginfo(f"--------------Publishing Request_Position: {position_request.data}-------------")
        # pub0.publish(position_request)
        # pub1.publish(position_request)
        # rate.sleep()
        

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

    sys.exit(0)

    
    