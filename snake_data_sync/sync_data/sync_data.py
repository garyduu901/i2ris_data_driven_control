import rospy
import cv2, sys
import numpy as np

from std_msgs.msg import String
from std_msgs.msg import Float32, Int32
from std_msgs.msg import Time

import message_filters
import time

import pandas as pd

   
# def callback_main(img_name, encoder_cnt0, encoder_cnt1):
def callback_main(encoder_cnt1):
    global img_list_top, img_list_side, encoder_list, img_ts_list_top, img_ts_list_side, encoder_ts_list
    # rospy.loginfo("Received from img_pub: %s, from encoder_pub1: %d", img_name.data, encoder_cnt1.data)
    # img_list_top.append(img_name_top.data)
    # img_list_side.append(img_name_side.data)

    # encoder_list0.append(encoder_cnt0.data)

    encoder_ts_list.append(time.time())
    encoder_list1.append(encoder_cnt1.data)

    # img_ts_list_top.append(img_ts_top.data)
    # img_ts_list_side.append(img_ts_side.data)
    

def gen_waypoints(cycle, max_range, interval):
    wayspoints = list(range(0, max_range + 1, interval))
    for i in range(cycle - 1):
        wayspoints += list(range(max_range, -max_range - 1, -interval))
        wayspoints += list(range(-max_range, max_range + 1, interval))
    wayspoints += list(range(max_range, -max_range - 1, -interval))
    wayspoints += list(range(-max_range, 1, interval))
    return wayspoints

def gen_sin_waypoints(q_max, f_signal, num_command, a = 0, q_offset = 0):
    f_command = 1
    f_signal = f_signal/100
    t = np.linspace(0, int(num_command / f_command), num_command)  #
    q_input = (q_max * np.exp(-f_signal * 0.1 * t)) * (np.sin(2 * np.pi * f_signal * t) + a) + q_offset
  
    return q_input

def main():
    global encoder_counts, img_name_str, pos_i
    rospy.init_node('Position_ROS_example', anonymous=True)
    # pub0 = rospy.Publisher('/Request_Position_USB0', Int32, queue_size=10)
    pub1 = rospy.Publisher('/Request_Position_USB1', Int32, queue_size=10)    
    
    # encoder_sub0 = message_filters.Subscriber("/Current_Position_USB0", Int32)
    # img_sub_top = message_filters.Subscriber("Publish_Image_top", String)
    # img_sub_side = message_filters.Subscriber("Publish_Image_side", String)

    encoder_sub1 = message_filters.Subscriber("/Current_Position_USB1", Int32)

    # img_ts_top_sub = message_filters.Subscriber("img_ts_top", String)
    # img_ts_side_sub = message_filters.Subscriber("img_ts_side", String)
    
    ts = message_filters.ApproximateTimeSynchronizer([encoder_sub1], 10,  slop=0.05, allow_headerless=True)
    # ts = message_filters.ApproximateTimeSynchronizer([img_sub 0, encoder_sub0, encoder_sub1], 10,  slop=0.05, allow_headerless=True)
    ts.registerCallback(callback_main)
    
    if len(sys.argv) > 4:
        cycle = int(sys.argv[1]) 
        max_range = int(sys.argv[2]) 
        step = int(sys.argv[3]) 
        freq = int(sys.argv[4])
        waypoints = gen_waypoints(cycle, max_range, step)
    else:
        max_range = int(sys.argv[1]) 
        pts = int(sys.argv[2]) 
        freq = int(sys.argv[3])
        waypoints = gen_sin_waypoints(max_range, freq, pts)
    
    rate = rospy.Rate(freq) 
    # loop_num = 0
    # cycle_num = 1
    # pos = 0
    # step = 100
    # range_max = 2000

    # waypoints = gen_waypoints(cycle_num, range_max, step)
    # zeros = [0] * 100
    # print(waypoints)
    
    while not rospy.is_shutdown():
        if pos_i > len(waypoints) - 1:
            break
    
        position_request = Int32()
        pos = int(waypoints[pos_i]) # change to zeros if needed
        position_request.data = pos
        pos_i += 1
        
        # rospy.loginfo(f"--------------Publishing Request_Position: {position_request.data}-------------")
        # pub0.publish(position_request)
        pub1.publish(position_request)
        rate.sleep()
        

if __name__ == '__main__':
    # encoder_list0= []
    encoder_list1= []
    # img_list_top = []
    # img_list_side = []
    # img_ts_list_top = []
    # img_ts_list_side = []
    encoder_ts_list = []
    pos_i = 0
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    df = pd.DataFrame({"Enc TS": encoder_ts_list, "Encoder counts": encoder_list1})
    
    # df = pd.DataFrame({"TS": time_stamps, "Img name": img_list0, "Encoder counts 0": encoder_list0, "Encoder counts 1": encoder_list1})
    df.to_csv('data_output.csv', index=False)
    print("DATA COLLECTION DONE")
    
    
    