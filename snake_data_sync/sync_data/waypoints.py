import sys
import numpy as np


def gen_waypoints(cycle, max_range, interval):
    wayspoints = list(range(0, max_range + 1, interval))
    for i in range(cycle - 1):
        wayspoints += list(range(max_range, -max_range - 1, -interval))
        wayspoints += list(range(-max_range, max_range + 1, interval))
    wayspoints += list(range(max_range, -max_range - 1, -interval))
    wayspoints += list(range(-max_range, 1, interval))
    return wayspoints

def generate_Input_Signal(q_max, f_signal, num_command, a = 0, q_offset = 0):
    f_command = 1
    f_signal = f_signal/100
    t = np.linspace(0, int(num_command / f_command), num_command)  #
    q_input = (q_max * np.exp(-f_signal * 0.1 * t)) * (np.sin(2 * np.pi * f_signal * t) + a) + q_offset
  
    return q_input




if __name__ == '__main__':
    if len(sys.argv) > 4:
        cycle = int(sys.argv[1]) 
        max_range = int(sys.argv[2]) 
        step = int(sys.argv[3]) 
        freq = int(sys.argv[4])
        print(freq)
        wp = gen_waypoints(cycle, max_range, step)
        print(wp)
    else:
        max_range = int(sys.argv[1]) 
        pts = int(sys.argv[2]) 
        freq = int(sys.argv[3])
        q = generate_Input_Signal(max_range, freq, pts)
        print(q)
