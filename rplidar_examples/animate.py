#!/usr/bin/env python3
'''Animates distances and measurement quality'''
from rplidar import RPLidar, RPLidarException
import matplotlib
matplotlib.use('Qt5Agg')  # Ensure this matches your environment
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.animation as animation
from rplidar_port_description import port_name

PORT_NAME = port_name
DMAX = 4000
IMIN = 0
IMAX = 50

def update_line(num, iterator, line, lidar):
    try:
        scan = next(iterator)
    except StopIteration:
        # Restart the iterator in case of StopIteration
        iterator = lidar.iter_scans(max_buf_meas=0)
        scan = next(iterator)
    except RPLidarException as e:
        print(f"RPLidar exception: {e}")
        return line,

    offsets = np.array([(np.radians(meas[1]), meas[2]) for meas in scan])
    line.set_offsets(offsets)
    intens = np.array([meas[0] for meas in scan])
    line.set_array(intens)
    return line,

def run():
    lidar = RPLidar(PORT_NAME)
    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    line = ax.scatter([0, 0], [0, 0], s=5, c=[IMIN, IMAX], cmap=plt.cm.Greys_r, lw=0)
    ax.set_rmax(DMAX)
    ax.grid(True)

    iterator = lidar.iter_scans(max_buf_meas=0, min_len=15)
    ani = animation.FuncAnimation(fig, update_line, fargs=(iterator, line, lidar), interval=50, cache_frame_data=False, blit=True)

    plt.show()
    lidar.stop()
    lidar.disconnect()

if __name__ == '__main__':
    run()