#!/usr/bin/env python3
import threading
from math import cos, sin, pi, floor
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from rplidar import RPLidar

# Set the appropriate port name for your LIDAR unit
PORT_NAME = '/dev/ttyUSB0'
DMAX = 4000  # Max display distance for the plot

class LidarThread:
    def __init__(self, port_name):
        self.lidar = RPLidar(port_name)
        self.last_scan = [0] * 360
        self.running = False
        self.thread = threading.Thread(target=self.update_scan, daemon=True)

    def start(self):
        self.running = True
        self.lidar.start_motor()
        self.thread.start()

    def update_scan(self):
        try:
            print("Starting to read from RPLIDAR in Express Scan mode...")
            for scan in self.lidar.iter_scans(max_buf_meas=0):
                for (_, angle, distance) in scan:
                    self.last_scan[min(359, floor(angle))] = distance
                if not self.running:
                    break
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.lidar.stop()
        self.lidar.disconnect()
        print("RPLIDAR disconnected.")

    def get_last_scan(self):
        return self.last_scan


def update_line(num, lidar_thread, scatter):
    scan_data = lidar_thread.get_last_scan()
    angles = np.radians(np.arange(360))
    distances = np.array(scan_data)
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    scatter.set_offsets(np.c_[x, y])  # Update the positions of the dots
    return scatter,

def run():
    lidar_thread = LidarThread(PORT_NAME)
    lidar_thread.start()

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.set_rmax(DMAX)
    ax.grid(True)

    # Initialize a scatter plot instead of a line plot
    scatter = ax.scatter([], [], s=10, color='b', alpha=0.7)

    ani = animation.FuncAnimation(fig, update_line, fargs=(lidar_thread, scatter), interval=50, cache_frame_data=False)

    plt.show()

    lidar_thread.stop()

if __name__ == '__main__':
    run()
