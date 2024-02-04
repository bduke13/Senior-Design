#!/usr/bin/env python3
'''Prints out data from RPLIDAR continuously in a separate thread and allows retrieving the last scan.'''
from rplidar import RPLidar
from rplidar_port_description import port_name
import threading
import time

PORT_NAME = port_name

class LidarThread:
    def __init__(self, port_name):
        self.lidar = RPLidar(port_name)
        self.last_scan = None
        self.running = False
        self.thread = threading.Thread(target=self.update_scan, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()

    def update_scan(self):
        try:
            print("Starting to read from RPLIDAR...")
            for scan in self.lidar.iter_scans(max_buf_meas=0):
                self.last_scan = scan
                if not self.running:
                    break
        except KeyboardInterrupt:
            print("Stopping RPLIDAR due to an interrupt...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.lidar.stop()
        self.lidar.disconnect()
        print("RPLIDAR disconnected.")

    def get_last_scan(self):
        return self.last_scan

if __name__ == '__main__':
    lidar_thread = LidarThread(PORT_NAME)
    lidar_thread.start()

    # Main program loop 
    try:
        while True:
            # Do other things here
            time.sleep(1)  # Simulate work
            last_scan = lidar_thread.get_last_scan()
            if last_scan is not None:
                print("Latest scan data:", last_scan)  # Or process it as needed
    except KeyboardInterrupt:
        print("Program interrupted by user, stopping LIDAR thread...")
    finally:
        lidar_thread.stop()
