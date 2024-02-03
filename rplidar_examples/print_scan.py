#!/usr/bin/env python3
'''Prints out data from RPLIDAR continuously.'''
from rplidar import RPLidar
from rplidar_port_description import port_name

PORT_NAME = port_name

def print_lidar_data():
    lidar = RPLidar(PORT_NAME)
    
    try:
        print("Starting to read from RPLIDAR...")
        for scan in lidar.iter_scans():
            print(scan)
    except KeyboardInterrupt:
        print("Stopping RPLIDAR...")
    finally:
        lidar.stop()
        lidar.disconnect()
        print("RPLIDAR disconnected.")

if __name__ == '__main__':
    print_lidar_data()
