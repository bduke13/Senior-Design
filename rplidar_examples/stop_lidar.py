#!/usr/bin/env python3
'''Records scans to a given file in the form of numpy array.
Usage example:

$ ./record_scans.py out.npy'''
import sys
import numpy as np
from rplidar import RPLidar
from rplidar_port_description import port_name


PORT_NAME = port_name

lidar = RPLidar(PORT_NAME)
try:
    lidar.start_motor()
    # Your scanning operation here
finally:
    lidar.stop_motor()
    lidar.stop()
    lidar.disconnect()
