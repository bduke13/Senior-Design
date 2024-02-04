import bmm150
import math
from pycreate2 import Create2
import struct
from lidar_thread import LidarThread

class MyCreate2(Create2):
    CREATE2_DEFAULT_PORT = '/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DN026CMI-if00-port0'
    RPLIDAR_DEFAULT_PORT = '/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0'

    def __init__(self, create2_port=CREATE2_DEFAULT_PORT, rplidar_port=RPLIDAR_DEFAULT_PORT):
        super().__init__(create2_port)
        print(f'Robot starting')
        self.start()
        self.full()
        self.bmm150_device = bmm150.BMM150()
        self.lidar_thread = LidarThread(rplidar_port)
        # Initialize sensor data storage
        self.bump_sensors = {'left': False, 'right': False}
        self.dirt_detect = 0

    def shutdown(self):
        """Shuts down the robot and cleans up resources."""
        print("Shutting down robot and cleaning up resources.")
        self.lidar_thread.stop()
        self.drive_stop()
        self.stop()
        self.close()
        # Add any additional cleanup steps here...

    # Modify __del__ to call shutdown, or better yet, ensure shutdown is called explicitly
    def __del__(self):
        self.shutdown()
        

    def control_vacuum(self, vacuum_on):
        """
        Control the vacuum: vacuum_on is either True (turn on) or False (turn off).
        """
        motor_byte = 7 if vacuum_on else 0
        data = struct.unpack('B', struct.pack('B', motor_byte))
        self.SCI.write(138, data)
    
    def update_sensors(self):
        """
        Fetches the latest sensor data from the robot and updates the internal storage.
        """
        sensor_state = self.get_sensors()
        print(sensor_state)
        self.bump_sensors['left'] = sensor_state.bumps_wheeldrops.bump_left
        self.bump_sensors['right'] = sensor_state.bumps_wheeldrops.bump_right
        self.dirt_detect = sensor_state.dirt_detect

    def get_dirt_sensor(self):
        """
        Returns the current value of the dirt detect sensor from internal storage.
        """
        return self.dirt_detect

    def get_bump_sensors(self):
        """
        Returns the current values of the left and right bump sensors from internal storage.
        """
        return self.bump_sensors['left'], self.bump_sensors['right']

    def read_heading(self):
        """
        Reads magnetic field data from the BMM150 sensor, calculates the heading in degrees.
        """
        x, y, z = self.bmm150_device.read_mag_data()
        heading_rads = math.atan2(x, y)
        heading_degrees = math.degrees(heading_rads)
        heading_degrees = heading_degrees if heading_degrees > 0 else heading_degrees + 360
        return heading_degrees

    def get_lidar_data(self):
        """
        Retrieves the last scan data from the LIDAR thread.
        """
        return self.lidar_thread.get_last_scan()
