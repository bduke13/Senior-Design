import bmm150
import math
from pycreate2 import Create2
import struct

class MyCreate2(Create2):
    DEFAULT_PORT_PATH = '/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DN026CMI-if00-port0'
    
    def __init__(self, port=DEFAULT_PORT_PATH):
        super().__init__(port)
        self.start()  # Start the robot
        self.safe()   # Set the robot to 'safe' mode
        self.bmm150_device = bmm150.BMM150()  # Initialize the BMM150 device here
    
    def control_vacuum(self, vacuum_on):
        """
        Control the vacuum: vacuum_on is either True (turn on) or False (turn off).
        """
        motor_byte = 13 if vacuum_on else 0
        data = struct.unpack('B', struct.pack('B', motor_byte))
        self.SCI.write(138, data)
    
    def get_dirt_sensor(self):
        """
        Gets the current value of the dirt detect sensor.
        """
        sensor_state = self.get_sensors()
        dirt_detect = sensor_state.dirt_detect
        return dirt_detect

    def get_bump_sensors(self):
        """
        Gets the current values of the left and right bump sensors.
        """
        sensor_state = self.get_sensors()
        bump_left = sensor_state.bumps_wheeldrops.bump_left
        bump_right = sensor_state.bumps_wheeldrops.bump_right
        return bump_left, bump_right

    def read_heading(self):
        """
        Reads magnetic field data from the BMM150 sensor, calculates the heading in degrees.
        """
        x, y, z = self.bmm150_device.read_mag_data()
        heading_rads = math.atan2(x, y)
        heading_degrees = math.degrees(heading_rads)
        heading_degrees = heading_degrees if heading_degrees > 0 else heading_degrees + 360

        # Optionally print the magnetic field data and heading
        print(f"X : {x:.2f}µT")
        print(f"Y : {y:.2f}µT")
        print(f"Z : {z:.2f}µT")
        print(f"Heading: {heading_degrees:.2f}°")

        return heading_degrees
