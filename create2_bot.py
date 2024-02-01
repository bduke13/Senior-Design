from pycreate2 import Create2
import struct

class MyCreate2(Create2):
    DEFAULT_PORT_PATH = '/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DN026CMI-if00-port0'
    
    def __init__(self, port=DEFAULT_PORT_PATH):
        super().__init__(port)
        self.start()  # Start the robot
        self.safe()   # Set the robot to 'safe' mode
    
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
