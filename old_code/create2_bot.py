# create2_bot.py
import smbus
from time import sleep, time
from math import pi
from pycreate2 import Create2
import struct
from lidar_thread import LidarThread
import numpy as np  # Import numpy for calculating standard deviation
from tqdm import tqdm

# MPU6050 Registers and their Address
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
INT_ENABLE = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H = 0x43
GYRO_YOUT_H = 0x45
GYRO_ZOUT_H = 0x47

class MyCreate2(Create2):
    # Port names for the Create2 and RPLIDAR
    CREATE2_DEFAULT_PORT = '/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DN026CMI-if00-port0'
    RPLIDAR_DEFAULT_PORT = '/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0'
    # Constants for the Create2
    WHEEL_DIAMETER = 0.072  # Wheel diameter in meters
    WHEEL_CIRCUMFERENCE = 3.14159 * WHEEL_DIAMETER  # Circumference of the wheel
    ENCODER_COUNTS_PER_REVOLUTION = 508.8  # Encoder counts per wheel revolution
    DISTANCE_BETWEEN_WHEELS = 0.235  # Distance between wheels in meters

    def __init__(self, create2_port=CREATE2_DEFAULT_PORT, rplidar_port=RPLIDAR_DEFAULT_PORT):
        super().__init__(create2_port)
        print(f'Robot starting')
        self.start()
        self.full()
        #self.bmm150_device = bmm150.BMM150()
        self.lidar_thread = LidarThread(rplidar_port)
        # Initialize sensor data storage
        self.data = self.get_sensors()

        # Initialize the MPU6050 sensor
        #self.mpu_bus = smbus.SMBus(1)
        #self.mpu_device_address = 0x68  # MPU6050 device address
        #self.MPU_Init()
        #self.gyro_offsets = self.calibrate_sensors()

        # Initialize variables for gyroscope data
        self.total_rotation_gyro_deg = 0.0  # Total rotation around the x-axis
        self.last_time = time()
        self.last_sensor_request_time = 0

        # Initialize variables for encoder data
        self.encoder_left_offset = 0
        self.encoder_right_offset = 0
        self.encoder_left_initial, self.encoder_right_initial = self.get_encoder_counts()


    def shutdown(self):
        """Shuts down the robot and cleans up resources."""
        print("Shutting down robot and cleaning up resources.")
        self.lidar_thread.stop()
        self.drive_stop()
        #self.stop()
        self.close()

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
        current_time = time()
        if current_time - self.last_sensor_request_time < 0.02:  # 20 ms
            return  # Do not request sensor data if less than 20 ms has passed
        self.last_sensor_request_time = current_time

        self.data = self.get_sensors()
        
        # Get current time for gyroscope integration
        dt = current_time - self.last_time
        self.last_time = current_time

        # Gyroscope data for orientation change
        #gyro_x = (self.read_raw_data() - self.gyro_offsets['x'])
        #self.total_rotation_gyro_deg += gyro_x * dt


    def get_lidar_data(self):
        """
        Retrieves the last scan data from the LIDAR thread.
        """
        return self.lidar_thread.get_last_scan()

    #MPU CODE
    def MPU_Init(self):
        self.mpu_bus.write_byte_data(self.mpu_device_address, SMPLRT_DIV, 7)
        self.mpu_bus.write_byte_data(self.mpu_device_address, CONFIG, 0)
        self.mpu_bus.write_byte_data(self.mpu_device_address, GYRO_CONFIG, 24)
        self.mpu_bus.write_byte_data(self.mpu_device_address, PWR_MGMT_1, 1)
        self.mpu_bus.write_byte_data(self.mpu_device_address, INT_ENABLE, 1)

    def read_raw_data(self):
        addr = GYRO_XOUT_H
        high = self.mpu_bus.read_byte_data(self.mpu_device_address, addr)
        low = self.mpu_bus.read_byte_data(self.mpu_device_address, addr + 1)
        value = ((high << 8) | low)
        if value > 32768:
            value -= 65536
        return value
 
    def calibrate_sensors(self, samples=1500):
        print("Calibrating sensors, keep the device still...")
        gyro_offsets = {'x': 0}
        gyro_values = {'x': []}  # Store all values for standard deviation calculation
        pbar = tqdm(total=samples, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')  # Initialize progress bar
        for i in range(samples):
            raw_data = self.read_raw_data()
            gyro_values['x'].append(raw_data)  # Store the value
            gyro_offsets['x'] = sum(gyro_values['x']) / (i+1)  # Calculate running average
            std_dev = np.std(gyro_values['x'])  # Calculate running standard deviation
            pbar.set_description(f"Calibrating (Average: {gyro_offsets['x']:.2f}, Std Dev: {std_dev:.2f})")  # Update progress bar description with average and standard deviation
            pbar.update(1)  # Update progress bar
            #print(raw_data)
            sleep(0.01)
        pbar.close()  # Close progress bar when calibration is done
        return gyro_offsets

    #ENCODER CODE
    def get_encoder_counts(self):
        """
        Fetches the current encoder counts from the robot, relative to the last "reset".
        Handles rollover of the encoder counts.
        """
        data = self.get_sensors()
        # print(f'Encoder counts: {data.encoder_counts_left}, {data.encoder_counts_right}')
        encoder_counts_left = data.encoder_counts_left - self.encoder_left_offset
        encoder_counts_right = data.encoder_counts_right - self.encoder_right_offset

        # Handle rollover
        if encoder_counts_left < -32768:
            encoder_counts_left += 65535
        elif encoder_counts_left > 32767:
            encoder_counts_left -= 65536

        if encoder_counts_right < -32768:
            encoder_counts_right += 65535
        elif encoder_counts_right > 32767:
            encoder_counts_right -= 65536

        return encoder_counts_left, encoder_counts_right

    def get_total_rotation(self):
        """
        Calculates the total rotational displacement from the starting encoder states.
        """
        # Get new encoder counts
        encoder_left_new, encoder_right_new = self.get_encoder_counts()

        # Calculate orientation change from encoders
        orientation_change_encoders = self.get_orientation_change_from_encoders(self.encoder_left_initial, self.encoder_right_initial, encoder_left_new, encoder_right_new)
        # Change to be withinn 360 degrees
        # Update initial encoders for the next calculation
        #self.encoder_left_initial, self.encoder_right_initial = encoder_left_new, encoder_right_new

        return orientation_change_encoders

    def get_orientation_change_from_encoders(self, encoder_left_initial, encoder_right_initial, encoder_left_new, encoder_right_new):
        distance_left = self.calculate_distance_traveled(encoder_left_new - encoder_left_initial)
        distance_right = self.calculate_distance_traveled(encoder_right_new - encoder_right_initial)
        rotation = (distance_right - distance_left) / self.DISTANCE_BETWEEN_WHEELS
        return rotation * 180 / pi

    def calculate_distance_traveled(self, encoder_counts):
        # Calculate the number of revolutions
        revolutions = encoder_counts / self.ENCODER_COUNTS_PER_REVOLUTION

        # Calculate distance traveled in meters
        distance_traveled = revolutions * self.WHEEL_CIRCUMFERENCE
        return distance_traveled 

