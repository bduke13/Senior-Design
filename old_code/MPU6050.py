# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

# import time
# import board
# import adafruit_mpu6050

# i2c = board.I2C()  # uses board.SCL and board.SDA
# # i2c = board.STEMMA_I2C()  # For using the built-in STEMMA QT connector on a microcontroller
# mpu = adafruit_mpu6050.MPU6050(i2c)

# while True:
#     print("Acceleration: X:%.4f, Y: %.4f, Z: %.4f m/s^2" % (mpu.acceleration))
#     print("Gyro X:%.4f, Y: %.4f, Z: %.4f rad/s" % (mpu.gyro))
#     print("Temperature: %.2f C" % mpu.temperature)
#     print("")
#     time.sleep(0.1)
import time
import board
import adafruit_mpu6050
from ahrs.filters import Madgwick

i2c = board.I2C()
mpu = adafruit_mpu6050.MPU6050(i2c)

# Create a Madgwick filter instance
filter = Madgwick()

# Initialize quaternion
quaternion = [1, 0, 0, 0]

while True:
    accel_data = mpu.acceleration
    gyro_data = mpu.gyro

    # Convert gyro data to radians/s
    gyro_data = [x * 0.0174533 for x in gyro_data]

    # Update the filter and capture the returned quaternion
    quaternion = filter.updateIMU(quaternion, gyr=gyro_data, acc=accel_data)

    print("Quaternion: ", quaternion)
    print("Euler angles: ", filter.to_euler())

    time.sleep(0.1)