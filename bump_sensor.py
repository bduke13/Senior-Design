#!/usr/bin/env python3
#-*-coding:utf-8-*-
##############################################
import time
from pycreate2 import Create2
from create2_port_definition import port_path
# Initialize the robot


bot = Create2(port_path)

# Start the robot
bot.start()
bot.safe()

print("Monitoring bump sensors. Press Ctrl+C to terminate.")

try:
    while True:
        # Get sensor data
        sensors = bot.get_sensors()

        # Check the bump sensors
        # sensors.bumps_wheeldrops will have the current bump sensor status
        if sensors.bumps_wheeldrops.bump_left or sensors.bumps_wheeldrops.bump_right:
            print("Bump detected!")
            # Add additional handling here if needed

        # Delay before next read
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping.")

finally:
    # Ensure the robot is stopped before we quit
    bot.drive_stop()
