from pycreate2 import Create2
import time
import struct
from create2_port_definition import port_path

def control_vacuum(vacuum_on, bot):
    """
    Control the vacuum: vacuum_on is either True (turn on) or False (turn off)
    """
    motor_byte = 13 if vacuum_on else 0
    data = struct.unpack('B', struct.pack('B', motor_byte))
    bot.SCI.write(138, data)

# Initialize the Create2
port = port_path
bot = Create2(port)

# Start the Create2 and put it into 'safe' mode
bot.start()
bot.safe()

# Turn on the vacuum
control_vacuum(True, bot)
start_time = time.time()
bot.drive_direct(100, 100)
vacuum_duration = 4  # Vacuum for 10 seconds

# Continuously check dirt sensor while the vacuum is on
while time.time() - start_time < vacuum_duration:
    sensor_state = bot.get_sensors()
    dirt_detect = sensor_state.dirt_detect
    print("Dirt Detect Sensor Value:", dirt_detect)
    time.sleep(0.05)  # Check every 0.5 seconds

# Turn off the vacuum
control_vacuum(False, bot)
bot.drive_direct(0, 0)
# Stop the bot
bot.drive_stop()

# Close the connection (optional)
# bot.close()
