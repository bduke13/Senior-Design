import pycreate2
import time
from create2_port_definition import port_path

# Initialize the Create2. Replace '/dev/ttyUSB0' with your serial port
port = port_path
baud = {
    'default': 115200,
    'alt': 19200  # shouldn't need this unless you accidentally set it to this
}
bot = pycreate2.Create2(port=port, baud=baud['default'])

# Start the robot
bot.start()
bot.safe()


# Request Dirt Detect sensor data
# The sensor ID for Dirt Detect can be found in the pycreate2 documentation or Open Interface Spec
sensor_state = bot.get_sensors()
dirt_detect = sensor_state.dirt_detect

print("Dirt Detect Sensor Value:", dirt_detect)

# Stop the robot
bot.stop()
