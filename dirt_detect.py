import serial
import struct
from  pycreate2 import Create2
import time
from create2_port_definition import port_path
import struct

def control_vacuum(vacuum_on, the_bot):
    """
    Control the vacuum: vacuum_on is either True (turn on) or False (turn off)
    """
    # Bit 1 controls the vacuum
    motor_byte = 0b000001101 if vacuum_on else 0
    # Pack the motor_byte into a byte format
    data = struct.unpack('B', struct.pack('B', motor_byte))
    the_bot.SCI.write(138, data)

bot = Create2(port_path)

# Start the Create 2
bot.start()

# Put the Create2 into 'safe' mode so we can drive it
# This will still provide some protection
bot.safe()

# You are responsible for handling issues, no protection/safety in
# this mode ... becareful
bot.full()



# Turn on the vacuum
#control_vacuum(True, bot)


for i in range(1000):
    print(bot.get_sensors().dirt_detect)

# Turn off the vacuum
#control_vacuum(False, bot)