from  pycreate2 import Create2
import time
from create2_port_definition import port_path
import struct

def drive_direct(r_vel, l_vel, the_bot):
    """
    Drive motors directly: [-500, 500] mm/sec
    """
    r_vel = the_bot.limit(r_vel, -500, 500)
    l_vel = the_bot.limit(l_vel, -500, 500)
    data = struct.unpack('4B', struct.pack('>2h', r_vel, l_vel))  # write do this?
    print(data)
    the_bot.SCI.write(145, data)

def control_vacuum(vacuum_on, the_bot):
    """
    Control the vacuum: vacuum_on is either True (turn on) or False (turn off)
    """
    # Bit 1 controls the vacuum
    motor_byte = 13 if vacuum_on else 0
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
control_vacuum(True, bot)
time.sleep(5)  # Vacuum on for 2 seconds

# Turn off the vacuum
control_vacuum(False, bot)




# Stop the bot
bot.drive_stop()

# query some sensors
sensors = bot.get_sensors()  # returns all data
print(sensors.light_bumper_left)

# Close the connection
#bot.close()