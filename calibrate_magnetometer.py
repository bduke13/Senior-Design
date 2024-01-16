import smbus
import time
import math

# Initialize the I2C bus
bus = smbus.SMBus(1)
address = 0x0d

def read_byte(adr):
    return bus.read_byte_data(address, adr)

def read_word(adr):
    low = bus.read_byte_data(address, adr)
    high = bus.read_byte_data(address, adr+1)
    val = (high << 8) + low
    return val

def read_word_2c(adr):
    val = read_word(adr)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

def write_byte(adr, value):
    bus.write_byte_data(address, adr, value)

def calibrate_magnetometer():
    print("Starting calibration...")
    x_max, y_max, z_max = -float('inf'), -float('inf'), -float('inf')
    x_min, y_min, z_min = float('inf'), float('inf'), float('inf')

    for i in range(300):  # Adjust the number of iterations as needed
        print(i)
        x_out, y_out, z_out = read_magnetometer_data()
        x_max, y_max, z_max = max(x_max, x_out), max(y_max, y_out), max(z_max, z_out)
        x_min, y_min, z_min = min(x_min, x_out), min(y_min, y_out), min(z_min, z_out)
        time.sleep(0.1)

    x_offset = (x_max + x_min) / 2
    y_offset = (y_max + y_min) / 2
    z_offset = (z_max + z_min) / 2
    print(x_offset, y_offset, z_offset)
    return x_offset, y_offset, z_offset

def read_magnetometer_data():
    x_out = read_word_2c(0) * scale
    y_out = read_word_2c(2) * scale
    z_out = read_word_2c(4) * scale
    return x_out, y_out, z_out

write_byte(11, 0b00000001) # Reset
write_byte(9, 0b00000000|0b00000000|0b00001100|0b00000001) # Configuration
write_byte(10, 0b00100000)

scale = 0.92

# Calibrate magnetometer
x_offset, y_offset, z_offset = calibrate_magnetometer()

for i in range(0, 500):
    x_out, y_out, z_out = read_magnetometer_data()

    # Apply offset correction
    x_out = (x_out - x_offset) * scale
    y_out = (y_out - y_offset) * scale
    z_out = (z_out - z_offset) * scale

    bearing = math.atan2(y_out, x_out)
    if bearing < 0:
        bearing += 2 * math.pi

    bearing = math.degrees(bearing)

    print("Bearing:", bearing)
    print("x:", x_out)
    print("y:", y_out)
    print("z:", z_out)
    time.sleep(0.1)

