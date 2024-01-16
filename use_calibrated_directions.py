import smbus
import time
import math
import json

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


def read_magnetometer_data():
    x_out = read_word_2c(0) * scale
    y_out = read_word_2c(2) * scale
    z_out = read_word_2c(4) * scale
    return x_out, y_out, z_out

def calculate_heading(x_out, y_out, cardinal_data):
    # Calculate distances from each calibrated cardinal point
    distances = {}
    for direction in cardinal_data:
        dx = x_out - cardinal_data[direction]['x']
        dy = y_out - cardinal_data[direction]['y']
        distances[direction] = math.sqrt(dx**2 + dy**2)

    # Find the closest cardinal direction
    closest_direction = min(distances, key=distances.get)

    # Return the closest cardinal direction
    return closest_direction

# Load calibration data from JSON file
with open('cardinal_calibration_data.json', 'r') as infile:
    cardinal_calibration_data = json.load(infile)

write_byte(11, 0b00000001) # Reset
write_byte(9, 0b00000000|0b00000000|0b00001100|0b00000001) # Configuration
write_byte(10, 0b00100000)

scale = 0.92

for i in range(0, 500):
    x_out, y_out, z_out = read_magnetometer_data()

    heading = calculate_heading(x_out, y_out, cardinal_calibration_data)

    print("Heading:", heading)
    print("x:", x_out)
    print("y:", y_out)
    print("z:", z_out)
    time.sleep(0.1)
