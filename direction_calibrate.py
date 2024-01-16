import smbus
import time
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


def calibrate_cardinal_directions():
    directions = ['North', 'East', 'South', 'West']
    calibration_values = {}

    for direction in directions:
        input(f"Point the sensor to the {direction} (use your iPhone for accuracy) and press Enter to continue...")
        x_out, y_out, z_out = read_magnetometer_data()
        calibration_values[direction] = {'x': x_out, 'y': y_out, 'z': z_out}
        time.sleep(1)

    return calibration_values


def read_magnetometer_data():
    x_out = read_word_2c(0) * scale
    y_out = read_word_2c(2) * scale
    z_out = read_word_2c(4) * scale
    return x_out, y_out, z_out



write_byte(11, 0b00000001) # Reset
write_byte(9, 0b00000000|0b00000000|0b00001100|0b00000001) # Configuration
write_byte(10, 0b00100000)

scale = 0.92

# Calibrate cardinal directions
cardinal_calibration = calibrate_cardinal_directions()

# Save cardinal direction calibration data to JSON file
with open('cardinal_calibration_data.json', 'w') as outfile:
    json.dump(cardinal_calibration, outfile)

print("Cardinal direction calibration complete. Data saved to 'cardinal_calibration_data.json'")
