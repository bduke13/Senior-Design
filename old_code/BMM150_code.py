import bmm150
import math
import time

device = bmm150.BMM150()  # Bus number will default to 1

# EMA parameters
alpha = 2 / (3 + 1)  # smoothing factor, for 3 time steps
heading_ema = 0  # initial EMA value

while True:
    x, y, z = device.read_mag_data()

    heading_rads = math.atan2(x, y)
    heading_degrees = math.degrees(heading_rads)

    # Update EMA
    heading_ema = alpha * heading_degrees + (1 - alpha) * heading_ema

    print(f"X : {x:.2f}µT")
    print(f"Y : {y:.2f}µT")
    print(f"Z : {z:.2f}µT")

    print(f"Heading: {heading_degrees:.2f}°")
    print(f"EMA Heading: {heading_ema:.2f}°")
    time.sleep(0.05)