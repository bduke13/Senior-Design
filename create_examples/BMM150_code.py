import bmm150
import math
import time

device = bmm150.BMM150()  # Bus number will default to 1



while True:
    x, y, z = device.read_mag_data()

    heading_rads = math.atan2(x, y)

    heading_degrees = math.degrees(heading_rads)
    print(f"X : {x:.2f}µT")
    print(f"Y : {y:.2f}µT")
    print(f"Z : {z:.2f}µT")

    heading_degrees = heading_degrees if heading_degrees > 0 else heading_degrees + 360

    print(f"Heading: {heading_degrees:.2f}°")
    time.sleep(0.05)
