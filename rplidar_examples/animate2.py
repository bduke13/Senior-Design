#!/usr/bin/env python3
import os
import threading
import pygame
from math import cos, sin, pi, floor
from rplidar import RPLidar

# Pygame setup
os.putenv('SDL_FBDEV', '/dev/fb1')
pygame.init()
lcd = pygame.display.set_mode((320, 240))
pygame.mouse.set_visible(False)
lcd.fill((0, 0, 0))
pygame.display.update()

PORT_NAME = '/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0'
max_distance = 0

def process_data(data):
    global max_distance, lcd
    lcd.fill((0, 0, 0))
    for angle in range(360):
        distance = data[angle]
        if distance > 0:  # Ignore initially ungathered data points
            max_distance = max(min(5000, distance), max_distance)
            radians = angle * pi / 180.0
            x = distance * cos(radians)
            y = distance * sin(radians)
            point = (160 + int(x / max_distance * 119), 120 + int(y / max_distance * 119))
            lcd.set_at(point, pygame.Color(255, 255, 255))
    pygame.display.update()

class LidarThread:
    def __init__(self, port_name):
        self.lidar = RPLidar(port_name)
        self.last_scan = [0] * 360
        self.running = False
        self.thread = threading.Thread(target=self.update_scan, daemon=True)

    def start(self):
        self.running = True
        self.lidar.start_motor()
        self.thread.start()

    def update_scan(self):
        try:
            print("Starting to read from RPLIDAR in Express Scan mode...")
            for scan in self.lidar.iter_scans(scan_type='express', max_buf_meas=0):
                for (_, angle, distance) in scan:
                    self.last_scan[min(359, floor(angle))] = distance
                if not self.running:
                    break
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.lidar.stop()
        self.lidar.disconnect()
        print("RPLIDAR disconnected.")

    def get_last_scan(self):
        return self.last_scan

if __name__ == '__main__':
    lidar_thread = LidarThread(PORT_NAME)
    lidar_thread.start()

    try:
        while True:
            last_scan = lidar_thread.get_last_scan()
            if last_scan:
                process_data(last_scan)
    except KeyboardInterrupt:
        print("Program interrupted by user, stopping.")
    finally:
        lidar_thread.stop()
