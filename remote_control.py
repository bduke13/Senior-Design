import curses
import threading
import pygame
from math import cos, sin, pi, floor
import os
from create2_bot import MyCreate2  # Adjust this import as necessary
import time
from rplidar import RPLidar

# Initialize Pygame for graphics display
os.putenv('SDL_FBDEV', '/dev/fb1')
pygame.init()
lcd = pygame.display.set_mode((320, 240))
pygame.mouse.set_visible(False)
max_distance = 0

def process_data(data):
    global max_distance
    lcd.fill((0, 0, 0))
    if data:
        for (quality, angle, distance) in data:
            if distance > 0:  # Ignore ungathered data points
                max_distance = max(min(5000, distance), max_distance)
                radians = angle * pi / 180.0
                x = distance * cos(radians)
                y = distance * sin(radians)
                point = (160 + int(x / max_distance * 119), 120 + int(y / max_distance * 119))
                lcd.set_at(point, pygame.Color(255, 255, 255))
    pygame.display.update()

def main(stdscr, bot):
    drive_speed = 5
    vacuum_on = False
    curses.curs_set(0)  # Turn off cursor visibility
    stdscr.nodelay(True)  # Don't block I/O calls

    # Display controls on terminal
    commands = ["'w' - Move forward", "'s' - Stop", "'a' - Turn left", "'d' - Turn right",
                "'x' - Move backward", "'v' - Toggle vacuum", "'t' - Increase speed",
                "'g' - Decrease speed", "'q' - Quit"]
    for command in commands:
        stdscr.addstr(command + "\n")
    stdscr.refresh()

    while True:
        try:
            # Clear previous sensor data lines
            stdscr.clear()
            # Redisplay the commands
            for command in commands:
                stdscr.addstr(command + "\n")
             # Display the sensor states and other information
            bump_left, bump_right = bot.get_bump_sensors()
            dirt_detect = bot.get_dirt_sensor()
            heading = bot.read_heading()  # Assuming this method exists in MyCreate2
            stdscr.addstr(f"Speed: {drive_speed}\n")
            stdscr.addstr(f"Bump Left: {bump_left}, Bump Right: {bump_right}\n")
            stdscr.addstr(f"Dirt Detect: {dirt_detect}\n")
            stdscr.addstr(f"Heading: {heading:.2f}°\n")

            # Existing code for handling key presses and LIDAR data processing
            key = stdscr.getch()
            if key == ord('w'):
                bot.drive_direct(drive_speed, drive_speed)
            elif key == ord('s'):
                bot.drive_direct(0, 0)
            elif key == ord('d'):
                bot.drive_direct(-drive_speed, drive_speed)
            elif key == ord('a'):
                bot.drive_direct(drive_speed, -drive_speed)
            elif key == ord('x'):
                bot.drive_direct(-drive_speed, -drive_speed)
            elif key == ord('v'):
                vacuum_on = not vacuum_on
                bot.control_vacuum(vacuum_on)
            elif key == ord('t'):
                drive_speed += 20
            elif key == ord('g'):
                drive_speed -= 20
            elif key == ord('q'):
                break  # Exit the loop

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False  # Or however you choose to handle a quit event

            # Process and display LIDAR data
            last_scan = bot.get_lidar_data()
            process_data(last_scan)


        except Exception as e:
            stdscr.addstr(0, 0, str(e))
            break

if __name__ == '__main__':
    try:
        bot = MyCreate2()  # Initialize robot with LIDAR
        curses.wrapper(main, bot)
    finally:
        bot.__del__()  # Ensure proper shutdown
        pygame.quit()  # Quit Pygame
