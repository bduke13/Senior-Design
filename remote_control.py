import curses
import struct
import time
from create2_bot import MyCreate2  # Assume this is your extended class file

# Note: The MyCreate2 class should already include methods for controlling the vacuum and reading sensors.

# Main function to control the robot using curses and display sensor states
def main(stdscr):
    # Initialize the MyCreate2 robot with the extended functionality
    bot = MyCreate2()

    drive_speed = 300
    vacuum_on = False
    curses.curs_set(0)  # Turn off cursor visibility
    stdscr.nodelay(True)  # Don't block I/O calls

    while True:
        try:
            key = stdscr.getch()
            stdscr.clear()
            if key == ord('w'):
                bot.drive_direct(drive_speed, drive_speed)
                stdscr.addstr("Moving forward\n")
            elif key == ord('s'):
                bot.drive_direct(-drive_speed, -drive_speed)
                stdscr.addstr("Moving backward\n")
            elif key == ord('d'):
                bot.drive_direct(-drive_speed, drive_speed)
                stdscr.addstr("Turning left\n")
            elif key == ord('a'):
                bot.drive_direct(drive_speed, -drive_speed)
                stdscr.addstr("Turning right\n")
            elif key == ord('v'):
                vacuum_on = not vacuum_on
                bot.control_vacuum(vacuum_on)
                stdscr.addstr("Toggling vacuum\n")
            elif key == ord(' '):
                bot.drive_direct(0, 0)
                stdscr.addstr("Stopping robot\n")
            elif key == ord('i'):
                drive_speed += 20
                stdscr.addstr("Increasing speed\n")
            elif key == ord('u'):
                drive_speed -= 20
                stdscr.addstr("Decreasing speed\n")
            elif key == 27: #escape
                break  # Exit the loop

            # Display the sensor states
            bump_left, bump_right = bot.get_bump_sensors()
            dirt_detect = bot.get_dirt_sensor()
            stdscr.addstr(f"Bump Left: {bump_left}, Bump Right: {bump_right}, Dirt Detect: {dirt_detect}\n")

            time.sleep(0.1)  # Small delay
        except Exception as e:
            stdscr.addstr(0, 0, str(e))

# Ensure the my_create2.py file is correctly placed and contains the MyCreate2 class with all necessary methods.
# Run the curses application
curses.wrapper(main)
