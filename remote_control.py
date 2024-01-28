import curses
from pycreate2 import Create2
import struct
import time
from create2_port_definition import port_path

# Function to control the vacuum
def control_vacuum(vacuum_state, bot_instance):
    motor_byte = 13 if vacuum_state else 0
    data = struct.unpack('B', struct.pack('B', motor_byte))
    bot_instance.SCI.write(138, data)

# Main function to control the robot using curses
def main(stdscr):
    # Initialize the Create2 robot
    port = port_path
    bot = Create2(port)
    bot.start()
    bot.safe()

    drive_speed = 300
    vacuum_on = False
    # Turn off cursor visibility
    curses.curs_set(0)
    stdscr.nodelay(True)  # Don't block I/O calls

    while True:
        try:
            key = stdscr.getch()
            stdscr.clear()
            if key == ord('w'):
                bot.drive_direct(drive_speed, drive_speed)
                stdscr.addstr("Moving forward")
            elif key == ord('s'):
                bot.drive_direct(-drive_speed, -drive_speed)
                stdscr.addstr("Moving backward")
            elif key == ord('d'):  # Switched 'd' and 'a' functionality
                bot.drive_direct(-drive_speed, drive_speed)
                stdscr.addstr("Turning left")
            elif key == ord('a'):
                bot.drive_direct(drive_speed, -drive_speed)
                stdscr.addstr("Turning right")
            elif key == ord('v'):  # 'v' to toggle the vacuum
                vacuum_on = not vacuum_on
                control_vacuum(vacuum_on, bot)
                stdscr.addstr("Toggling vacuum")
            elif key == ord(' '):  # Space bar to stop the robot
                bot.drive_direct(0, 0)
                stdscr.addstr("Stopping robot")
            elif key == ord('i'):
                drive_speed += 20
                stdscr.addstr("Increasing speed")
            elif key == ord('u'):
                drive_speed -= 20
                stdscr.addstr("Decreasing speed")
            elif key == curses.KEY_ESCAPE or key == ord('q'):
                break  # Exit the loop if ESC or 'q' is pressed

            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        except Exception as e:
            stdscr.addstr(0, 0, str(e))

    # Optional: Close the connection
    # bot.close()

# Run the curses application
curses.wrapper(main)
