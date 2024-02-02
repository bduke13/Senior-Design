import curses
from create2_bot import MyCreate2  # Adjust this import as necessary to match your file structure
import time 

def main(stdscr):
    bot = MyCreate2()

    drive_speed = 300
    vacuum_on = False
    curses.curs_set(0)  # Turn off cursor visibility
    stdscr.nodelay(True)  # Don't block I/O calls

    # Display controls
    stdscr.addstr("Controls:\n")
    stdscr.addstr("'w' - Move forward\n")
    stdscr.addstr("'d' - Turn left\n")
    stdscr.addstr("'s' - Stop\n")
    stdscr.addstr("'a' - Turn right\n")
    stdscr.addstr("'x' - Move backward\n")
    stdscr.addstr("'v' - Toggle vacuum\n")
    stdscr.addstr("'t' - Increase speed\n")
    stdscr.addstr("'g' - Decrease speed\n")
    stdscr.addstr("'q' - Quit\n")
    stdscr.addstr("\nStatus:\n")

    while True:
        try:
            key = stdscr.getch()
            stdscr.clear()

            # Re-display controls
            stdscr.addstr("Controls:\n")
            stdscr.addstr("'w' - Move forward\n")
            stdscr.addstr("'d' - Turn left\n")
            stdscr.addstr("'s' - Stop\n")
            stdscr.addstr("'a' - Turn right\n")
            stdscr.addstr("'x' - Move backward\n")
            stdscr.addstr("'v' - Toggle vacuum\n")
            stdscr.addstr("'t' - Increase speed\n")
            stdscr.addstr("'g' - Decrease speed\n")
            stdscr.addstr("'q' - Quit\n")
            stdscr.addstr("\nStatus:\n")

            if key == ord('w'):
                bot.drive_direct(drive_speed, drive_speed)
                stdscr.addstr("Moving forward\n")
            elif key == ord('d'):
                bot.drive_direct(-drive_speed, drive_speed)
                stdscr.addstr("Turning left\n")
            elif key == ord('s'):
                bot.drive_direct(0, 0)
                stdscr.addstr("Stopping robot\n")
            elif key == ord('a'):
                bot.drive_direct(drive_speed, -drive_speed)
                stdscr.addstr("Turning right\n")
            elif key == ord('x'):
                bot.drive_direct(-drive_speed, -drive_speed)
                stdscr.addstr("Moving backward\n")
            elif key == ord('v'):
                vacuum_on = not vacuum_on
                bot.control_vacuum(vacuum_on)
                stdscr.addstr("Vacuum toggled\n")
            elif key == ord('t'):
                drive_speed += 20
                stdscr.addstr("Increasing speed\n")
            elif key == ord('g'):
                drive_speed -= 20
                stdscr.addstr("Decreasing speed\n")
            elif key == ord('q'):
                break  # Exit the loop

            # Display the sensor states and other information
            bump_left, bump_right = bot.get_bump_sensors()
            dirt_detect = bot.get_dirt_sensor()
            heading = bot.read_heading()  # Assuming this method exists in MyCreate2
            stdscr.addstr(f"Speed: {drive_speed}\n")
            stdscr.addstr(f"Bump Left: {bump_left}, Bump Right: {bump_right}\n")
            stdscr.addstr(f"Dirt Detect: {dirt_detect}\n")
            stdscr.addstr(f"Heading: {heading:.2f}°\n")

            time.sleep(0.1)  # Small delay
        except Exception as e:
            stdscr.addstr(0, 0, str(e))

# Run the curses application
curses.wrapper(main)
