import time
from create2_bot import MyCreate2

def turn_to_angle(bot, target_angle):
    tolerance = 0.5  # degrees

    while True:
        bot.update_sensors()
        current_angle = bot.get_total_rotation()  # Use encoder data
        distance_to_target = abs(current_angle - target_angle)
        if distance_to_target <= tolerance:
            bot.drive_direct(0, 0)  # Stop the robot
            break
        else:
            # Set speed as distance to target + 50
            speed = -90
            if current_angle < target_angle:
                bot.drive_direct(-speed, speed)  # Turn right
            else:
                bot.drive_direct(speed, -speed)  # Turn left
        print(bot.get_total_rotation())  # Print encoder data
        time.sleep(0.01)

def main():
    bot = MyCreate2()
    bot.reset()
    time.sleep(10)
    try:
        for angle in [360, 0, 360, 720, 1080, 720, 360, 0, -360, -720, -1080, -720, 0]:
            print(f"Turning to {angle} degrees")
            turn_to_angle(bot, angle)
            time.sleep(2)
    finally:
        bot.shutdown()

if __name__ == "__main__":
    main()