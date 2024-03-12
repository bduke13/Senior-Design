import time
# Assuming the MyCreate2 class is defined in a separate file named 'my_create2.py'
from create2_bot import MyCreate2
from rplidar import RPLidar


PORT_NAME = '/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0'

lidar = RPLidar(PORT_NAME)


def run_vacuum():
    # Initialize the robot; assuming the default port is correct or has been set appropriately
    robot = MyCreate2()
    
    # Turn on the vacuum
    robot.control_vacuum(True)
    print("Vacuum is now on.")
    
    # Wait for 2 seconds
    time.sleep(0.5)
    print(robot.get_sensors())
    # Turn off the vacuum
    robot.control_vacuum(False)
    print("Vacuum is now off.")

if __name__ == "__main__":
    run_vacuum()
