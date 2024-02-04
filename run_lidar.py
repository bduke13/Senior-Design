import time
from lidar_thread import LidarThread

# Define the main function to run the script
def main():
    # Specify the serial port for the RPLIDAR
    rplidar_port = '/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0'
    
    # Initialize and start the LidarThread
    lidar_thread = LidarThread(rplidar_port)
    lidar_thread.start()
    
    print("LIDAR thread started. Printing last scan every second.")
    
    try:
        # Main loop to print the last scan every second
        while True:
            last_scan = lidar_thread.get_last_scan()
            if last_scan is not None:
                print("Last scan data:", last_scan)
            else:
                print("No scan data received yet.")
            
            # Wait for 1 second before the next print
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping LIDAR thread...")
        lidar_thread.stop()
        print("LIDAR thread stopped.")

if __name__ == "__main__":
    main()
