from rplidar import RPLidar
import threading

class LidarThread:
    """
    Class that handles starting, stopping, and reading from the RPLIDAR serial device.

    This class runs in its own thread because the LiDAR prefers to run by continuously
    outputting data and this interferes with other program execution so we handle it by
    isolating this process to its own thread.
    """
    def __init__(self, port_name):
        """
        Initializes the lidar serial port object and starts a thread
        that will constantly update the last available LiDAR scan.
        """
        self.lidar = RPLidar(port_name)
        print('RPLIDAR thread starting')
        print(f'RPLIDAR health: {self.lidar.get_health()}')
        self.last_scan = None
        self.running = True
        self.lidar.start_motor()
        print(f'RPLIDAR up to speed: {self.lidar.motor_speed}')
        self.thread = threading.Thread(target=self.update_scan, daemon=True)
        self.thread.start()
    
    # Add an explicit stop method
    def stop(self):
        if self.running:
            self.running = False
            self.thread.join()  # Wait for the thread to finish
            self.lidar.stop_motor()
            self.lidar.stop()
            self.lidar.disconnect()
            print("RPLIDAR stopped and disconnected.")

    def __del__(self):
        self.stop()  # Ensure resources are cleaned up

    def update_scan(self):
        """
        Main method to read in from RPLIDAR data and set available scan.
        """
        try:
            print("Starting to read from RPLIDAR...")
            for scan in self.lidar.iter_scans(max_buf_meas=0):
                self.last_scan = scan
                if not self.running:
                    break
        except KeyboardInterrupt:
            print("Stopping RPLIDAR due to an interrupt...")
        finally:
            self.lidar.stop_motor()
            self.lidar.stop()
            self.lidar.disconnect()
            print("RPLIDAR stopped and disconnected.")

    def get_last_scan(self):
        """
        Access to the lidar data of this thread.

        Returns a list of scans with each measurement of format:
            (quality, angle, distance)
        """
        return self.last_scan