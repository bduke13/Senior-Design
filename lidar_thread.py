from rplidar import RPLidar
import threading

class LidarThread:
    def __init__(self, port_name):
        self.lidar = RPLidar(port_name)
        print(self.lidar.get_health())
        self.last_scan = None
        self.running = False
        self.thread = threading.Thread(target=self.update_scan, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()

    def update_scan(self):
        try:
            print("Starting to read from RPLIDAR...")
            for scan in self.lidar.iter_scans(max_buf_meas=0):
                self.last_scan = scan
                if not self.running:
                    break
        except KeyboardInterrupt:
            print("Stopping RPLIDAR due to an interrupt...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.lidar.stop()
        self.lidar.disconnect()
        print("RPLIDAR disconnected.")

    def get_last_scan(self):
        return self.last_scan