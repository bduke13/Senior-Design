import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan  # Import the message type
import time
import threading

# Define a global variable to hold the scan message
global_scan_message = None

class ScanSubscriber(Node):
    def __init__(self):
        super().__init__('scan_subscriber')
        # Define a QoS profile with a Reliability policy compatible with the publisher
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            qos_profile)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        global global_scan_message
        # Update the global variable with the latest scan message
        global_scan_message = msg

def main(args=None):
    rclpy.init(args=args)
    subscriber = ScanSubscriber()

    # Spin in a separate thread or use a non-blocking spin_once alternative
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(subscriber)

    # Use a separate thread for the executor
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        while True:
            if global_scan_message is not None:
                print("Latest scan message:")
                # Assume `msg` is an instance of `sensor_msgs.msg.LaserScan`
                print(global_scan_message.ranges)
                #print(global_scan_message)
            else:
                print("No data received yet.")
            time.sleep(0.01)  # Adjust the sleep time as needed
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup and shutdown
        executor.shutdown()
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()