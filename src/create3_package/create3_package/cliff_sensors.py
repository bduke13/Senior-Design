import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from irobot_create_msgs.msg import IrIntensityVector  # Import the message type
import time
import threading

# Define a global variable to hold the cliff sensor readings
global_sensor_readings = None

# Define a global variable to hold the cliff detection status
global_cliff_detection = False

class CliffIntensitySubscriber(Node):
    def __init__(self):
        super().__init__('cliff_intensity_subscriber')
        # Define a QoS profile with a Reliability policy compatible with the publisher
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.subscription = self.create_subscription(
            IrIntensityVector,
            '/cliff_intensity',
            self.listener_callback,
            qos_profile)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        global global_sensor_readings, global_cliff_detection
        # Update the global variable with the latest readings
        global_sensor_readings = [(reading.header.frame_id, reading.value) for reading in msg.readings]

        # Check if any cliff sensor reading is over 3000
        global_cliff_detection = any(value > 3000 for _, value in global_sensor_readings)

def main(args=None):
    rclpy.init(args=args)
    subscriber = CliffIntensitySubscriber()

    # Spin in a separate thread or use a non-blocking spin_once alternative
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(subscriber)

    # Use a separate thread for the executor
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        while True:
            print(global_sensor_readings)
            if global_cliff_detection:
                print("Cliff detection detected!")
            else:
                print("No cliff detection.")
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