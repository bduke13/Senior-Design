import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from irobot_create_msgs.msg import HazardDetectionVector
import time
import threading

# Define a global variable to hold the hazard detection vector
global_hazard_detection_vector = None

# Define a global variable to hold the bump detection status
global_has_bump_detection = False

class HazardDetectionSubscriber(Node):
    def __init__(self):
        super().__init__('hazard_detection_subscriber')
        # Define a QoS profile with a Reliability policy compatible with the publisher
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.subscription = self.create_subscription(
            HazardDetectionVector,
            '/hazard_detection',
            self.listener_callback,
            qos_profile)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        global global_hazard_detection_vector, global_has_bump_detection
        # Update the global variable with the hazard detection vector
        global_hazard_detection_vector = msg

        # Check if any detection contains the substring "bump" in the frame_id field
        global_has_bump_detection = any(frame_id.lower().find("bump") != -1 for detection in msg.detections for frame_id in (detection.header.frame_id,))

def main(args=None):
    rclpy.init(args=args)
    subscriber = HazardDetectionSubscriber()

    # Spin in a separate thread or use a non-blocking spin_once alternative
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(subscriber)

    # Use a separate thread for the executor
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        while True:
            if global_has_bump_detection:
                print("Bump detection detected!")
            else:
                print("No bump detection.")
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