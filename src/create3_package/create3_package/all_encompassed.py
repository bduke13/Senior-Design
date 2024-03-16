import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from irobot_create_msgs.msg import HazardDetectionVector, IrIntensityVector
from nav_msgs.msg import Odometry
import threading
import time

class CombinedSensorSubscriber(Node):
    def __init__(self):
        super().__init__('combined_sensor_subscriber')
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.lock = threading.Lock()

        self.scan_data = None
        self.ground_reward = None
        self.hazard_detections = None
        self.bump_detection = False
        self.odom_position = None
        self.odom_orientation = None

        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.create_subscription(IrIntensityVector, '/cliff_intensity', self.cliff_intensity_callback, qos_profile)
        self.create_subscription(HazardDetectionVector, '/hazard_detection', self.hazard_detection_callback, qos_profile)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile)

    def scan_callback(self, msg):
        with self.lock:
            self.scan_data = msg

    def cliff_intensity_callback(self, msg):
        with self.lock:
            self.ground_reward = any(reading.value > 3000 for reading in msg.readings)
            
    def hazard_detection_callback(self, msg):
        with self.lock:
            self.hazard_detections = msg
            self.bump_detection = any("bump" in detection.header.frame_id.lower() for detection in msg.detections)

    def odom_callback(self, msg):
        with self.lock:
            self.odom_position = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
            self.odom_orientation = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

def run_node(node):
    rclpy.spin(node)

def print_sensor_data(node):
    while True:
        with node.lock:
            if node.scan_data:
                print(f"Scan data available.")
            else:
                print("Scan data: No data received yet.")

            if node.ground_reward:
                print("Ground Reward: Detected!")
            else:
                print("Ground Reward: No reward detected.")

            if node.bump_detection:
                print("Bump detection: Detected!")
            else:
                print("Bump detection: No bump detected.")

            if node.odom_position and node.odom_orientation:
                print(f"Odom position: {node.odom_position}")
                print(f"Odom orientation: {node.odom_orientation}")
            else:
                print("Odom: No data received yet.")
        print() 
        
        time.sleep(0.1)  # Adjust this sleep time as needed

def main():
    rclpy.init()
    node = CombinedSensorSubscriber()
    node_thread = threading.Thread(target=run_node, args=(node,), daemon=True)
    node_thread.start()

    print_thread = threading.Thread(target=print_sensor_data, args=(node,), daemon=True)
    print_thread.start()

    try:
        node_thread.join()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
