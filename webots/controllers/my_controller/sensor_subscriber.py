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

        self.scan_data = None # 720 long list of distances
        self.ground_reward = None # boolean whether we see tinfoil
        self.bump_detection = False # boolean whether we have any bumper in the hazard detections
        self.odom_position = None # position of robot
        self.odom_orientation = None # angle of robot

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
            self.bump_detection = any("bump" in detection.header.frame_id.lower() for detection in msg.detections)

    def odom_callback(self, msg):
        with self.lock:
            self.odom_position = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
            self.odom_orientation = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

    # Getter for scan_data
    def get_scan_data(self):
        with self.lock:
            return self.scan_data

    # Getter for ground_reward
    def get_ground_reward(self):
        with self.lock:
            return self.ground_reward

    # Getter for bump_detection
    def get_bump_detection(self):
        with self.lock:
            return self.bump_detection

    # Getter for odom_position
    def get_odom_position(self):
        with self.lock:
            return self.odom_position

    # Getter for odom_orientation
    def get_odom_orientation(self):
        with self.lock:
            return self.odom_orientation

def run_node(node):
    rclpy.spin(node)

def main():
    rclpy.init()
    node = CombinedSensorSubscriber()
    node_thread = threading.Thread(target=run_node, args=(node,), daemon=True)
    node_thread.start()

    while True:
        print(node.get_bump_detection())
        time.sleep(0.01)



if __name__ == '__main__':
    main()