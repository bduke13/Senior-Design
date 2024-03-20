import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from irobot_create_msgs.msg import HazardDetectionVector, IrIntensityVector
from irobot_create_msgs.action import RotateAngle
from irobot_create_msgs.srv import ResetPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
import threading
import time
import asyncio 
import math

class CombinedSensorSubscriber(Node):
    def __init__(self):
        super().__init__('combined_sensor_subscriber')
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.lock = threading.Lock()

        self.scan_data = None
        self.ground_reward = None
        self.bump_detection = False
        self.odom_position = None
        self.odom_orientation = None
        self.odom_heading_deg = None

        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.create_subscription(IrIntensityVector, '/cliff_intensity', self.cliff_intensity_callback, qos_profile)
        self.create_subscription(HazardDetectionVector, '/hazard_detection', self.hazard_detection_callback, qos_profile)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile)

        # Initialize action client
        #self.action_client = ActionClient(self, RotateAngle, '/rotate_angle')

    def scan_callback(self, msg):
        with self.lock:
            self.scan_data = msg

    def cliff_intensity_callback(self, msg):
        with self.lock:
            self.ground_reward = any(reading.value > 3000 for reading in msg.readings)
            
    def hazard_detection_callback(self, msg):
        with self.lock:
            self.bump_detection = any("bump" in detection.header.frame_id.lower() for detection in msg.detections)

    def quaternion_to_yaw(self, w, x, y, z):
        """
        Convert a quaternion into yaw (heading) angle in radians.
        Parameters:
            w, x, y, z: Quaternion components
        Returns:
            Yaw angle in radians.
        """
        yaw = math.atan2(2.0 * (y * z + w * x), w * w - x * x - y * y + z * z)

        return yaw

    def odom_callback(self, msg):
        global global_odom_pose_position, global_odom_pose_orientation, global_odom_heading
        # Update the global variables with the latest odom pose position and orientation
        global_odom_pose_position = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
        global_odom_pose_orientation = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        # Calculate the heading (yaw) from the quaternion in radians and convert it to degrees
        w, x, y, z = global_odom_pose_orientation
        yaw_radians = self.quaternion_to_yaw(w, x, y, z)
        yaw_degrees = math.degrees(yaw_radians)

        # Normalize the yaw degrees to be within 0 to 360 degrees
        yaw_degrees_normalized = yaw_degrees % 360

        # Apply the flipping logic
        if 0 < yaw_degrees_normalized < 180:
            flipped_yaw_degrees = 360 - yaw_degrees_normalized
        elif 180 < yaw_degrees_normalized < 360:
            flipped_yaw_degrees = 360 - yaw_degrees_normalized
        else:
            flipped_yaw_degrees = yaw_degrees_normalized  # Keeps 0 and 180 degrees unchanged

        self.odom_heading_deg = flipped_yaw_degrees

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
    
    def get_odom_heading(self):
        with self.lock:
            return self.odom_heading_deg

def run_node(node):
    rclpy.spin(node)

def main():
    rclpy.init()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    node = CombinedSensorSubscriber()
    node_thread = threading.Thread(target=run_node, args=(node,), daemon=True)
    node_thread.start()
    print('spun up and ready')

    while True:
        print(node.get_odom_heading())
        time.sleep(0.05)

if __name__ == '__main__':
    main()
