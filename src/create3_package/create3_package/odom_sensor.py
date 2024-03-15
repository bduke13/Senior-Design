import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry  # Import the Odometry message type
import time
import threading

# Define global variables to hold the odom pose
global_odom_pose_position = None
global_odom_pose_orientation = None

class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        # Define a QoS profile with a Reliability policy compatible with the publisher
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.listener_callback,
            qos_profile)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        global global_odom_pose_position, global_odom_pose_orientation
        # Update the global variables with the latest odom pose position and orientation
        global_odom_pose_position = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
        global_odom_pose_orientation = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

def main(args=None):
    rclpy.init(args=args)
    subscriber = OdomSubscriber()

    # Spin in a separate thread or use a non-blocking spin_once alternative
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(subscriber)

    # Use a separate thread for the executor
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        while True:
            if global_odom_pose_position is not None and global_odom_pose_orientation is not None:
                print("Latest odom pose:")
                print(f'Position (x, y, z): {global_odom_pose_position[0]}, {global_odom_pose_position[1]}, {global_odom_pose_position[2]}')
                print(f'Orientation (x, y, z, w): {global_odom_pose_orientation[0]}, {global_odom_pose_orientation[1]}, {global_odom_pose_orientation[2]}, {global_odom_pose_orientation[3]}')
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