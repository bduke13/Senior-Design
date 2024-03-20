import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import LaserScan
from irobot_create_msgs.msg import HazardDetectionVector, IrIntensityVector
from irobot_create_msgs.action import RotateAngle
from nav_msgs.msg import Odometry
import threading
import time
from sensor_subscriber import CombinedSensorSubscriber


class RotateAngleClient(Node):
    def __init__(self):
        super().__init__('rotate_angle_client')
        self._action_client = ActionClient(self, RotateAngle, 'rotate_angle')
        self._action_complete = False  # Add a flag to indicate action completion

    def send_goal(self, angle, max_rotation_speed):
        goal_msg = RotateAngle.Goal()
        goal_msg.angle = angle
        goal_msg.max_rotation_speed = max_rotation_speed

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            #self.get_logger().info('Goal rejected :(')
            return

        #self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        #self.get_logger().info(f'Result: {result}')
        self._action_complete = True  # Set the flag when the action is complete

    def action_complete(self):
        return self._action_complete

    def reset_action_complete_flag(self):
        self._action_complete = False

def main():
    rclpy.init()

    rotate_angle_client = RotateAngleClient()
    sensor_node = CombinedSensorSubscriber()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(rotate_angle_client)
    executor.add_node(sensor_node)

    # Start executor in a separate thread to keep the main thread free for polling
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    print(f"ground reward test {sensor_node.get_ground_reward()}")
    print(f"bump detecton test {sensor_node.get_bump_detection()}") 

    angle = 1.57  # radians
    max_rotation_speed = 0.5  # some units per second

    # print(f"Starting Turning. Start heading: {}")
    rotate_angle_client.send_goal(angle, max_rotation_speed)

    # Wait for the action to complete
    while not rotate_angle_client.action_complete():
        time.sleep(0.1)  # Sleep to prevent busy waiting

    print("Done turning.")

    # Reset the flag if needed
    rotate_angle_client.reset_action_complete_flag()

    # Cleanup
    rclpy.shutdown()
    executor_thread.join()

if __name__ == '__main__':
    main()
