import rclpy
from rclpy.node import Node
from irobot_create_msgs.srv import ResetPose

class SimpleResetPoseClient(Node):
    def __init__(self):
        super().__init__('simple_reset_pose_client')
        self.client = self.create_client(ResetPose, '/reset_pose')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /reset_pose not available, waiting again...')

    def call_reset_pose(self):
        request = ResetPose.Request()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Success: %r' % future.result())
        else:
            self.get_logger().error('Failed to call service /reset_pose')

def main():
    rclpy.init()

    # Your existing setup
    # ...

    reset_pose_client = SimpleResetPoseClient()
    reset_pose_client.call_reset_pose()

    # Your additional code
    # ...

    rclpy.shutdown()

if __name__ == '__main__':
    main()
