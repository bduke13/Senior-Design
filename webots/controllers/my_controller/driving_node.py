import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
import threading

class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        qos_profile = QoSProfile(depth=10)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', qos_profile)
        self.linear_x_lock = threading.Lock()  # A lock to manage access to `linear_x`
        self._linear_x = 0.0  # Default speed, use property to access
        self.timer = self.create_timer(1/20, self.timer_callback)  # 20Hz

    @property
    def linear_x(self):
        with self.linear_x_lock:
            return self._linear_x

    @linear_x.setter
    def linear_x(self, value):
        with self.linear_x_lock:
            self._linear_x = value
            self.get_logger().info(f'Setting linear x to: {self._linear_x}')

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = self.linear_x  # Use the property here
        msg.angular.z = 0.0
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    cmd_vel_publisher = CmdVelPublisher()

    try:
        # Example on how to change the linear_x from another thread
        def update_linear_x():
            import time
            while not rclpy.ok():
                time.sleep(1)  # Wait for rclpy to initialize
            time.sleep(2)  # Simulate waiting for some condition
            cmd_vel_publisher.linear_x = 0.5  # Update the speed

        update_thread = threading.Thread(target=update_linear_x)
        update_thread.start()

        rclpy.spin(cmd_vel_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        cmd_vel_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
