from networks import *
import _pickle as pickle
import os
from astropy.stats import circmean, circvar
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox
#from controller import Supervisor
import rclpy
import threading
import asyncio
import time
from sensor_subscriber import CombinedSensorSubscriber
from turning_node import RotateAngleClient
from driving_node import CmdVelPublisher
from reset_pose_node import SimpleResetPoseClient
import logging
import math
import sys


# real_direction = {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3}
real_direction = {i:i for i in range(9)}
num_place_cells = 1000
dim = 12
cmap = get_cmap('plasma') # colormap
timestep = 32 * 3
tau_w = 10 
goal_radius = {"explore":0.25, "exploit":0.1524}

class create3Driver():
    
    
    timestep = 32 * 3
    #wheel_radius = 0.036 # From Create3 documentation
    #axle_length = 0.235 # From Create3 documentation
    runTime = 5
    num_steps = int(runTime*60//(2*timestep/1000))
    num_head_directions = 8
    hmap_x = np.zeros(num_steps)
    hmap_y = np.zeros(num_steps)
    hmap_z = np.zeros((num_steps, num_place_cells))
    hmap_h = np.zeros((num_steps, num_head_directions))
    hmap_g = np.zeros((num_steps))
    ts = 0   

    def __init__(self, 
                 bot_context = 0, 
                 bot_mode = None, 
                 sensor_node = None, 
                 turning_node = None, 
                 driving_node = None):
        super().__init__()
        # Define nodes to communicate with to robot
        self.sensor_node = sensor_node
        self.turning_node = turning_node
        self.driving_node = driving_node
        self.create_networks()

        self.maxspeed = 0.8
        self.turning_speed = 1.0
        self.mode = bot_mode
        self.compass = None
        self.collided = False

        self.boundaries = tf.Variable(tf.zeros(720, 1))
        self.act = tf.zeros(self.num_head_directions)
        self.step(self.timestep) # Inherited from Supervisor, advances simulation by timestep
        self.goalLocation = [[-1, 1], [-1.5, -3], [-3, 2.75], [-1, -1]][bot_context]
        self.expectedReward = 0
        self.lastReward = 0
        self.context = bot_context
        self.s = tf.zeros_like(self.pcn.v)
        self.s_prev = tf.zeros_like(self.pcn.v)

        self.sense()
        self.compute()

    def create_networks(self):
        file_prefix = "webots/controllers/my_controller/"
        try:
            with open(file_prefix + 'pcn.pkl', "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.z *= 0
                self.pcn.v *= 0
                self.pcn.trace = None
                print("Using pre-existing PCN")
        except:
            print('Creating new PCN')
            self.pcn = placeCellLayer(num_place_cells, 720, self.timestep, dim, self.num_head_directions)
        try:
            with open(file_prefix + 'rcn.pkl', 'rb') as f:
                self.rcn = pickle.load(f)
                print("Using pre-existing RCN")
        except:
            print('Creating new RCN')
            self.rcn = rewardCellLayer(10, num_place_cells, 3)

    def forward(self):
        self.driving_node.linear_x = self.maxspeed
        self.sense()

    # def turn(self, angle):
    #     if abs(angle) > np.pi:
    #         angle = angle - np.sign(angle) * 2 * np.pi        
    #     self.stop()
    #     print(f"starting angle: {angle}")
    #     self.turning_node.send_goal(angle, self.turning_speed)
    #     print(f"new angle: {int(self.sensor_node.get_odom_heading())}")
    #     # Wait for the action to complete
    #     while not self.turning_node.action_complete():
    #         time.sleep(0.05)  # Sleep to prevent busy waiting
    #     self.turning_node.reset_action_complete_flag()
    #     self.stop()
    #     self.sense()
        
    def turn(self, target_angle_radians):
        # Convert current and target angles to radians
        current_angle_radians = np.deg2rad(self.sensor_node.get_odom_heading())
        print(f"curr: {self.current_degree_heading}")#, target: {target_angle_radians}")
        
        # Calculate relative angle to turn (in radians)
        angle_to_turn = target_angle_radians - current_angle_radians
        
        # Normalize the angle to the range [-pi, pi] aka stop loop di loops
        angle_to_turn = (angle_to_turn + np.pi) % (2 * np.pi) - np.pi
        #print(f'angle_to_turn: {angle_to_turn}')
        
        self.stop()
        #print(f"Starting turn by {angle_to_turn} radians")
        
        # Send the turn command as a relative angle
        self.turning_node.send_goal(-angle_to_turn, self.turning_speed)
        
        # Wait for the action to complete
        while not self.turning_node.action_complete():
            time.sleep(0.05)  # Sleep to prevent busy waiting
        
        self.turning_node.reset_action_complete_flag()
        print(f"after_turn: {self.sensor_node.get_odom_heading()}")
        
        self.stop()
        self.sense()
        
        # Update current heading
        self.current_degree_heading = (self.current_degree_heading + np.degrees(angle_to_turn)) % 360
        print(f"New angle: {self.current_degree_heading} degrees")

    def stop(self):
        self.driving_node.linear_x = 0.0

    # # TODO: write this
    def step(self, timestep):
        time.sleep(0.0096)
        pass

    def sense(self):
        self.boundaries = self.sensor_node.get_scan_data().ranges
        # Rotate by 180 degrees and reverse the list
        self.boundaries = [(x if x != 'inf' else float('inf')) for x in reversed(self.boundaries)]
        # Adjust the angles by adding pi (180 degrees) to each angle and then normalizing
        num_readings = len(self.boundaries)
        self.boundaries = [self.boundaries[(i + num_readings // 2) % num_readings] for i in range(num_readings)]
        # with open('real_boundaries.txt', 'w') as file:
        #     file.writelines([str(boundary) + '\n' for boundary in self.boundaries])
        # sys.exit()
        self.current_degree_heading = int(self.sensor_node.get_odom_heading())
        self.boundaries = np.roll(self.boundaries, 2*self.current_degree_heading)
        rad = np.deg2rad(self.current_degree_heading)
        v = np.array([np.cos(rad), np.sin(rad)])
        self.hdv = self.head_direction(0, v)
        self.collided = self.sensor_node.get_bump_detection()
        self.step(self.timestep) #TODO: DETERMINE IF WE NEED THIS

    def radian_to_heading(self, radian):
        # Convert radians to degrees
        degrees = radian * (180.0 / 3.141592653589793)
        
        # Normalize the degrees to ensure it falls within 0 to 360 degrees
        heading = degrees % 360
        
        # Return the heading
        return heading

    def tuning_kernel(self, theta_0):
        theta_i = np.arange(0, 2*np.pi, np.deg2rad(360//self.num_head_directions))
        D = np.empty(2, dtype=np.ndarray)
        D[0] = np.cos(np.add(theta_i, theta_0))
        D[1] = np.sin(np.add(theta_i, theta_0))
        return D

    def head_direction(self, theta_0, v_in=[1, 1]):
        k = self.tuning_kernel(theta_0)
        return np.dot(v_in, k)

    # TODO: get rid of all references to currPos
    def atGoal(self, exploit, s=0):
        #currPos = self.robot.getField('translation').getSFVec3f()
        # TODO: change logic to detect tin foil using IR      
        if (self.mode=="dmtp" and self.sensor_node.get_ground_reward() and exploit):
            print("Made it")
            self.stop()
            sys.exit()
            #print("Distance:", self.compute_path_length())
            #print("Started:", np.array([self.hmap_x[0], self.hmap_y[0]]))
            #print("Goal:", np.array([currPos[0], currPos[2]]))
            #print("Distance:", np.linalg.norm(np.array([self.hmap_x[0], self.hmap_y[0]]) - self.goalLocation) - goal_radius["exploit"])
            #print("Time taken:", self.getTime())

            #if self.mode=="dmtp":
                #self.auto_pilot(s, currPos)
            
            # # Wait for console action instead of using a GUI window
            # input("Press Enter to save networks...")
            # self.save(True)
            # print("Saved!")
            # self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            
            # Create a simple GUI window with a message box
            #root = tk.Tk()
            #root.withdraw()  # Hide the main window
            #root.attributes("-topmost", True)  # Always keep the window on top
            #root.update()
            #messagebox.showinfo("Information", "Press OK to save networks")
            #print("Saved!")
            # Reset the topmost attribute and destroy the window after closing the message box
            #root.attributes("-topmost", False)
            #root.destroy()  # Destroy the main window
            #self.save(True)
            #self.simulationSetMode(self.SIMULATION_MODE_PAUSE)

    # def normalize_left_turn_angle(self, angle):
    #     # Ensure the angle is within [0, 2π]
    #     angle = angle % (2 * np.pi)
        
    #     # If the angle is greater than π, convert it to a negative value representing the equivalent right turn.
    #     if angle > np.pi:
    #         angle = angle - 2 * np.pi
        
    #     return angle

    def exploit(self):
        self.s *= 0
        self.stop()
        self.sense()
        self.compute()
        self.atGoal(exploit = True)

        if self.ts > tau_w:
            act, max_rew, n_s = 0, 0, 1 
            pot_rew = np.empty(self.num_head_directions)
            pot_e = np.empty(self.num_head_directions)
            self.rcn(self.pcn.v, True, self.context)

            if (self.rcn.v[self.context] <= 1e-6):
                print("entering explore mode")
                self.explore()
                return
                
            obstacles = np.dot(self.pcn.w_bc, self.pcn.v)

            # iterate over all head directions
            for d in range(self.num_head_directions):
                # generate a prediction for the current head direction
                pcn_v = self.pcn.exploit(d, self.context, num_steps=n_s)
                # compute the potential reward for the current head direction
                self.rcn(pcn_v)
                # compute the L1 norm (sum of absolute values) for the current head direction
                pot_e[d] = tf.norm(pcn_v, 1)
                # compute the potential reward for the current head direction
                pot_rew[d] = np.nan_to_num(self.rcn.v[self.context]) # rnc.v: firing rate - activation value
                
            print(pot_rew) # TODO: may not need this

            # compute the head direction with the highest potential reward
            self.act +=  1 * (pot_rew - self.act) 

            # compute the head direction with the highest potential reward
            act = np.nan_to_num(circmean(np.linspace(0, np.pi*2, self.num_head_directions, endpoint=False), weights=self.act))    
            var = np.nan_to_num(circvar(np.linspace(0, np.pi*2, self.num_head_directions, endpoint=False), weights=self.act)) 
            max_rew = pot_rew[int(act//(2*np.pi/self.num_head_directions))]
            print(f"act: {math.degrees(act)}, max_rew: {math.degrees(max_rew)}")
            # didn't explore enough
            if (max_rew <= 1e-3):
                self.explore()
                return
            
            # fig = plot.figure(2); fig.clf()
            # ax = fig.add_subplot(projection='polar')
            # ax.set_theta_zero_location("N")
            # ax.set_theta_direction(-1)
            # ax.plot(np.linspace(0, np.pi*2, self.num_head_directions, endpoint=False), self.act)
            # title = str(np.rad2deg(act)) + ", " + str(np.rad2deg(var)) + ", " + str(tf.reduce_max(self.act).numpy())
            # plot.title(title)
            # plot.pause(.01)

            # Set up the plot
            fig = plt.figure(2)
            fig.clf()
            ax = fig.add_subplot(projection='polar')
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            
            # Calculate the circular mean
            angles = np.linspace(0, 2 * np.pi, self.num_head_directions, endpoint=False)
            circ_mean_angle = circmean(angles, weights=act)

            # Calculate angles and the circular mean
            angles = np.linspace(0, np.pi*2, self.num_head_directions, endpoint=False)
            circ_mean_angle = circmean(angles, weights=self.act)

            # Set up the plot
            fig = plt.figure(2)
            fig.clf()
            ax = fig.add_subplot(projection='polar')
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)

            # Plot activation values as a function of angle
            ax.plot(angles, self.act)

            # Plot an arrow or marker towards the circular mean
            # You might need to adjust the arrow length or use a fixed value
            arrow_length = np.max(self.act)  # Example: max activation value for arrow length
            ax.annotate('', xy=(circ_mean_angle, arrow_length), xytext=(0, 0),
                        arrowprops=dict(facecolor='red', shrink=0.05))

            # Optionally, set title to include the circular mean angle
            title = f"Circular Mean: {np.rad2deg(circ_mean_angle):.2f} degrees"
            plt.title(title)
            plt.show(block=False)
            plt.pause(0.01)

            #curr_estimate = np.dot(hmap_z, self.pcn.v)
            try:
                pass
                #ax.tricontourf(hmap_x, hmap_y, curr_estimate, cmap=cmap)
            except:
                pass

            # BEK
            # ax.set_aspect('equal')
            # ax.set_ylim(5, -5)
            # ax.set_title("Max firing rate {v}".format(v=tf.math.argmax(self.pcn.v))) #int(100*tf.reduce_max(self.pcn.v).numpy())/100))
            # # plot.pause(0.01)

            #TODO: fix self.turn()
            if self.collided:
                print("Ow, I hit something :(")
                self.turn(np.deg2rad(60))
                self.stop()
                self.rcn.td_update(self.pcn.v, max_rew, self.context)
                return

            else:
                # Andrew's self.turn() accepts the number of degrees we should turn
                # Ade's self.turn() accepts the absolute heading to which we should turn
                if abs(act) > np.pi:
                    act = act - np.sign(act)*2*np.pi
                # print(f"turning towards {math.degrees((-np.deg2rad(np.rad2deg(act) - self.current_degree_heading)%(np.pi*2)))}")  
                print(f"turning towards (degrees): {np.rad2deg(circ_mean_angle) % 360}")
                
                # input("press enter to continue")
                #time.sleep(10)
                #self.turn((-np.deg2rad(np.rad2deg(act) - self.current_degree_heading)%(np.pi*2)))
                self.turn(circ_mean_angle)
                print(np.rad2deg(act), self.current_degree_heading, np.rad2deg(act) - self.current_degree_heading)
            

            for s in range(tau_w):
                self.sense()
                self.compute()
                self.forward()
                self.s += self.pcn.v
                self.atGoal(False, s)

            self.s /= tau_w

            self.expectedReward = max_rew/pot_e[int(act//(2*np.pi/self.num_head_directions))]
            self.lastReward = self.rcn.v[self.context]
    
    def save(self, include_maps=True):
        with open('pcn.pkl', 'wb') as output:
            pickle.dump(self.pcn, output)
        with open('rcn.pkl', 'wb') as output:
            pickle.dump(self.rcn, output)
        if include_maps:
            with open('hmap_x.pkl', 'wb') as output:
                pickle.dump(self.hmap_x[:self.ts], output)
            with open('hmap_y.pkl', 'wb') as output:
                pickle.dump(self.hmap_y[:self.ts], output)
            with open('hmap_z.pkl', 'wb') as output:
                pickle.dump(self.hmap_z[:self.ts], output)
            with open('hmap_g.pkl', 'wb') as output:
                pickle.dump(self.hmap_g[:self.ts], output)
            with open('hmap_h.pkl', 'wb') as output:
                pickle.dump(self.hmap_h[:self.ts], output)
        return

    def clear(self):
        try:
            os.remove('pcn.pkl')
            os.remove('rcn.pkl')
            os.remove('hmap_x.pkl')
            os.remove('hmap_y.pkl')
            os.remove('hmap_z.pkl')
            os.remove('hmap_g.pkl')
        except:
            pass

    def compute(self):
        self.pcn([self.boundaries, np.linspace(0, 2*np.pi, 720, False)], self.hdv, self.context, self.mode, self.collided)
        self.step(self.timestep)
        #currPos = self.robot.getField('translation').getSFVec3f()
        if self.ts >= self.hmap_x.size:
            return
        #self.hmap_x[self.ts] = currPos[0]
        #self.hmap_y[self.ts] = currPos[2]
        self.hmap_z[self.ts] = self.pcn.v
        self.hmap_h[self.ts] = self.hdv
        self.hmap_g[self.ts] = tf.reduce_sum(self.pcn.bvc_v)
        self.ts += 1 

    def explore(self):

        self.s_prev = self.s
        self.s *= 0

        for s in range(tau_w):
            self.sense()

            if self.collided:
                self.turn(np.deg2rad(60))
                break
            
            if self.mode == "dmtp":
                self.s += self.pcn.v
                self.atGoal(False, s)
                
            self.compute()
            self.forward()
            self.atGoal(False, s)

        if self.mode == "dmtp":
            self.s /= s

        self.turn(np.random.normal(0, np.deg2rad(30)))

    # # dmtp
    # def auto_pilot(self, s_start, currPos):
    #     while not np.allclose(self.goalLocation, [currPos[0], currPos[2]], 0, goal_radius["explore"]):
    #         #currPos = self.robot.getField('translation').getSFVec3f()
    #         delta_x = currPos[0] - self.goalLocation[0]
    #         delta_y = currPos[2] - self.goalLocation[1]
            
    #         if delta_x >= 0:
    #             theta = tf.math.atan(abs(delta_y), abs(delta_x))
    #             desired =  np.pi * 2 - theta if delta_y >= 0 else np.pi + theta
    #         elif delta_y >= 0:
    #             theta = tf.math.atan(abs(delta_y), abs(delta_x))
    #             desired = np.pi/2 - theta
    #         else:
    #             theta = tf.math.atan(abs(delta_x), abs(delta_y))
    #             desired = np.pi - theta

    #         self.turn(-(desired - np.deg2rad(self.current_degree_heading)))
            
    #         self.sense()
    #         self.compute()
    #         self.forward()
    #         self.s += self.pcn.v
    #         s_start += 1

        # self.s /= s_start
        # s_start = 0
        # plot.imshow(tf.reduce_max(self.pcn.w_rec, 0))
        # plot.show()
        # self.rcn.newReward(self.pcn, self.context)

    def run(self, mode="explore"):
        print(f"goal at {self.goalLocation}")
        print(f"starting in mode {mode}")
        while True:
            if mode=="exploit":
                self.exploit()
            else:
                self.explore()
        
    def compute_path_length(self):
        path_length = 0
        for n in range(self.hmap_x.shape[0]-1):
            path_length += np.linalg.norm(np.array([self.hmap_y[n+1], self.hmap_x[n+1]])-np.array([self.hmap_y[n], self.hmap_x[n]]))
        return path_length



def main():
    np.set_printoptions(precision=2)

    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This suppresses TensorFlow C++ warnings.

    # Set TensorFlow logging to ERROR only
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Suppress other libraries' logging if needed
    logging.getLogger('another_library').setLevel(logging.ERROR)
    
    print("Program starting")
    rclpy.init()

    reset_pose_node = SimpleResetPoseClient()
    reset_pose_node.call_reset_pose()

    turning_node = RotateAngleClient()
    driving_node = CmdVelPublisher()
    sensor_node = CombinedSensorSubscriber()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(turning_node)
    executor.add_node(sensor_node)
    executor.add_node(driving_node)

    # Start executor in a separate thread to keep the main thread free for polling
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    print()

    print("waiting for sensors to read")
    while (not sensor_node.get_bump_detection() is not None
            or not sensor_node.get_ground_reward() is not None
            or not sensor_node.get_scan_data() is not None
            or not sensor_node.get_odom_heading() is not None):
        print(f"bump detection detected: {sensor_node.get_bump_detection() is not None}")
        print(f"ground detection detected: {sensor_node.get_ground_reward() is not None}")
        print(f"scan detection detected: {sensor_node.get_scan_data() is not None}")
        print(f"Heading detected: {sensor_node.get_odom_heading() is not None}")
        print()
        time.sleep(1)
    print("ALL SENSORS DETECTED")
    print()
    print("Initializing Bot")

    #bot = create3Driver()
    bot_context = 0
    mode = "exploit"

    if mode not in ["learn_context", "learn_path", "exploit"]:
        raise ValueError("Invalid mode. Choose either 'learn_context', 'learn_path', or 'exploit'")

    if mode == "learn_context":
        bot.clear()
        bot = create3Driver( bot_context=bot_context, 
                            bot_mode='learning',
                            sensor_node=sensor_node,
                            turning_node=turning_node,
                            driving_node=driving_node)
        bot.run("explore")
    elif mode == "learn_path":
        bot = create3Driver( bot_context=bot_context, 
                            bot_mode='dmtp',
                            sensor_node=sensor_node,
                            turning_node=turning_node,
                            driving_node=driving_node)
        bot.run("explore")
    else: # mode == "exploit"
        bot = create3Driver( bot_context=bot_context, 
                            bot_mode='dmtp',
                            sensor_node=sensor_node,
                            turning_node=turning_node,
                            driving_node=driving_node)
        bot.run("exploit")

    # Cleanup
    rclpy.shutdown()
    executor_thread.join()


if __name__ == '__main__':
    main()

    