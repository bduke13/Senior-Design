from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Create3
from networks import *
import _pickle as pickle
import os
from astropy.stats import circmean, circvar
from matplotlib.cm import get_cmap
import numpy as np

create3 = Create3(Bluetooth())

np.set_printoptions(precision=2)

# real_direction = {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3}
real_direction = {i:i for i in range(9)}
num_place_cells = 1000
dim = 12
cmap = get_cmap('plasma') # colormap
timestep = 32 * 3
tau_w = 10 
goal_radius = {"explore":0.1, "exploit":0.6}

try:
    with open('hmap_x.pkl', 'rb') as f:
        hmap_x = pickle.load(f)
    with open('hmap_y.pkl', 'rb') as f:
        hmap_y = pickle.load(f)
    with open('hmap_z.pkl', 'rb') as f:
        hmap_z = np.asarray(pickle.load(f))
except:
    pass

class bot():
    maxspeed = 4
    leftSpeed = maxspeed
    rightSpeed = maxspeed
    timestep = 32 * 3
    wheel_radius = 0.036 # From Create3 documentation
    axle_length = 0.235 # From Create3 documentation
    runTime = 5
    num_steps = int(runTime*60//(2*timestep/1000))
    num_head_directions = 8
    hmap_x = np.zeros(num_steps)
    hmap_y = np.zeros(num_steps)
    hmap_z = np.zeros((num_steps, num_place_cells))
    hmap_h = np.zeros((num_steps, num_head_directions))
    hmap_g = np.zeros((num_steps))
    ts = 0   

    def __init__(self, context, bot_mode):
        super().__init__(Bluetooth())
        self.bot = create3
        self.maxspeed = 4
        self.leftSpeed = self.maxspeed
        self.rightSpeed = self.maxspeed
        self.timestep = timestep
        self.mode = bot_mode
        
        # Placeholder values for hardware components, actual instances will be handled by subclasses
        self.compass = None
        self.lidar = None  
        self.leftBumper = None
        self.rightBumper = None
        self.collided = None
        self.leftMotor = None
        self.rightMotor = None
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(self.leftSpeed)
        self.rightMotor.setVelocity(self.rightSpeed)
        self.leftWheelSensor = None
        self.rightWheelSensor = None

        self.boundaries = tf.Variable(tf.zeros(720, 1))
        self.act = tf.zeros(self.num_head_directions)
        self.step(self.timestep)
        self.goalLocation = [[-3, 3], [-1.5, -3], [-3, 2.75], [-1, -1]][context]
        self.expectedReward = 0
        self.lastReward = 0
        self.context = context
        self.s = tf.zeros_like(self.pcn.v)
        self.s_prev = tf.zeros_like(self.pcn.v)

        # Initialization method to be defined by subclasses
        self.initialize_components()
        self.create_networks()
        self.sense()
        self.compute()

    def initialize_components(self):
        """Initialize and enable robot components with proper timestep."""
        pass

    def create_networks(self):
        try:
            with open('pcn.pkl', "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.z *= 0
                self.pcn.v *= 0
                self.pcn.trace = None
                print("Using pre-existing PCN")
        except:
            print('Creating new PCN')
            self.pcn = placeCellLayer(num_place_cells, 720, self.timestep, dim, self.n_hd)
        try:
            with open('rcn.pkl', 'rb') as f:
                self.rcn = pickle.load(f)
                print("Using pre-existing RCN")
        except:
            print('Creating new RCN')
            self.rcn = rewardCellLayer(10, num_place_cells, 3)

    def forward(self):
        self.leftSpeed = self.maxspeed
        self.rightSpeed = self.maxspeed
        
        # Set left motor to rotate continuously at specified speed
        self.leftMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(self.leftSpeed)
        # Set right motor to rotate continuously at specified speed
        self.rightMotor.setPosition(float('inf'))
        self.rightMotor.setVelocity(self.rightSpeed)

        self.sense()

    def turn(self, angle, circle=False):
        self.stop()
        l_offset = self.leftPositionSensor.getValue()
        r_offset = self.rightPositionSensor.getValue()
        self.sense()
        neg = -1.0 if (angle < 0.0) else 1.0
        if circle:
            self.leftMotor.setVelocity(0)
        else:
            self.leftMotor.setVelocity(neg * self.maxspeed/2)
        self.rightMotor.setVelocity(-neg * self.maxspeed/2)
        while True:
            l = self.leftPositionSensor.getValue() - l_offset
            r = self.rightPositionSensor.getValue() - r_offset
            dl = l * self.wheel_radius                 
            dr = r * self.wheel_radius
            orientation = neg * (dl - dr) / self.axle_length
            self.sense()
            if not orientation < neg * angle:
                break
        self.stop()
        self.sense()

    def stop(self):
        self.leftMotor.setVelocity(0)
        self.rightMotor.setVelocity(0)

    def move(self):
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(self.leftSpeed)
        self.rightMotor.setVelocity(self.rightSpeed)

    def sense(self):
        self.boundaries = self.rangeFinder.getRangeImage()
        self.n_index = int(self.get_bearing_in_degrees(self.compass.getValues()))
        self.boundaries = np.roll(self.boundaries, 2*self.n_index)
        rad = np.deg2rad(self.n_index)
        v = np.array([np.cos(rad), np.sin(rad)])
        self.hdv = self.head_direction(0, v)
        self.collided.scatter_nd_update([[0]], [int(self.leftBumper.getValue())])
        self.collided.scatter_nd_update([[1]], [int(self.rightBumper.getValue())])
        self.step(self.timestep)

    def get_bearing_in_degrees(self, north):
        rad = np.arctan2(north[0], north[2])
        bearing = (rad - 1.5708) / np.pi * 180.0
        if bearing < 0:
            bearing = bearing + 360.0
        return bearing

    def tuning_kernel(self, theta_0):
        theta_i = np.arange(0, 2*np.pi, np.deg2rad(360//self.num_head_directions))
        D = np.empty(2, dtype=np.ndarray)
        D[0] = np.cos(np.add(theta_i, theta_0))
        D[1] = np.sin(np.add(theta_i, theta_0))
        return D

    def head_direction(self, theta_0, v_in=[1, 1]):
        k = self.tuning_kernel(theta_0)
        return np.dot(v_in, k)

    def atGoal(self, exploit, s=0):
        currPos = self.robot.getField('translation').getSFVec3f()
        # TODO: change logic to detect tin foil using IR      
        if (self.mode=="dmtp" \
            and np.allclose(self.goalLocation, [currPos[0], currPos[2]], 0, goal_radius["exploit"])) \
            or ((self.mode=="cleanup" or self.mode=="learning") and (self.getTime() >=60*self.runTime)):
            print("Made it")
            print("Distance:", self.compute_path_length())
            print("Started:", np.array([self.hmap_x[0], self.hmap_y[0]]))
            print("Goal:", np.array([currPos[0], currPos[2]]))
            print("Distance:", np.linalg.norm(np.array([self.hmap_x[0], self.hmap_y[0]]) - self.goalLocation) - goal_radius["exploit"])
            print("Time taken:", self.getTime())

            if self.mode=="dmtp":
                self.autopilot(s, currPos)
                
            self.save(True)
            print("Saved!")
            self.worldReload()


    def exploit(self):
        self.s *= 0
        self.stop()
        self.sense()
        self.compute()
        self.atGoal(True)

        if self.ts > tau_w:
            act, max_rew, n_s = 0, 0, 1 
            pot_rew = np.empty(self.n_hd)
            pot_e = np.empty(self.n_hd)
            self.rcn(self.pcn.v, True, self.context)
            if (self.rcn.v[self.context] <= 1e-6):
                self.explore()
                return
                
            obstacles = np.dot(self.pcn.w_bc, self.pcn.v)
           
            for d in range(self.n_hd):
                pcn_v = self.pcn.exploit(d, self.context, num_steps=n_s)
                self.rcn(pcn_v)
                pot_e[d] = tf.norm(pcn_v, 1) # 
                pot_rew[d] = np.nan_to_num(self.rcn.v[self.context]) # rnc.v: firing rate - activation value
                
            print(pot_rew)

            self.act +=  1 * (pot_rew - self.act) 

            act = np.nan_to_num(circmean(np.linspace(0, np.pi*2, self.n_hd, endpoint=False), weights=self.act))    
            var = np.nan_to_num(circvar(np.linspace(0, np.pi*2, self.n_hd, endpoint=False), weights=self.act)) 
            max_rew = pot_rew[int(act//(2*np.pi/self.n_hd))]

            if (max_rew <= 1e-3):
                self.explore()
                return
            
            fig = plot.figure(2); fig.clf()
            ax = fig.add_subplot(projection='polar')
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.plot(np.linspace(0, np.pi*2, self.n_hd, endpoint=False), self.act)
            title = str(np.rad2deg(act)) + ", " + str(np.rad2deg(var)) + ", " + str(tf.reduce_max(self.act).numpy())

            curr_estimate = np.dot(hmap_z, self.pcn.v)
            try:
                ax.tricontourf(hmap_x, hmap_y, curr_estimate, cmap=cmap)
            except:
                pass

            # BEK
            # ax.set_aspect('equal')
            # ax.set_ylim(5, -5)
            # ax.set_title("Max firing rate {v}".format(v=tf.math.argmax(self.pcn.v))) #int(100*tf.reduce_max(self.pcn.v).numpy())/100))
            # # plot.pause(0.01)
           
            if np.any(self.collided):
                self.turn(np.deg2rad(60))
                self.stop()
                self.rcn.td_update(self.pcn.v, pot_e[int(act//(2*np.pi/self.n_hd))], max_rew, self.context)
                return

            else:
                if abs(act) > np.pi:
                    act = act - np.sign(act)*2*np.pi  
                self.turn(-np.deg2rad(np.rad2deg(act) - self.n_index)%(np.pi*2)) # 360*act/self.n_hd - self.n_index))
                print(np.rad2deg(act), self.n_index, np.rad2deg(act) - self.n_index)
            

            for s in range(tau_w):
                self.sense()
                self.compute()
                self.forward()
                self.s += self.pcn.v
                self.atGoal(False, s)

            self.s /= tau_w

            
            # self.trans_prob += np.nan_to_num(.1 * self.hdv[:, np.newaxis, np.newaxis] * (self.s[:, np.newaxis] * (self.s[:, np.newaxis] - tf.reduce_mean(tf.pow(self.s, 2))) * self.s_prev[np.newaxis, :]/tf.reduce_mean(tf.pow(self.s, 2))))
            # # self.trans_prob += self.hdv[:, np.newaxis, np.newaxis] * (self.s[:, np.newaxis] * self.s_prev[np.newaxis, :] - self.s_prev[:, np.newaxis] * self.s[np.newaxis, :])
            # self.pcn.w_rec = self.trans_prob
            self.expectedReward = max_rew/pot_e[int(act//(2*np.pi/self.n_hd))]
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
        self.pcn([self.boundaries, np.linspace(0, 2*np.pi, 720, False)], self.hdv, self.context, self.mode, np.any(self.collided))
        self.step(self.timestep)
        currPos = self.robot.getField('translation').getSFVec3f()
        if self.ts >= self.hmap_x.size:
            return
        self.hmap_x[self.ts] = currPos[0]
        self.hmap_y[self.ts] = currPos[2]
        self.hmap_z[self.ts] = self.pcn.v
        self.hmap_h[self.ts] = self.hdv
        self.hmap_g[self.ts] = tf.reduce_sum(self.pcn.bvc_v)
        self.ts += 1 

    def explore(self):

        self.s_prev = self.s
        self.s *= 0

        for s in range(tau_w):
            self.sense()

            if np.any(self.collided):
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

    # dmtp
    def auto_pilot(self, s_start, currPos):
        while not np.allclose(self.goalLocation, [currPos[0], currPos[2]], 0, goal_radius["explore"]):
            currPos = self.robot.getField('translation').getSFVec3f()
            delta_x = currPos[0] - self.goalLocation[0]
            delta_y = currPos[2] - self.goalLocation[1]
            
            if delta_x >= 0:
                theta = tf.math.atan(abs(delta_y), abs(delta_x))
                desired =  np.pi * 2 - theta if delta_y >= 0 else np.pi + theta
            elif delta_y >= 0:
                theta = tf.math.atan(abs(delta_y), abs(delta_x))
                desired = np.pi/2 - theta
            else:
                theta = tf.math.atan(abs(delta_x), abs(delta_y))
                desired = np.pi - theta

            self.turn(-(desired - np.deg2rad(self.n_index)))
            
            self.sense()
            self.compute()
            self.forward()
            self.s += self.pcn.v
            s_start += 1

        self.s /= s_start
        s_start = 0
        plot.imshow(tf.reduce_max(self.pcn.w_rec, 0))
        plot.show()
        self.rcn.newReward(self.pcn, self.context)

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

def __main__():
    bot = bot(64)
    bot.move()
    bot.stop()
    bot.bot.close()
    return