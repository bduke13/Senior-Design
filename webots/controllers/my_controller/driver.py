from controller import Supervisor, Robot
import numpy as np
from math import hypot
from networks import *
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
import _pickle as pickle
from matplotlib import cm
import os
import cmath
from sklearn.preprocessing import StandardScaler
from scipy.integrate import quad
from scipy.signal import convolve
import pandas as pd
import random
from astropy.stats import circmean, circvar

np.set_printoptions(precision=2)

# real_direction = {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3}
real_direction = {i:i for i in range(9)}
num_pc = 1000
num_gcs = 99
probed = 10
dim = 12
cmap = get_cmap('plasma')
timestep = 32 * 3
tau_w = 10 
goal_r = {"explore":0.1, "exploit":0.6}

h_t = np.arange(0, tau_w)
H = .1*np.exp(-h_t/10)

try:
    with open('hmap_x.pkl', 'rb') as f:
        hmap_x = pickle.load(f)
    with open('hmap_y.pkl', 'rb') as f:
        hmap_y = pickle.load(f)
    with open('hmap_z.pkl', 'rb') as f:
        hmap_z = np.asarray(pickle.load(f))
except:
    pass

class Driver (Supervisor):
    # original maxspeed is 16
    maxspeed = 4
    leftSpeed = maxspeed
    rightSpeed = maxspeed
    timestep = 32 * 3
    wheel_radius = .031
    axle_length = 0.271756
    runTime = 60 * 2
    num_steps = int(runTime*60//(2*timestep/1000))
    n_hd = 8
    hmap_x = np.zeros(num_steps)
    hmap_y = np.zeros(num_steps)
    hmap_z = np.zeros((num_steps, num_pc))
    hmap_h = np.zeros((num_steps, n_hd))
    hmap_g = np.zeros((num_steps))
    ts = 0

    def initialization(self, context, mode, randomize=False):
        print(f"bot mode {mode}")
        self.mode = mode
        self.robot = self.getFromDef('agent')
        self.step(self.timestep)
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timestep)
        self.compass = self.getDevice('compass')
        # self.leftcamera = self.getCamera('lefteye')
        # self.rightcamera = self.getCamera('righteye')
        self.rangeFinderNode = self.getDevice('range-finder')
        self.rangeFinder = self.getDevice('range-finder')
        self.leftBumper = self.getDevice('bumper_left')
        self.rightBumper = self.getDevice('bumper_right') 
        self.collided = tf.Variable(np.zeros(2, np.int32))
        self.display = self.getDevice('display')
        self.rotationField = self.robot.getField('rotation')
        # self.gps = self.getGPS('gps')
        # self.robotLocation = tf.Variable(tf.zeros((2)))
        self.leftMotor = self.getDevice('left wheel motor')
        self.rightMotor = self.getDevice('right wheel motor')
        self.leftPositionSensor = self.getDevice('left wheel sensor')
        self.rightPositionSensor = self.getDevice('right wheel sensor')
        self.leftBumper.enable(self.timestep)
        self.rightBumper.enable(self.timestep)
        self.leftPositionSensor.enable(self.timestep)
        self.rightPositionSensor.enable(self.timestep)
        # self.leftcamera.enable(self.timestep)
        # self.leftcamera.recognitionEnable(self.timestep)
        # self.rightcamera.enable(self.timestep)
        # self.rightcamera.recognitionEnable(self.timestep)
        self.rangeFinder.enable(self.timestep)
        self.compass.enable(self.timestep)
        try:
            with open('pcn.pkl', "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.z *= 0
                self.pcn.v *= 0
                self.pcn.trace = None
                print("Using pre-existing PCN")
        except:
            print('Creating new PCN')
            self.pcn = placeCellLayer(num_pc, 720, self.timestep, dim, self.n_hd)
        try:
            with open('rcn.pkl', 'rb') as f:
                self.rcn = pickle.load(f)
                print("Using pre-existing RCN")
        except:
            print('Creating new RCN')
            self.rcn = rewardCellLayer(10, num_pc, 3)
        # try:
        #     with open('gcn.pkl', 'rb') as f:
        #         self.gcn = pickle.load(f)
        #         print("Using pre-existing GCN")
        # except:
        #     self.gcn = gridCellLayer(num_gcs, num_pc, self.n_hd, self.timestep)

        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(self.leftSpeed)
        self.rightMotor.setVelocity(self.rightSpeed)
        self.knownLandmarks = list()
        self.landmarks = np.inf * np.ones((5, 1))
        self.boundaries = tf.Variable(tf.zeros(720, 1))
        self.act = tf.zeros(self.n_hd)
        self.step(self.timestep)
        self.goalLocation = [[-3, 3], [-1.5, -3], [-3, 2.75], [-1, -1]][context] # [1, .9]
        self.expectedReward = 0
        self.lastReward = 0
        self.context = context
        self.s = tf.zeros_like(self.pcn.v)
        self.s_prev = tf.zeros_like(self.pcn.v)

        if randomize:
            INITIAL = [rng.uniform(-5, 5), 0.5, rng.uniform(-5, 5)]
            self.robot.getField('translation').setSFVec3f(INITIAL)
            self.robot.resetPhysics()

        self.sense()
        self.compute()

    def forward(self):
        self.leftSpeed = self.maxspeed
        self.rightSpeed = self.maxspeed
        self.move()
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
        # print([obj.get_id() for obj in self.rightcamera.getRecognitionObjects()])
        self.step(self.timestep)
        
    def get_bearing_in_degrees(self, north):
        rad = np.arctan2(north[0], north[2])
        bearing = (rad - 1.5708) / np.pi * 180.0
        if bearing < 0:
            bearing = bearing + 360.0
        return bearing

    def tuning_kernel(self, theta_0):
        theta_i = np.arange(0, 2*np.pi, np.deg2rad(360//self.n_hd))
        D = np.empty(2, dtype=np.ndarray)
        D[0] = np.cos(np.add(theta_i, theta_0))
        D[1] = np.sin(np.add(theta_i, theta_0))
        return D

    def head_direction(self, theta_0, v_in=[1, 1]):
        # theta_0 = np.deg2rad(self.n_index)
        # theta_i = np.arange(0, 2*np.pi, np.deg2rad(360//self.n_hd))
        # diff = theta_i-theta_0
        # diff = np.where(diff<np.deg2rad(180), diff, np.deg2rad(360)-diff)
        # return np.exp(-np.power(diff, 2)/np.deg2rad(15))
        k = self.tuning_kernel(theta_0)
        return np.dot(v_in, k)

    def atGoal(self, exploit, s=0):
        currPos = self.robot.getField('translation').getSFVec3f()
        # print(f"curPos {currPos[0], currPos[2]}")
        # print(f"at goal {np.allclose(self.goalLocation, [currPos[0], currPos[2]], 0, goal_r['exploit'])}")        
        if (self.mode=="dmtp" \
            and np.allclose(self.goalLocation, [currPos[0], currPos[2]], 0, goal_r["exploit"])) \
            or ((self.mode=="cleanup" or self.mode=="learning") and (self.getTime() >=60*self.runTime)):
            print("Made it")
            print("Distance:", self.compute_path_length())
            print("Started:", np.array([self.hmap_x[0], self.hmap_y[0]]))
            print("Goal:", np.array([currPos[0], currPos[2]]))
            print("Distance:", np.linalg.norm(np.array([self.hmap_x[0], self.hmap_y[0]]) - self.goalLocation) - goal_r["exploit"])
            print("Time taken:", self.getTime())

            if self.mode == "dmtp":
                self.auto_pilot(s, currPos)
                # self.pcn.w_rec = self.trans_prob
            # plot.figure()
            # plot.imshow(tf.reduce_max(self.pcn.w_rec, 0))
            # plot.figure()
            # plot.stem(self.rcn.w_in[self.context].numpy())
            plot.title(tf.math.atan2(currPos[2] - self.goalLocation[1], currPos[0] - self.goalLocation[0]).numpy())
            plot.show()
            self.save(True)
            self.worldReload()

    def exploit(self):

        self.s *= 0
        self.stop()
        self.sense()
        self.compute()
        self.atGoal(True)

        if self.ts > tau_w:
            act, max_rew, n_s = 0, 0, 1 
            # while True: 
            pot_rew = np.empty(self.n_hd)
            pot_e = np.empty(self.n_hd)
            self.rcn(self.pcn.v, True, self.context)
            # print("Reward", self.rcn.v[self.context], "Most Active", self.pcn.v.numpy().argsort()[-3:])
            if (self.rcn.v[self.context] <= 1e-6): # or (self.rcn.v[self.context] <= self.lastReward):
                # for s in range(tau_w):
                self.explore()
                return
            

            # if self.rcn.v[self.context] < self.lastReward:
            #     # Focal search
            #     print("Focal search") #Surprise! From {b} tp {a}".format(b=self.expectedReward, a=max_rew))

            #     # loop
            #     #  self.turn(2*np.pi, True)
            
            #     # continue
            #     for s in range(5):
            #         self.stop()
            #         self.sense()
            #         self.compute()
            #         self.forward()

            #     self.rcn.td_update(self.pcn.v, pot_e[act], max_rew, self.context)
            obstacles = np.dot(self.pcn.w_bc, self.pcn.v)
           
            for d in range(self.n_hd):
                # self.compute(exploit=True)
                # for s in range(4):
                pcn_v = self.pcn.exploit(d, self.context, num_steps=n_s)
                # plot.figure()
                # plot.stem(pcn_v)
                # plot.title(d)
                # plot.show()
                self.rcn(pcn_v)
                # print(self.rcn.v)
                # boundary =  tf.tensordot(self.pcn.w_bc[d], pcn_v, 1).numpy()
                pot_e[d] = tf.norm(pcn_v, 1)
                pot_rew[d] = np.nan_to_num(self.rcn.v[self.context]) # if pot_e[d]>.1 else 0
                # print("Direction: {}, Reward: {}, Energy: {}".format(real_direction[d], pot_rew[d], pot_e[d]))
                # actions[d] = cmath.rect(self.rcn.v, actions[d])
                
            # if pot_rew > max_rew:
            #     act, max_rew = real_direction[d], pot_rew  # else act, self.expectedReward
            print(pot_rew)

            self.act +=  1 * (pot_rew - self.act)   # int(var<np.rad2deg(50))*

            act = np.nan_to_num(circmean(np.linspace(0, np.pi*2, self.n_hd, endpoint=False), weights=self.act))    # real_direction[pot_rew.argmax()] # random.choices(range(self.n_hd), np.exp(pot_rew))[0] # 
            var = np.nan_to_num(circvar(np.linspace(0, np.pi*2, self.n_hd, endpoint=False), weights=self.act)) 
            max_rew = pot_rew[int(act//(2*np.pi/self.n_hd))]

            if (max_rew <= 1e-3): # or (np.rad2deg(var)>50): # or (self.rcn.v[self.context] <= self.lastReward):
                # for s in range(tau_w):
                self.explore()
                return
            
            fig = plot.figure(2); fig.clf()
            ax = fig.add_subplot(projection='polar')
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.plot(np.linspace(0, np.pi*2, self.n_hd, endpoint=False), self.act)
            title = str(np.rad2deg(act)) + ", " + str(np.rad2deg(var)) + ", " + str(tf.reduce_max(self.act).numpy())
            # plot.title(title) # np.rad2deg(circmean(np.linspace(0, np.pi*2, self.n_hd, endpoint=False), weights=self.act)))
            # plot.pause(.01)



            # if max_rew < self.expectedReward and n_s < 4: # prob > max_rew or 
            #     # self.rcn.newReward(self.pcn, self.context, True)
            #     # self.turn(np.random.normal(0, np.deg2rad(180)))
            #     # print("Moving randomly")
            #     n_s += 1
            # else:
            #     break

            fig = plot.figure(1)
            ax = fig.add_subplot(335)
            curr_estimate = np.dot(hmap_z, self.pcn.v)
            try:
                # ax.stem(self.pcn.v)
                ax.tricontourf(hmap_x, hmap_y, curr_estimate, cmap=cmap)
            except:
                pass
            ax.set_aspect('equal')
            ax.set_ylim(5, -5)
            ax.set_title("Max firing rate {v}".format(v=tf.math.argmax(self.pcn.v))) #int(100*tf.reduce_max(self.pcn.v).numpy())/100))
            # plot.pause(0.01)
           
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

    def visualize_replay(self, num_steps, direction=None):
        self.sense()
        self.compute()
        with open('hmap_x.pkl', 'rb') as f:
            hmap_x = pickle.load(f)
        with open('hmap_y.pkl', 'rb') as f:
            hmap_y = pickle.load(f)
        with open('hmap_z.pkl', 'rb') as f:
            hmap_z = np.asarray(pickle.load(f))

        fig = plot.figure(1)
        
        if not direction:
            for p in range(1, num_steps+1):
                ax = fig.add_subplot(1, num_steps, p)
                curr_estimate = np.dot(hmap_z, self.pcn.v)
                ax.tricontourf(hmap_x, hmap_y, curr_estimate, cmap=cmap)
                ax.set_aspect('equal')
                ax.set_ylim(5, -5)
                self.rcn(self.pcn.v)
                ax.set_title("Max firing rate {v}".format(v=tf.reduce_max(self.pcn.v)))
                self.pcn.v = tf.linalg.normalize(tf.nn.relu(tf.tensordot(tf.reduce_max(self.pcn.w_rec, 0), self.pcn.v, 1)), 1)[0]
            

        elif direction=='all':
            direction = range(self.n_hd)
            for d in direction:
                self.compute()
                for p in range(1, num_steps+1):
                    ax = fig.add_subplot(len(direction), num_steps, num_steps*d+p)
                    curr_estimate = np.dot(hmap_z, self.pcn.v)
                    ax.tricontourf(hmap_x, hmap_y, curr_estimate, cmap=cmap)
                    ax.set_aspect('equal')
                    ax.set_ylim(5, -5)
                    self.rcn(self.pcn.v)
                    c = self.pcn.exploit(d)
                    ax.set_title("Max firing rate {v}".format(v=tf.reduce_max(self.pcn.v).numpy()))

        elif type(direction) is int:
            self.compute()
            for p in range(1, num_steps+1):
                ax = fig.add_subplot(1, num_steps, p)
                curr_estimate = np.dot(hmap_z, self.pcn.v)
                ax.tricontourf(hmap_x, hmap_y, curr_estimate, cmap=cmap)
                ax.set_aspect('equal')
                ax.set_ylim(5, -5)
                self.rcn(self.pcn.v)
                ax.set_title(self.rcn.v.numpy()[self.context])
                self.pcn.exploit(d)

        plot.show()
        return

    def save(self, include_maps=True):
        with open('pcn.pkl', 'wb') as output:
            # self.pcn.init_wpb = self.pcn.init_wpb
            pickle.dump(self.pcn, output)
        with open('rcn.pkl', 'wb') as output:
            pickle.dump(self.rcn, output)
        # with open('gcn.pkl', 'wb') as output:
        #     pickle.dump(self.gcn, output)
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
            # os.remove('gcn.pkl')
            os.remove('hmap_x.pkl')
            os.remove('hmap_y.pkl')
            os.remove('hmap_z.pkl')
            os.remove('hmap_g.pkl')
        except:
            pass

    def compute(self):
        self.pcn([self.boundaries, np.linspace(0, 2*np.pi, 720, False)], self.hdv, self.context, self.mode, np.any(self.collided)) # , self.hmap_z[self.ts-tau_w])
        # self.hmap_h[self.ts-tau_w:self.ts].mean(0)
        # self.rcn(self.pcn.v)
        ## self.gcn(self.hdv, self.pcn.v - self.pcn.v_prev)
        self.step(self.timestep)
        currPos = self.robot.getField('translation').getSFVec3f()
        # rgba = to_hex(cmap(self.rcn[self.context]))
        # self.display.setColor(int(rgba[1:], 16))
        # self.display.drawPixel(int(currPos[0]*10) + 50, int(currPos[2]*10) + 50)
        if self.ts >= self.hmap_x.size:
            return
        self.hmap_x[self.ts] = currPos[0]
        self.hmap_y[self.ts] = currPos[2]
        self.hmap_z[self.ts] = self.pcn.v  # [0:num_pc])
        self.hmap_h[self.ts] = self.hdv
        self.hmap_g[self.ts] = tf.reduce_sum(self.pcn.bvc_v)

        # if 60<self.ts<=self.num_steps and self.mode=='dmtp':
        #     pre = self.hmap_z[self.ts-60:self.ts-30]
        #     post = self.hmap_z[self.ts-30:self.ts]
        #     # self.pcn.w_rec +=  self.hdv[:, np.newaxis, np.newaxis] * post[:, np.newaxis] * ( pre[:, np.newaxis]  - 1/np.sqrt(8) * post[:, np.newaxis] * self.pcn.w_rec)  # 8 not 10
        #     plot.subplot(121)
        #     plot.title(str(self.ts) + " " + str(self.ts-60)+ " " + str(self.ts-30))
        #     plot.stem(pre[-1])
        #     plot.subplot(122)
        #     plot.stem(self.hmap_z[self.ts-30])
        #     plot.pause(.1)

        # self.hmap_g.append(self.gcn.v)  # [0:num_gcs])    # [probed])
            # [int(currPos[0]*10) + 50, int(10*currPos[2]) + 50] = self.pcn[probed] 
        # [int(currPos[0]*10) + 50, int(10*currPos[2]) + 50] = self.pcn[probed]  
            # [int(currPos[0]*10) + 50, int(10*currPos[2]) + 50] = self.pcn[probed] 
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
            # self.trans_prob += np.nan_to_num(.1 * self.hdv[:, np.newaxis, np.newaxis] * (self.s[:, np.newaxis] * (self.s[:, np.newaxis] - tf.reduce_mean(tf.pow(self.s, 2))) * self.s_prev[np.newaxis, :]/tf.reduce_mean(tf.pow(self.s, 2))))
            # # self.trans_prob += self.hdv[:, np.newaxis, np.newaxis] * (self.s[:, np.newaxis] * self.s_prev[np.newaxis, :] - self.s_prev[:, np.newaxis] * self.s[np.newaxis, :])
            # self.pcn.w_rec = self.trans_prob

        self.turn(np.random.normal(0, np.deg2rad(30)))

        
        # self.sense()
        # self.atGoal(False)
        # self.compute()
        
        # if self.ts > tau_w:
        #     if np.any(self.collided):
        #         self.turn(np.deg2rad(60))
        #     elif np.random.binomial(1, .1):   # turn
        #         self.turn(np.random.normal(0, np.deg2rad(30)))
        #     else:
        #         self.forward()

        
    def manualControl(self):
        k = self.keyboard.getKey()
        if k != -1:
            print("Before:", self.hdv.argmax(), self.n_index)
        if k==ord('W'):
            self.forward()
        elif k==ord('D'):
            self.turn(-np.deg2rad(90))
        elif k==ord('A'):
            self.turn(np.deg2rad(90))
        elif k==ord('S'):
            self.stop()
        if k != -1:
            print("After:", self.hdv.argmax(), self.n_index)

    # dmtp
    def auto_pilot(self, s_start, currPos):
        # self.pcn.w_rec = self.trans_prob
        while not np.allclose(self.goalLocation, [currPos[0], currPos[2]], 0, goal_r["explore"]):
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

            self.turn(-(desired - np.deg2rad(self.n_index))) # - np.pi - np.deg2rad(self.n_index))
            
            self.sense()
            self.compute()
            self.forward()
            self.s += self.pcn.v
            s_start += 1
        self.s /= s_start
        # self.trans_prob += np.nan_to_num(.1 * self.hdv[:, np.newaxis, np.newaxis] * (self.s[:, np.newaxis] * (self.s[:, np.newaxis] - tf.reduce_mean(tf.pow(self.s, 2))) * self.s_prev[np.newaxis, :]/tf.reduce_mean(tf.pow(self.s, 2))))
        # self.pcn.w_rec = self.trans_prob
        s_start = 0
        # currPos = self.robot.getField('translation').getSFVec3f()
        # print("New location", currPos[0], currPos[2])
        # print(self.hmap_z.shape, self.hmap_h.shape)
        # self.pcn.offline_learning(self.hmap_z.T, self.hmap_h.T)
        plot.imshow(tf.reduce_max(self.pcn.w_rec, 0))
        plot.show()
        self.rcn.newReward(self.pcn, self.context)

    def run(self, mode="explore"):
        # with open('w_rec.pkl', 'rb') as f:
        #     self.pcn.w_rec = tf.cast(pickle.load(f)[:, :, :, 0], tf.float32)
        #     # print(self.pcn.w_rec.shape)
        print(f"goal {self.goalLocation}")
        print(f"starting in mode {mode}")
        while True:
            if mode=="exploit":
                self.exploit()
            else:
                self.explore()
            if self.keyboard.getKey() in (ord('W'), ord('D'), ord('A'), ord('S')):
                print("Switching to manual control")
                while True:
                    self.manualControl()

    
    def compute_path_length(self):
        l = 0
        for n in range(self.hmap_x.shape[0]-1):
            l += np.linalg.norm(np.array([self.hmap_y[n+1], self.hmap_x[n+1]])-np.array([self.hmap_y[n], self.hmap_x[n]]))
        return l

    def grid_probe(self, context, x_dim=10, y_dim=10):
        x = np.arange(-x_dim/2, x_dim/2, .1); y = np.arange(-x_dim/2, x_dim/2, .1)
        ts = x.flatten().shape[0] *  y.flatten().shape[0] * tau_w
        print(x.shape, ts)
        self.hmap_x = np.zeros(ts)
        self.hmap_y = np.zeros(ts)
        self.hmap_z = np.zeros((ts, num_pc))

        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                pos = [x[i], 1, y[j]]
                self.robot.getField('translation').setSFVec3f(pos)
                self.robot.resetPhysics()
                for _ in range(tau_w):
                    self.stop()
                    self.sense()
                    self.compute()
                
        self.save()

                

