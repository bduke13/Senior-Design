# TODO: Use smaller goals
# TODO: verify TD learning
# TODO: use self-normalizing learning rule that doesn't give unbounded s'

# changed v_b threshold from .3 to .35

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.cm import get_cmap
from scipy.stats.mstats import gmean
tf.random.set_seed(5)
from numpy.random import default_rng
import _pickle as pickle
from copy import deepcopy
rng = default_rng()
cmap = get_cmap('plasma')

PI = tf.constant(np.pi)
real_direction = {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3}
raw_plot_location = {0: 2, 1: 3, 2: 6, 3: 9, 4: 8, 5: 7, 6: 4, 7: 1}
plot_location = raw_plot_location #{i:raw_plot_location[real_direction[i]] for i in range(8)}
psi_pb = 1 #2.5
tau_rec = 30//2
threshold_on = .05

try:
    with open('hmap_x.pkl', 'rb') as f:
        hmap_x = pickle.load(f)
    with open('hmap_y.pkl', 'rb') as f:
        hmap_y = pickle.load(f)
    with open('hmap_z.pkl', 'rb') as f:
        hmap_z = np.asarray(pickle.load(f))
except:
    pass

class placeCellLayer():
    #self.pcn = placeCellLayer(num_place_cells, 720, self.timestep, dim, self.num_head_directions)
    def __init__(self, num, input_dim, timescale, max_dist, n_hd, n_contxts=10):
        #n is num of cells in place cell layer
        self.n = num
        # bvclayer is boundary cells. max_dist = dim = 12. input_dim = 720
        self.bvcLayer = bvcLayer(max_dist, input_dim) #input dim is number of head direction cells
        # n bvc is set to input dimension of bvcLayer
        n_bvc = self.bvcLayer.n
        #W_rec_c is not used - expiremental variable
        self.w_rec_c = tf.Variable(np.zeros((n_hd, num, num)), dtype=tf.float32)  # (rng.binomial(1, .2, (n_hd, num, num))
        #w_in is boundary vec to place cell weights
        self.w_in = tf.Variable(rng.binomial(1, .2, (num, n_bvc)), dtype=tf.float32) # .2
        # self.w_rec_h = tf.zeros((num, num, n_hd), dtype=tf.float64)
        #w_rec is the recurrent weights
        self.w_rec = tf.zeros(shape=(n_hd, num, num), dtype=tf.float32)
        # ,counts=[.1, 0], probs=[.15], dtype=tf.float64)
        # v is firing rate - activation value
        self.v = tf.zeros(num, dtype=tf.float32)
        self.tau = timescale/1000
        # bvc_v is num boundary boundary vector cells
        self.bvc_v = tf.zeros(n_bvc)
        self.alpha = .5 # 1
        self.v_prev = tf.zeros(num, dtype=tf.float32)
        # init_wpb is initial weight for boundary -> place cell wwight matrix
        self.init_wpb = tf.Variable(self.w_in)
        self.z = tf.zeros_like(num, dtype=tf.float32)
        self.rec = 0
        # thaeta_m is maybe tuning width??
        self.theta_m = None
        self.bc_v = 0   # boundary cell
        self.w_bc = tf.zeros((n_hd, num))

        self.trace = tf.zeros_like(self.v)
        #hd_trace is heaad direction trace - integrates value over time - in papers
        self.hd_trace = tf.zeros((n_hd, 1, 1), tf.float64)
        # print("Mean: ", tf.reduce_mean(self.v))

    def __call__(self, x_in, hdv, contxt=0, mode="learn", collided=False): #, prev_state=0):
        self.v_prev = tf.identity(self.v)
        self.bvc_v = self.bvcLayer(x_in[0], x_in[1])
        u = tf.tensordot(self.w_in, self.bvc_v, 1) - .3*tf.reduce_sum(self.bvc_v) 
        self.z += .1 * (u - self.z - self.alpha * (tf.reduce_sum(tf.cast(self.v, tf.float32))))
        self.v = tf.tanh(tf.nn.relu(self.z))

        if np.any(self.v) and mode == "dmtp" and not collided:

            if self.trace is None:
                self.trace = tf.zeros_like(self.v)

            self.trace += self.tau/3 * (self.v-self.trace)
            self.hd_trace += self.tau/3 * (np.nan_to_num(hdv)[:, np.newaxis, np.newaxis]-self.hd_trace)
            self.w_rec += tf.cast(np.nan_to_num(hdv)[:, np.newaxis, np.newaxis], tf.float32) * (tf.tensordot(self.v[:, np.newaxis], self.trace[np.newaxis, :], 1) - tf.tensordot(self.trace[:, np.newaxis], self.v[np.newaxis, :], 1))

        if np.any(self.v) and not (mode == 'learning'):
            dw = self.tau * (self.v[:, np.newaxis] * (self.bvc_v[np.newaxis, :]  - 1/np.sqrt(.5) * self.v[:, np.newaxis] * self.w_in))  # .5, 8 not 10
            self.w_in.assign_add(dw)

    # update the state of the place cells 
    def exploit(self, direction, contxt=0, num_steps=1):
        
        v = tf.identity(self.v)
        
        for s in range(num_steps):
            v_prev = tf.identity(v)
            z = 0
            v = tf.tanh(tf.nn.relu(tf.tensordot(tf.cast(self.w_rec[direction], tf.float32), v_prev, 1)- v_prev)) 

        fig = plot.figure(1)
        ax = fig.add_subplot(3,3,plot_location[direction])
        curr_estimate = np.dot(hmap_z, v)
        try:
            ax.tricontourf(hmap_x, hmap_y, curr_estimate, cmap=cmap)
            ax.set_aspect('equal')
            ax.set_ylim(5, -5)
            ax.set_title("Norm {0:.2f}".format(tf.linalg.norm(v, 1)))
        except:
            pass
       
        return v # return the updated state of the place cells after num_steps iterations

    def __getitem__(self, key):
        return self.v[key]  # tf.gather(self.v, key)

    def offline_learning(self, v, hd):
        H_f = lambda tau : tf.cast(tf.sign(tau), tf.double) * tf.exp(-abs(tau)/5) * np.greater_equal(abs(tau), tau_rec )
        x = tf.range(-tau_rec*4, tau_rec*4)
        H = H_f(x)
        H = tf.linalg.normalize(H, np.inf)[0]

        rel_hd_f = lambda tau : np.logical_and(np.less_equal(tau, tau_rec), np.greater_equal(tau, -tau_rec))
        rel_hd = rel_hd_f(x)

        inner = tf.Variable([np.convolve(row, H, 'full') for row in v])
        hdv = tf.Variable([np.convolve(row, rel_hd, 'full')/sum(rel_hd) for row in hd])[:, tf.newaxis, :] 
        v = np.pad(v, [[0, 0], [tf.size(x)//2, -1+tf.size(x)//2]], mode='constant')
        
        self.w_rec += tf.tensordot(tf.math.multiply(hdv, v), tf.transpose(inner), 1)

class bvcLayer():
    
    def __init__(self, max_dist, input_dim, sigma_ang=90, n_hd=8, sigma_d=.5):
        self.d_i = np.tile(np.arange(0, max_dist, sigma_d/2), n_hd)[np.newaxis, :]  
        self.n = self.d_i.size
        self.in_i = np.repeat(np.linspace(0, input_dim, n_hd, endpoint=False, dtype=int), max_dist/(sigma_d/2))[np.newaxis, :]
        self.phi_i = np.linspace(0, 2*np.pi, input_dim)[self.in_i]
        self.sigma_ang = tf.constant(np.deg2rad(sigma_ang), dtype=tf.float32)
        self.bvc_out = None
        self.sigma_d = tf.constant(sigma_d, dtype=tf.float32)

    def g(self, r, theta):
        a = tf.exp(-(r[self.in_i] - self.d_i)**2/(2*self.sigma_d**2))/ tf.sqrt(2*PI*self.sigma_d**2)
        b = tf.exp(-((theta[self.in_i]-self.phi_i)**2)/(2*self.sigma_ang**2))/ tf.sqrt(2*PI*self.sigma_ang**2)
        
        return a*b

    def __call__(self, r, theta):
        return tf.reduce_sum(self.g(r, theta), 0)

class rewardCellLayer():
    def __init__(self, num, input_dim, num_replay):
        self.n = num
        self.w_in = tf.Variable(np.zeros((num, input_dim)), dtype=tf.float32)
        self.v = tf.zeros((num, 1))
        self.n_rs = num_replay
        self.w_in_effective = tf.Variable(np.zeros((num, input_dim)), dtype=tf.float32)

    def __call__(self, x_in, visit=False, contxt=1):
        self.v = tf.tensordot(self.w_in_effective, x_in, 1)/tf.linalg.norm(x_in, 1)
        if visit:
            self.w_in_effective = tf.tensor_scatter_nd_update(self.w_in_effective, [[contxt]], [self.w_in_effective[contxt] - 0.2 * x_in * self.w_in_effective[contxt]])

    def newReward(self, pc_net, contxt=0, target=None):
        pc_net = deepcopy(pc_net)
        dw = tf.zeros_like(self.w_in)
        z = 0

        for t in range(10):
            try:
                fig = plot.figure(2)
                ax = fig.add_subplot(111)
                curr_estimate = np.dot(hmap_z, pc_net.v)
                ax.tricontourf(hmap_x, hmap_y, curr_estimate, cmap=cmap)
                ax.set_aspect('equal')
                ax.set_ylim(5, -5)
                ax.set_title("Max firing rate {v}".format(v=tf.reduce_max(pc_net.v))) 
                plot.show()
            except:
                pass
            
            dw = tf.tensor_scatter_nd_add(dw, [[contxt]], [tf.math.exp(-t/6) * tf.linalg.normalize(pc_net.v, np.inf)[0]]) 
            v = tf.identity(pc_net.v)
            v = tf.nn.relu(tf.tensordot(tf.cast(tf.reduce_max(pc_net.w_rec, 0), tf.float32), pc_net.v, 1) + v)
            pc_net.v = tf.tanh(v)

        self.w_in.assign_add(tf.linalg.normalize(dw, np.inf)[0])
        self.w_in_effective = tf.identity(self.w_in)

    def unlearn(self, pc_net):
        print("unlearning")
        self.w_in -= 2/3 * tf.multiply(self.w_in, tf.cast(pc_net.v, tf.float32))

    def __getitem__(self, key):
        return tf.gather(self.v, key)

    def td_update(self, x_in, x_prime_e, r_prime, contxt=0):
        print("r':", r_prime)
        print("Before:", tf.tensordot(self.w_in, x_in, 1))
        v_prime = (.6 * self.w_in[contxt] * x_in) -  self.w_in[contxt] * x_in
        self.w_in_effective = tf.tensor_scatter_nd_add(self.w_in_effective, [[contxt]], [v_prime])
        print("After:", tf.tensordot(self.w_in[contxt], x_in, 1))


if __name__ == "__main__":
    gcn = gridCellLayer(10, 10, 10, 10)