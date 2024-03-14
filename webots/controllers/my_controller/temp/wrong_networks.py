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

    def __init__(self, num, input_dim, timescale, max_dist, n_hd, n_contxts=10):
        self.n = num
        self.bvcLayer = bvcLayer(max_dist, input_dim)
        n_bvc = self.bvcLayer.n
        self.w_rec_c = tf.Variable(np.zeros((n_hd, num, num)), dtype=tf.float32)
        self.w_in = tf.sparse.from_dense(rng.binomial(1, .05, (num, n_bvc)))
        self.w_rec = tf.zeros(shape=(n_hd, num, num), dtype=tf.float32)
        self.v = tf.zeros(num, dtype=tf.float32)
        self.tau = timescale/1000
        self.bvc_v = tf.zeros(n_bvc)
        self.alpha = .5 # 1
        self.v_prev = tf.zeros(num, dtype=tf.float32)
        self.init_wpb = tf.identity(self.w_in)
        self.z = tf.zeros_like(num, dtype=tf.float32)
        self.rec = 0
        self.theta_m = None
        self.bc_v = 0   
        self.w_bc = tf.zeros((n_hd, num))
        self.trace = tf.zeros_like(self.v)
        self.hd_trace = tf.zeros((n_hd, 1, 1), tf.float64)


    def __call__(self, x_in, hdv, contxt=0, mode="learn", collided=False): #, prev_state=0):
        self.v_prev = tf.identity(self.v)
        self.bvc_v = self.bvcLayer(x_in[0], x_in[1])
        u = tf.exp(tf.sparse.sparse_dense_matmul(self.w_in, self.bvc_v)) - 1 - .3*tf.reduce_sum(self.bvc_v) #
        # print(.08*tf.reduce_sum(self.bvc_v))
        # u = tf.exp(psi_pb*tf.tensordot(self.w_in, self.bvc_v, 1)/tf.reduce_sum(self.bvc_v)) - 1
        self.z += .1 * (u - self.z - self.alpha * (tf.reduce_sum(tf.cast(self.v, tf.float32))))
        # print(tf.reduce_min(u), self.alpha * (tf.reduce_sum(tf.cast(self.v, tf.float32))))
        # tf.tensordot(self.v, tf.tensordot(self.w_rec, tf.cast(hdv, tf.float32), [0, 0]), 1) + (.3 * exploit * 
        # self.z = u - self.alpha * (tf.reduce_sum(self.v) - self.v)
        self.v = tf.tanh(tf.nn.relu(self.z))
        


        # self.v = tf.sigmoid(10*(self.z-self.alpha))

        if np.any(self.v) and mode == "dmtp" and not collided:
            
            # plot.figure(1); plot.clf()
            # plot.imshow(tf.reduce_max(self.w_rec, 0))
            # plot.pause(.1)

            if self.trace is None:
                self.trace = tf.zeros_like(self.v)
            # print(self.trace)
            self.trace += self.tau/3 * (self.v-self.trace)
            self.hd_trace += self.tau/3 * (np.nan_to_num(hdv)[:, np.newaxis, np.newaxis]-self.hd_trace)
            self.w_rec += tf.cast(np.nan_to_num(hdv)[:, np.newaxis, np.newaxis], tf.float32) * (tf.tensordot(self.v[:, np.newaxis], self.trace[np.newaxis, :], 1) - tf.tensordot(self.trace[:, np.newaxis], self.v[np.newaxis, :], 1))
            # self.w_rec += tf.cast(hdv[:, np.newaxis, np.newaxis], tf.float64) * tf.cast(self.v[:, np.newaxis], tf.float64) * (prev_state[np.newaxis, :]  - 1/np.sqrt(8) * tf.cast(self.v[:, np.newaxis], tf.float64) * self.w_rec)
            # tf.linalg.set_diag(self.w_rec, tf.zeros(self.w_rec.shape[:-1], tf.float64))


            # if collided:
            #     self.w_bc += self.tau * (1 - self.w_bc) * hdv[:, np.newaxis] * self.v[np.newaxis, :]

        # plot.figure(2); plot.clf()
        # plot.subplot(311)
        # # plot.stem(self.bvc_v)
        # # plot.subplot(222)
        # plot.stem(u)
        # plot.subplot(312)
        # plot.stem(self.z)
        # plot.subplot(313)
        # plot.stem(self.v)   
        # # plot.title(tf.reduce_mean(self.v).numpy())    # self.v)
        # # # plot.show()
        # plot.pause(.001)

        if np.any(self.v) and not (mode == 'learning'):
            make = tf.sparse.from_dense(tf.greater_equal(self.v, .2) * tf.greater_equal(self.bvc_v[np.newaxis, :], self.bvc_v.mean()) * rng.binomial(1, .05, self.w_in.shape))
            drop = tf.sparse.from_dense(tf.greater_equal(self.v, .2) * tf.less_equal(self.bvc_v[np.newaxis, :], self.bvc_v.mean()) * rng.binomial(1, .05, self.w_in.shape))
            dw = self.tau * (self.v[:, np.newaxis] * (self.bvc_v[np.newaxis, :]  - 1/np.sqrt(.5) * self.v[:, np.newaxis] * self.w_in))  # .5, 8 not 10
            self.w_in.assign_add(dw)

    def exploit(self, direction, contxt=0, num_steps=1):
        
        v = tf.identity(self.v)
        
        # check for collision
        # if tf.tensordot(self.w_bc, self.v, 1)[direction] >= .5:
        #     return tf.zeros_like(v)

        # plot.figure()
        for s in range(num_steps):
            v_prev = tf.identity(v)
            z = 0
            v = tf.tanh(tf.nn.relu(tf.tensordot(tf.cast(self.w_rec[direction], tf.float32), v_prev, 1)- v_prev)) # /tf.norm(v_prev, 1)) # 
            # tf.linalg.normalize(tf.nn.relu(tf.tensordot(tf.cast(self.w_rec[direction], tf.float32), v_prev, 1)), 1)[0] * tf.reduce_sum(v_prev)  # np.heaviside(v_prev - threshold_on, 0), 1))
            # for _ in range(100): 
            #     z += .1*(tf.exp(10*tf.tensordot(self.w_rec[direction], v_prev, 1)  - 1) - self.alpha * (tf.reduce_sum(v)) - z)   # tf.linalg.normalize(self.w_rec[direction], axis=1)[0], 1)
            #     # plot.stem(z)
            #     # plot.show()
            #     # tf.tensordot(self.v, self.w_rec[direction], 1) - self.alpha/9 * (tf.reduce_sum(self.v))
            #     v = tf.sigmoid(10*(z-self.alpha)) # tf.linalg.normalize((z/tf.reduce_sum(z))**self.alpha)[0]   # 

                # if tf.reduce_max(v) >= 1:
                #     break
            # for _ in range(10): 
            #     z += .1*(tf.exp(10*tf.tensordot(v_prev, self.w_rec[direction], 1)  - 1) - 0 * self.alpha * (tf.reduce_sum(v)) - z)   # tf.linalg.normalize(self.w_rec[direction], axis=1)[0], 1)
            #     # plot.stem(z)
            #     # plot.show()
            #     # tf.tensordot(self.v, self.w_rec[direction], 1) - self.alpha/9 * (tf.reduce_sum(self.v))
            #     v = tf.sigmoid(10*(z-self.alpha)) # tf.linalg.normalize((z/tf.reduce_sum(z))**self.alpha)[0]   # 
            #     if tf.reduce_max(v) >= 1:
            #         break
            #     # 

        # v = tf.linalg.normalize(v)[0]

        fig = plot.figure(1)
        ax = fig.add_subplot(3,3,plot_location[direction])
        curr_estimate = np.dot(hmap_z, v)
        try:
            ax.tricontourf(hmap_x, hmap_y, curr_estimate, cmap=cmap)
            ax.set_aspect('equal')
            ax.set_ylim(5, -5)
            ax.set_title("Norm {0:.2f}".format(tf.linalg.norm(v, 1)))
            # ax.set_title("Max firing rate {v}".format(v=int(100*tf.reduce_max(v).numpy())/100))
        except:
            pass
       
        return v #tf.linalg.normalize(v)[0]
        # return conf

    def __getitem__(self, key):
        return self.v[key]  # tf.gather(self.v, key)

    def offline_learning(self, v, hd):
        H_f = lambda tau : tf.cast(tf.sign(tau), tf.double) * tf.exp(-abs(tau)/5) * np.greater_equal(abs(tau), tau_rec ) # +  tf.cast(tf.exp(-abs(tau_rec)/5)/tau_rec * tf.cast(tau, tf.float32) * np.less(abs(tau), tau_rec ), tf.float64)# can add second falling exponential
        x = tf.range(-tau_rec*4, tau_rec*4)
        H = H_f(x)
        H = tf.linalg.normalize(H, np.inf)[0]

        rel_hd_f = lambda tau : np.logical_and(np.less_equal(tau, tau_rec), np.greater_equal(tau, -tau_rec))
        rel_hd = rel_hd_f(x)

        inner = tf.Variable([np.convolve(row, H, 'full') for row in v])
        hdv = tf.Variable([np.convolve(row, rel_hd, 'full')/sum(rel_hd) for row in hd])[:, tf.newaxis, :] 
        v = np.pad(v, [[0, 0], [tf.size(x)//2, -1+tf.size(x)//2]], mode='constant')

        # inner = tf.Variable([np.convolve(row, H, 'same') for row in v]) # [tf.newaxis, :]    
        # hdv = tf.Variable([np.convolve(row, rel_hd, 'same')/sum(rel_hd) for row in hd])[:, tf.newaxis, :] 
        # print(hdv.shape, v.shape, inner.shape)
        
        self.w_rec += tf.tensordot(tf.math.multiply(hdv, v), tf.transpose(inner), 1)

class bvcLayer():
    
    def __init__(self, max_dist, input_dim, sigma_ang=90, n_hd=8, sigma_d=.5):   #sigma_d = .8
        # self.d_i = np.tile(np.arange(0, max_dist, sigma_d/2), n_hd)[np.newaxis, :]  # max_dist * rng.random(size=(1, n_bvc))
        self.d_i = np.tile(np.arange(0, max_dist, sigma_d/2), n_hd)[np.newaxis, :]  # max_dist * rng.random(size=(1, n_bvc))
        self.n = self.d_i.size
        self.in_i = np.repeat(np.linspace(0, input_dim, n_hd, endpoint=False, dtype=int), max_dist/(sigma_d/2))[np.newaxis, :] # np.random.randint(0, input_dim, (1, n_bvc))     
        self.phi_i = np.linspace(0, 2*np.pi, input_dim)[self.in_i]
        self.sigma_ang = tf.constant(np.deg2rad(sigma_ang), dtype=tf.float32)
        self.bvc_out = None
        self.sigma_d = tf.constant(sigma_d, dtype=tf.float32)
        # print( "Distance", self.d_i[0, 210], "Direction:", np.rad2deg(self.phi_i[0, 210]) )

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

    def __call__(self, x_in, visit=False, contxt=1):
        self.v = tf.tensordot(self.w_in_effective, x_in, 1)/tf.linalg.norm(x_in, 1)
        if visit:
            self.w_in_effective = tf.tensor_scatter_nd_update(self.w_in_effective, [[contxt]], [self.w_in_effective[contxt] - 0.2 * x_in * self.w_in_effective[contxt]])

    def newReward(self, pc_net, contxt=0, target=None):
        pc_net = deepcopy(pc_net)
        dw = tf.zeros_like(self.w_in)
        # for j in range(self.n_rs):
        #     self.w_in = tf.tensor_scatter_nd_update(self.w_in, [[contxt]], [tf.math.maximum(self.w_in[contxt], tf.cast(pc_net.v, tf.float32))])
        #     # self.w_in.numpy()[contxt] = tf.math.maximum(self.w_in, tf.cast(pc_net.v, tf.float32))    # self.v * (self.v - theta_m) * pc_net.v/theta_m
        #     pc_net.v = tf.linalg.normalize(tf.nn.relu(tf.tensordot(tf.reduce_max(pc_net.w_rec, 0), pc_net.v, 1)), 1)[0]
        z = 0
        # plot.imshow(pc_net.w_rec)
        # plot.show()

        for t in range(10):
            try:
                fig = plot.figure(2)
                ax = fig.add_subplot(111)
                curr_estimate = np.dot(hmap_z, pc_net.v)
                ax.tricontourf(hmap_x, hmap_y, curr_estimate, cmap=cmap)
                ax.set_aspect('equal')
                ax.set_ylim(5, -5)
                ax.set_title("Max firing rate {v}".format(v=tf.reduce_max(pc_net.v))) # tf.norm(pc_net.v/tf.norm(pc_net.v))))
                plot.show() #pause(1)
            except:
                pass

            # if target != None:
            #     self.w_in = tf.tensor_scatter_nd_update(self.w_in, [[contxt]], [np.maximum(0, self.w_in_effective[contxt] - .6*pc_net.v)]) # * ((target * tf.cast(np.exp(-1/10)*pc_net.v/tf.norm(pc_net.v), tf.float32)) - self.w_in[contxt])])
            #     print("Unlearn", tf.reduce_min(self.w_in))
            #     return

            dw = tf.tensor_scatter_nd_add(dw, [[contxt]], [tf.math.exp(-t/6) * tf.linalg.normalize(pc_net.v, np.inf)[0]]) 
            # self.w_in = tf.tensor_scatter_nd_update(self.w_in, [[contxt]], [tf.math.maximum(self.w_in[contxt], tf.cast(tf.math.exp(-t/3) * tf.linalg.normalize(pc_net.v, np.inf)[0], tf.float32))])
            # plot.figure()
            # plot.subplot(131)
            # plot.stem(pc_net.v)
            
            v = tf.identity(pc_net.v)
            # for _ in range(10): 
            #     z += .1 * (-z + tf.exp(tf.tensordot(10*tf.reduce_max(pc_net.w_rec, 0), pc_net.v, 1) - 1) -  pc_net.alpha * tf.reduce_sum(v))   # tf.linalg.normalize(self.w_rec[direction], axis=1)[0], 1)
            #     v = tf.sigmoid(10*(z-pc_net.alpha))
            # v = tf.linalg.normalize(tf.nn.relu(tf.tensordot(tf.cast(tf.reduce_max(pc_net.w_rec, 0), tf.float32), pc_net.v, 1)), 1)[0] * tf.math.exp(-t/10) * tf.reduce_sum(pc_net.v)  # np.heaviside(v_prev - threshold_on, 0), 1))
            v = tf.nn.relu(tf.tensordot(tf.cast(tf.reduce_max(pc_net.w_rec, 0), tf.float32), pc_net.v, 1) + v) # * tf.math.exp(-t/10))
            # v = tf.nn.relu(tf.tensordot(tf.cast(tf.reduce_max(pc_net.w_rec, 0), tf.float32), pc_net.v, 1) * tf.math.exp(-t/10))
            pc_net.v = tf.tanh(v)
            #  z = tf.exp(psi_pb*tf.tensordot(tf.reduce_max(pc_net.w_rec, 0), pc_net.v, 1) - 1)   # tf.linalg.normalize(self.w_rec[direction], axis=1)[0], 1)
            # plot.subplot(132)
            # plot.stem(tf.tensordot(tf.cast(tf.reduce_max(pc_net.w_rec, 0), tf.float32), pc_net.v, 1))
            # plot.subplot(133)
            # plot.stem(pc_net.v)
            # plot.show()
            # plot.stem(z)
            # plot.show()

            
            

            # tf.linalg.normalize((z/tf.reduce_sum(z))**self.alpha)[0]   # 

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
        # plot.figure(1)
        # plot.clf()
        # plot.subplot(221)
        # plot.stem(x_in)
        v_prime = (.6 * self.w_in[contxt] * x_in) -  self.w_in[contxt] * x_in
        # (r_prime/x_prime_e) * .6 * tf.linalg.normalize(x_in, 1)[0] - self.w_in[contxt] # pc_net.v/tf.norm(pc_net.v, 1) * 
        # plot.subplot(222)
        # plot.stem(self.w_in[contxt] * x_in)
        self.w_in_effective = tf.tensor_scatter_nd_add(self.w_in_effective, [[contxt]], [v_prime])
        # plot.subplot(223)
        # plot.stem(x_in * v_prime)
        # plot.subplot(224)
        # plot.stem(self.w_in[contxt] * x_in)
        # plot.pause(.1)
        print("After:", tf.tensordot(self.w_in[contxt], x_in, 1))

# class gridCellLayer():

#     def __init__(self, num_gcs, num_pcs, num_hdv, timescale):
#         self.w_gc_gc = tf.Variable(np.zeros((num_gcs**2, num_gcs**2)), dtype=tf.float32)
#         marr_wavelet = tf.Variable(np.zeros((num_gcs, num_gcs)), dtype=tf.float32)
#         sigma = 13
#         beta = 3/sigma**2
#         gamma = 1.05 * beta
#         a = 1
#         for i in range(num_gcs):
#             for j in range(num_gcs):
#                 r_squared = (i-num_gcs//2)**2 + (j-num_gcs//2)**2
#                 # marr_wavelet[i, j] = 1/(np.pi*sig**4) * (0*1 - 1/2 * r_squared/sig**2) * np.exp(-r_squared/(2*sig**2))
#                 marr_wavelet.scatter_nd_update([[i, j]], [-1/(sigma*tf.sqrt(2*PI))*np.exp(-1/2 * (np.linalg.norm((np.array([r_squared]))/sigma)**2))])
#         #  [a * np.exp(-gamma * np.linalg.norm(np.array([r_squared]))**2) - np.exp(-beta * np.linalg.norm(np.array([r_squared]))**2)])
#         # marr_wavelet = tf.linalg.normalize(marr_wavelet, np.inf)[0]
#         # plot.imshow(marr_wavelet)
#         # plot.show()
#         marr_wavelet = tf.reshape(marr_wavelet, -1)
#         marr_wavelet = tf.roll(marr_wavelet, num_gcs**2//2, 0)
#         # plot.imshow(tf.reshape(marr_wavelet, [num_gcs, num_gcs]))
#         # plot.show()
        
        
#         for i in range(num_gcs**2):
#             # plot.figure(1)
#             # plot.clf()
#             # plot.subplot(211)
#             # plot.imshow(tf.reshape(marr_wavelet, (num_gcs, num_gcs)))
#             # plot.subplot(212)
#             # plot.imshow(tf.reshape(tf.roll(marr_wavelet, i, 0), (num_gcs, num_gcs)))
#             # plot.pause(.1)
#             self.w_gc_gc.scatter_nd_update([[i]], [tf.roll(marr_wavelet, i, 0)])
#         self.w_gc_gc = tf.linalg.set_diag(self.w_gc_gc, tf.zeros(self.w_gc_gc.shape[1:]))

#         self.k = tf.linalg.inv(tf.eye(num_gcs**2)-self.w_gc_gc)
#         self.w_gc_pc = tf.Variable(.1 * rng.binomial(1, .2, (num_gcs**2, num_pcs)), dtype=tf.float32)
#         self.w_gc_hdv = tf.Variable(.1 * rng.binomial(1, .2, (num_gcs**2, num_hdv)), dtype=tf.float32)
#         self.tau = timescale/1000
#         self.v = tf.Variable(np.zeros((num_gcs**2, 1)), dtype=tf.float32)
#         self.alpha = 2
#         # plot.subplot(211)
#         # plot.imshow(self.w_gc_gc)
#         # plot.subplot(212)
#         # plot.imshow(self.k)
#         # plot.show()

#     def __call__(self, hdv, pc_v, learn=True):
#         # pc_v -= tf.reduce_mean(pc_v)
#         z = tf.tensordot(self.w_gc_pc, tf.cast(pc_v, tf.float32), 1) + tf.tensordot(self.w_gc_hdv, tf.cast(hdv, tf.float32), 1)         #  + np.dot(w_gc_hdv, hdv)) # np.dot(k1, np.dot(w_gc_pc, v) + np.dot(w_gc_hdv, hdv))   #1 * (-gc + np.dot(k1, 1*np.dot(w_gc_pc, v) + 0*np.dot(w_gc_gc, gc) + 0*))
#         z = tf.expand_dims(z, -1)
#         self.v.assign_add(self.tau * (-self.v + tf.nn.relu(tf.tensordot(self.k, z, 1))))

#         # z = tf.nn.relu(np.dot(self.w_gc_pc, pc_v) + np.dot(self.w_gc_hdv, hdv))
#         # self.v = tf.linalg.normalize((z/tf.reduce_sum(z))**self.alpha)[0]
#         # z = np.maximum(0, z)[:, np.newaxis] 
#         # plot.subplot(211)
#         # plot.stem(z)
#         # plot.subplot(212)
#         # plot.stem(self.v.numpy())
#         # # # plot.title(z.numpy().any())
#         # # # plot.subplot(313)
#         # # # plot.imshow(self.w_gc_pc)
#         # # plot.stem(self.v.numpy())
#         # plot.show()
        
#         if self.v.numpy().any() and learn:
#             self.w_gc_pc.assign_add(self.tau/10 * self.v * (tf.cast(tf.transpose(pc_v), tf.float32) - self.v * self.w_gc_pc)) 
#             self.w_gc_hdv.assign_add(self.tau/10 * self.v * (hdv.T -  self.v * self.w_gc_hdv))
#             self.w_gc_pc = tf.Variable(tf.nn.relu(self.w_gc_pc))
#             self.w_gc_hdv = tf.Variable(tf.nn.relu(self.w_gc_hdv))

            
#     def __getitem__(self, key):
#         return self.v[key]


if __name__ == "__main__":
    pcn = placeCellLayer(1000, 100, .1, 5, 720)
    print(pcn.w_in)