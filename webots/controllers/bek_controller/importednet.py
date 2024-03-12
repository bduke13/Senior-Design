import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot
from scipy.stats.mstats import gmean


#todo: fix bvc model by integrating firing over theta

f = lambda x: tf.nn.relu(x).numpy()
# f = np.vectorize(f_s)
env = 0
np.random.seed(env)

class PlaceCellLayer():

    def __init__(self, num: int=1000, input_dim: int=180, tuning_width: float=.05, eta: float=1, inh_prop: float=0.05, dim: float=4, num_hd: int=8):
        """
        :param num: number of place cells
        :param input_dim: dimensionaility of boundary vector input
        :param tuning_width: tuning  with of boundary vector input
        Initializes the network
        """
        n_bvc = 200
        self.n_e = num
        self.w_ee = np.zeros((num_hd, num, num))
        self.w_in = dim * np.random.random((input_dim, num))
        self.v = np.zeros((num, 1))
        self.c = tuning_width
        self.n_hd = num_hd
        self.eta = eta
        self.d_i = dim * np.random.random((1, n_bvc))   # np.random.normal(loc=dim/2, scale=dim/2, size=(1, n_bvc))   # 
        self.phi_i = np.random.random((1, n_bvc)) * 2 * np.pi     # np.linspace(0, 2*np.pi, n_bvc, False)[np.newaxis, :]
        self.sigma_ang = np.deg2rad(30)
        self.w_in_bvc = np.zeros((num, n_bvc))
        for row in range(num):
            cnctd = np.random.choice(n_bvc, 10, False)
            self.w_in_bvc[row][cnctd] = 1
        self.bvc_out = None
        # print(self.w_in_bvc)
        # replace with precise picking

    def __call__(self, sens_in, hd, contxt_in=None, train=True):
        """
        Updates the state of the network
        """
        if contxt_in is None:
            contxt_in = np.ones((self.n_e, self.n_e))
        h_e = np.exp(-gmean(np.abs(sens_in - self.w_in))/(2*self.c**2))[:, np.newaxis]
        d_ve = f(h_e + int(not train)*np.dot(self.w_ee*contxt_in, self.v).sum(0))
        prev_v = self.v
        self.v = np.greater(d_ve, np.percentile(d_ve, 95)) * d_ve
        self.w_ee += hd[:, np.newaxis, np.newaxis] * contxt_in[np.newaxis, :] * (self.eta * np.dot(prev_v, self.v.T)[np.newaxis, :] * np.abs(1 - self.w_ee)) 


    def __expcall__(self, sens_in, hd, contxt_in=None, train=True):
        """
        Updates the state of the network
        """
        self.bvc_out = self.f_bvc(sens_in, np.arange(0, 2*np.pi, 2*np.pi/len(sens_in))[:, np.newaxis])
        d_ve = 1/2 * np.maximum(np.dot(self.w_in_bvc, self.bvc_out)[:, np.newaxis] - 10, 0)
        # chi = np.mean(d_ve/2)**2 * np.mean(d_ve)
        # phi = d_ve * (d_ve - chi)   # np.minimum(.02, np.maximum(-.02, d_ve * (d_ve - chi)))
        # plot.stem(phi)
        # plot.show()
        # plot.subplot(1, 3, 1)
        # plot.stem(self.bvc_out)
        # plot.subplot(1, 3, 2)
        # plot.stem(d_ve)
        # dw = (self.bvc_out * phi) * (self.w_in_bvc > 0)
        # plot.subplot(1, 3, 3)
        # plot.imshow(dw)
        # plot.show()
        # self.w_in_bvc += dw
        if contxt_in is None:
            contxt_in = np.ones((self.n_e, self.n_e))
        # self.w_ee += hd[:, np.newaxis, np.newaxis] * contxt_in[np.newaxis, :] * (self.v * phi.T)[np.newaxis][:]
        self.v = d_ve  
        # diff = np.linalg.norm(sens_in - self.w_in, axis=0)

        # # # if min(diff) < self.c:
        # # #     self.w_in += self.platicity * 
        # # h_e = np.exp(-diff/(2*self.c**2))[:, np.newaxis]
        # # d_ve = f(h_e + int(not train)*np.dot(self.w_ee*contxt_in, self.v).sum(0))
        # prev_v = self.v
             
        # np.greater(d_ve, np.percentile(d_ve, 95))
        # plot.stem(self.v)
        # plot.show()
        # self.w_ee += hd[:, np.newaxis, np.newaxis] * contxt_in[np.newaxis, :] * (self.eta * np.dot(prev_v, self.v.T)[np.newaxis, :] * np.abs(1 - self.w_ee)) 


    def g(self, r, theta):
        return  (np.exp(-(r - self.d_i)**2/(2*self.sigma_rad(self.d_i)**2)) / np.sqrt(2*np.pi*self.sigma_rad(self.d_i)**2)) * \
            (np.exp(-((theta-self.phi_i)**2)/(2*self.sigma_ang**2)) / np.sqrt(2*np.pi*self.sigma_ang**2))


    @staticmethod
    def sigma_rad(d_i):
        return .5*d_i + .5    # smaller c more centralized 


    def f_bvc(self, r, theta):
        return self.g(r, theta).sum(0)


    def __getitem__(self, key):
        """
        :param key: place cell index
        returns firing rate of selected neuron
        """
        return self.v[key]


class RewardCellNet():

    def __init__(self, num, num_pc, num_replay):
        self.v = np.zeros((num, 1))
        self.w_in = np.zeros((num, num_pc))
        self.num_replay = num_replay

    def new_reward(self, pc_net: PlaceCellLayer):
        ac = env    # np.random.randint(0, len(self.v))
        pc_t = pc_net
        for _ in range(self.num_replay):
            self.w_in[ac] = np.maximum(self.w_in[ac], pc_t.v.T[0])
            pc_t.v = np.dot(pc_t.w_ee, pc_t.v).sum(0)
            pc_t.v /= np.linalg.norm(pc_t.v, ord=1)
        plot.stem(self.w_in[ac])
        plot.show()

    def __call__(self, pc_net, avail_actions: np.ndarray=np.ones((8, 1))):
        if pc_net.v.argmax() == self.w_in.sum(0).argmax():
            print("At goal")
            return -1, -1
        self.v = np.dot(self.w_in, pc_net.v)
        if self.v.max() == 0:
            return 0, 0
        potS = np.maximum(np.tensordot(pc_net.w_ee, pc_net.v, [1, 0]), 0)
        np.divide(potS, potS.sum(1)[:, np.newaxis], potS)
        potRew = np.multiply(avail_actions, np.dot(self.w_in, potS)[env])
        # if potRew.max() <= self.v.max():
        #     print("Reward island")
        #     self.w_in[ac] -= .1 * np.multiply(self.w_in[ac], pc_net.v.T[0])
        return potRew.argmax(), potRew.max()


# class BiasingNet():

#     def __init__(self, n: int, dt: float, tau: float=250):
#         self.lamdas = np.random.random((n, 1))
#         self.lamdas[0] = 1
#         self.s = np.random.randint(-10, 10, size=(n, n))
#         self.lam = np.zeros_like(self.s)
#         np.fill_diagonal(self.lam, self.lamdas)
#         self.w = np.matmul(np.matmul(self.s, self.lam), np.linalg.inv(self.s))
#         self.v = np.zeros_like(self.lamdas)
#         self.tau = tau
#         self.dt = dt

#     def __call__(self, h):
#         self.v += self.dt/self.tau * (tf.math.sigmoid(np.dot(self.w, self.v)  + h) - self.v)


class HopfieldNet():

    def __init__(self, n, prop, n_attr):
        self.x = np.zeros((n))
        self.attr = np.random.binomial(1, prop, (n_attr, n))
        self.w = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                self.w[i, j] = (np.mean(np.multiply(self.attr[:, j], self.attr[:, i]))/np.mean(self.attr[:, j])) - np.mean(self.attr[:, i])
        self.w[np.isinf(self.w)] = 0
        self.w = np.nan_to_num(self.w)
        np.fill_diagonal(self.w, 0)
        plot.imshow(self.w)
        plot.show()

    def __call__(self, I=0):
        self.x = f(np.dot(self.w, self.x) + I)


if __name__ == '__main__':
    a = HopfieldNet(100, .5, 10)
    I = a.attr[0]
    # for _ in range(10):
    a(I)
    for _ in range(10):
        for ra in range(10):
            plot.subplot(5, 2, ra+1)
            plot.stem(a.attr[ra])
        a()
        # plot.figure()
        # plot.subplot(2, 1, 1)
        # plot.stem(I)
        # plot.subplot(2, 1, 2)
        # plot.stem(a.x)
        # plot.show()
    # from scipy.integrate import solve_ivp
    # n = 100
    # n_attr = 10
    # on_prop = .05
    # attr = f(np.random.binomial(1, .5, (n_attr, n)))   # -1 + 2*np.random.binomial(1, on_prop, (n_attr, n))
    # w = np.dot(attr.T, attr) - n_attr * np.eye(n) # 1/n * np.dot(attr, attr.T) - n_attr/n * np.eye(n)
    # # plot.imshow(attr)
    # # plot.figure()
    # # plot.imshow(w)
    # # plot.show()
    # # w = np.zeros((n, n))
    # # for i in range(n):
    # #     for j in range(n):
    # #         if i==j:
    # #             continue
    # #         w[i, j] = 1/n * (attr[i]*attr[j]).sum()     # ((2*attr[j]-1)*(2*attr[i]-1)).sum()
    # # print("minimum weight", w.min())
    # # plot.figure()
    # # plot.imshow(w)
    # # plot.show()
    # # plot attractors
    # for p in range(n_attr):
    #     plot.subplot(n_attr//2, 2, p+1)
    #     plot.stem(attr[p])

    # I = np.random.random((n, 1))
    # tau = .5
    # eta = .005
    # np.random.seed(10)
    # y = I # attr[0]   # -1 + 2*np.random.binomial(1, on_prop, (n, 1))   #  n

    # for t in range(1000):
    #     if t%100 == 0:
    #         plot.figure()
    #         plot.stem(y)
    #     y = f(np.dot(w, y))
    # # plot.stem(attr.T[0])
    # # plot.stem(attr.T[1])
    # # plot.stem(attr.T[0][:, np.newaxis])
    # # plot.legend(['Final Pattern', 'Attractor 1', 'Attractor 2'])
    # plot.show()
    
    # # sol = solve_ivp(HopNetODE, [0, 10], x_init, vectorized=True, args=[I, w])
    # # j = -1
    # # # for t in sol['t']:
    # # plot.figure()
    # # plot.stem(sol['y'].T[j])
    # # plot.show()
    # # # j += 1
    
