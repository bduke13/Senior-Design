class directionalplaceCellLayer():

    def __init__(self, num, input_dim, timescale, max_dist, n_hd, n_contxts=10):
        self.n = num
        self.w_rec_c = tf.Variable(rng.binomial(1, .2, (n_contxts, num, num)), dtype=tf.float64)
        # self.w_in_c = tf.Variable(rng.binomial(1, .2, (n_contxts, num, n_bvc)), dtype=tf.float64)
        self.w_rec_h = tf.zeros((num, num, n_hd), dtype=tf.float64)
        self.w_rec = tf.zeros(shape=(num, num), dtype=tf.float64)
        # ,counts=[.1, 0], probs=[.15], dtype=tf.float64)
        self.v = tf.zeros(num, dtype=tf.float64)
        self.tau = timescale/1000
        self.bvcNet = bvcNet(max_dist, input_dim)
        self.w_pc_bvc = np.empty((num, *self.bvcNet.shape))  # .1 * np.random.normal(size=(num, n_bvc))  # 
        self.w_pc_bvc[:, :, 0] = 1
        self.alpha = 20
        self.v_prev = tf.zeros(num, dtype=tf.float64)
        rng.permuted(self.w_pc_bvc, axis=-1, out=self.w_pc_bvc)
        self.init_wpb = self.w_pc_bvc.copy()

    def __call__(self, x_in, hdv, exploit=None, contxt=0):
        w_rec = tf.multiply(tf.tensordot(self.w_rec_h, hdv, 1), self.w_rec_c[contxt] * self.w_rec)
        self.v_prev = tf.identity(self.v)
        self.bvcNet(x_in[0], x_in[1])
        z = tf.reduce_mean(tf.reduce_sum(tf.multiply(self.w_pc_bvc, self.bvcNet.v[np.newaxis, :]), -1), -1)
        # z = tf.tensordot(self.w_pc_bvc, self.bvcNet.v, [[1, 2], [0, 1]])
        self.v = tf.linalg.normalize((z/tf.reduce_sum(z))**self.alpha)[0]
        if np.any(self.v):
            self.learn(hdv, contxt)
        if exploit is not None:
            # plot.subplot(211)
            # plot.stem(self.v)
            w_rec = self.w_rec_h[:, :, exploit] * self.w_rec_c[contxt] * self.w_rec
            # print(w_rec.shape)
            z = tf.nn.relu(tf.tensordot(self.v, w_rec, 1))
            self.v = tf.linalg.normalize((z/tf.reduce_sum(z))**self.alpha)[0]

    def learn(self, hdv, contxt=0):
        dw_rec = self.w_rec_c[contxt] * (self.tau * tf.linalg.set_diag(self.v[:, np.newaxis] * (self.v_prev[np.newaxis, :] - self.v[:, np.newaxis] * self.w_rec ), tf.zeros(self.w_rec.shape[:-1], dtype=tf.float64)))
        self.w_rec += dw_rec
        self.w_rec_h += dw_rec[:, :,  np.newaxis] * (hdv - dw_rec[:, :,  np.newaxis] * self.w_rec_h )
        # self.w_pc_bvc += self.tau * self.v[:, np.newaxis, np.newaxis] * (self.bvcNet.v[np.newaxis, :]  - self.v[:, np.newaxis, np.newaxis] * self.w_pc_bvc)

    def __getitem__(self, key):
        return self.v.numpy()[key]  # tf.gather(self.v, key)

class bvcNet():

    def __init__(self, max_dist, input_dim):
        self.net = [directionalBVCLayer(max_dist, input_dim, direction) for direction in np.linspace(0, np.pi*2, 8, endpoint=False)]
        self.shape = [8, len(self.net[0])]

    def __call__(self, r, theta):
        self.v = tf.convert_to_tensor([layer(r, theta) for layer in self.net])

class directionalBVCLayer():
    
    def __init__(self, max_dist, input_dim, direction=0, sigma_ang=45, d_step=.1):   #60
        self.d_i = np.arange(0, max_dist, d_step)[np.newaxis, :]
        self.in_i = int(np.rad2deg(direction) * input_dim/360) * np.ones_like(self.d_i, dtype=int)
        self.phi_i = np.linspace(0, 2*np.pi, input_dim)[self.in_i]
        # print(self.phi_i)
        self.sigma_ang = np.deg2rad(sigma_ang)
        self.bvc_out = None
        self.size = self.d_i.size
        self.shape = [self.size]
        # print( "Distance", self.d_i[0, 210], "Direction:", np.rad2deg(self.phi_i[0, 210]) )

    def __len__(self):
        return self.size
    
    def g(self, r, theta):
        return  (np.exp(-(r[self.in_i] - self.d_i)**2/(2*self.sigma_rad(self.d_i)**2)) / np.sqrt(2*np.pi*self.sigma_rad(self.d_i)**2)) * \
            (np.exp(-((theta[self.in_i]-self.phi_i)**2)/(2*self.sigma_ang**2)) / np.sqrt(2*np.pi*self.sigma_ang**2))

    def __call__(self, r, theta):
        return self.g(r, theta).sum(0)

    @staticmethod
    def sigma_rad(d_i): #.15
        return .5    # .075*d_i + .075    # smaller c more centralized 

def w_gmean(x, w):
    x = tf.cast(x, tf.float64)
    # print(tf.math.pow(x, 1/w))
    return tf.reduce_prod(tf.math.pow(x, 1/tf.reduce_sum(w)), 0)


if __name__ == "__main__":
    a = tf.Variable([[2], [8]])     # np.random.random((10, 5))
    b = gmean([2, 8])
    c = w_gmean(a, np.ones((2, 1)))
    print(b)
    print(c)