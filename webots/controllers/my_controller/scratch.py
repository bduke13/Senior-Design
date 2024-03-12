import numpy as np 
import matplotlib.pyplot as plot
import pickle
import pandas as pd
from scipy.signal import convolve
import tensorflow as tf

plot.scatter([0, -.65], [2.5, 3.5])
plot.show()

with open('hmap_x.pkl', 'rb') as f:
    hmap_x = pickle.load(f)
with open('hmap_y.pkl', 'rb') as f:
    hmap_y = pickle.load(f)
with open('hmap_z.pkl', 'rb') as f:
    hmap_z = np.asarray(pickle.load(f))
with open('hmap_h.pkl', 'rb') as f:
    hmap_h = np.asarray(pickle.load(f))

tau_rec = 15
tau = 0.1
# h_t = np.arange(0, tau_w)
# H = .1*np.exp(-h_t/10)

def offline_learning(v, hd):
    print(v.shape, hd.shape)
    H_f = lambda tau : tf.cast(tf.sign(tau), tf.double) * tf.exp(-abs(tau)/5) * np.greater_equal(abs(tau), tau_rec ) +  tf.cast(tf.exp(-abs(tau_rec)/5)/tau_rec * tf.cast(tau, tf.float32) * np.less(abs(tau), tau_rec ), tf.float64)# can add second falling exponential
    x = tf.range(-tau_rec*4, tau_rec*4)
    H = H_f(x)
    H = tf.linalg.normalize(H, np.inf)[0]

    # plot.plot(x, H)
    # plot.grid()
    # plot.show()

    rel_hd_f = lambda tau : np.logical_and(np.less_equal(tau, tau_rec), np.greater_equal(tau, -tau_rec))
    rel_hd = rel_hd_f(x)

    inner = tf.Variable([np.convolve(row, H, 'full') for row in v])[tf.newaxis, :]    
    hdv = tf.Variable([np.convolve(row, rel_hd, 'full')/sum(rel_hd) for row in hd])[:, tf.newaxis, :] 
    v = np.pad(v, [[0, 0], [tf.size(x)//2, -1+tf.size(x)//2]], mode='constant')
    print(x.shape, v.shape, hdv.shape, inner.shape)
    # plot.stem(hdv[:, 0, 0]) 
    # plot.show()
    
    # print(v.shape, tf.reduce_max(v), tf.reduce_min(v))
    # print(hdv.shape, tf.reduce_max(hdv), tf.reduce_min(hdv))
    # print(x.shape, tf.reduce_max(x), tf.reduce_min(x)) 
    # exit()

    # w_rec = tf.zeros((hdv.shape[0], inner.shape[0], inner.shape[0]), dtype=tf.float32)
    # for i in range(inner.shape[-1]):
    #     w_rec += (1 - w_rec) * tf.cast(hdv[:, :, i], tf.float32) * v[:, :, i] * tf.transpose(inner)[i]
 
    return tf.maximum(tau * tf.tensordot(tf.math.multiply(hdv, v), tf.transpose(inner), 1), 0)

with open('w_rec.pkl', 'wb') as output:
    pickle.dump(offline_learning(hmap_z.T, tf.transpose(hmap_h)), output)

exit()

# v_j = hmap_z[:1000, j].T; v_i = hmap_z[:1000, i].T; v_h = (hmap_h[:1000, h].T)
# plot.figure(1)
# plot.plot(v_j); plot.plot(v_i); plot.plot(v_h)
# plot.figure(2)
# conv = v_h * np.convolve(H, np.pad(v_j, (tau_w, 0), 'constant')[:-tau_w], 'same')
# plot.plot(conv); plot.plot(v_j); plot.plot(v_i); plot.plot(np.multiply(v_i, conv))
# plot.legend(['convolution', 'pre synaptic', 'post synaptic', 'product']) 
# plot.show()
# exit()

w_rec = np.load('w_rec.npy') # np.empty((hmap_h[0].size, hmap_z[0].size, hmap_z[0].size))

# for hd in range(w_rec.shape[0]):
#     for i in range(w_rec.shape[1]):
#         for j in range(w_rec.shape[2]):
#             w_rec[hd, i, j] = np.multiply(hmap_z[i], np.convolve(H, np.pad(hmap_z[j], (tau_w, 0), 'constant')[:-tau_w], 'same')).sum()

# np.save('w_rec.npy', w_rec)
plot.imshow(w_rec[0])

plot.show()

def H(tau):
    if tau>0:
        return np.exp(-tau/20)
    elif tau<0:
        return -np.exp(tau/20)
    else:
        return 0

def ma(x, w=30):
    return np.convolve(x, np.ones(w), 'valid') / w

post = pd.DataFrame(np.pad(hmap_z, ((30, 0), (0, 0)), 'constant')).rolling(30).mean().fillna(0).to_numpy()
pre = pd.DataFrame(np.pad(hmap_z, ((0, 30), (0, 0)), 'constant')).rolling(30).mean().fillna(0).to_numpy()

print(pre.shape, post.shape)
# plot.plot(hmap_z.T[0])
# plot.plot(pre.T[0])
# plot.plot(post.T[0])
# plot.show()
# exit()

ma_int = np.zeros((hmap_z.shape[1], hmap_z.shape[1]))

for i in range(len(pre)):
    ma_int += (post[i][:, np.newaxis] * (pre[i][np.newaxis, :]  - 1/np.sqrt(10) * post[i][:, np.newaxis] * ma_int))

plot.imshow(ma_int)
plot.show()
exit()

plot.imshow(hmap_z.T)
plot.show()

plot.figure(1)

h_x = np.arange(-50, 50)
H_v = np.repeat(np.vectorize(H)(h_x)[np.newaxis, :], hmap_z.shape[1], axis=0)
print(H_v.shape)


def t_integral(T):
    integral = np.zeros((hmap_h.shape[-1], hmap_z.shape[-1], hmap_z.shape[-1]))
    for t in range(T):
        tau_integral = convolve(H_v, hmap_z[t][:, np.newaxis], 'same')
        integral += 1/300 * hmap_h[t][:, np.newaxis, np.newaxis] * np.dot(np.pad(hmap_z, ((50, 50), (0, 0)), 'constant')[t:t+100].T, tau_integral.T)
        # print(H_v.shape, hmap_z[t].shape, tau_integral.shape)
        # plot.subplot(211)
        # plot.stem(hmap_z[t])
        # plot.subplot(212)
        # plot.imshow(integral[hmap_h[t].argmax()])
        # plot.stem(tau_integral[0])
        # plot.imshow(np.dot(hmap_z[t][:, np.newaxis], tau_integral[np.newaxis, :]))
        # plot.show()
        # integral += np.trapz(hmap_z[t][np.newaxis, :] * tau_integral) # hmap_h[t][:, np.newaxis, np.newaxis] * np.dot(hmap_z[t][:, np.newaxis], tau_integral[np.newaxis, :])
        # print(np.trapz(hmap_z[t][:, np.newaxis] * tau_integral).shape)
        # plot.subplot(311)
        # plot.stem(hmap_z[t])
        # plot.subplot(312)
        # plot.stem(tau_integral)
        # plot.subplot(313)
        # plot.imshow(integral[hmap_h[t].argmax()])
        # if tau_integral.any():
        #     plot.show()
        # plot.clf()
    return integral

print(t_integral(len(hmap_z)))
