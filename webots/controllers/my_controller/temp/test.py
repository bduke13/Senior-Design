import numpy as np
import matplotlib.pyplot as plot



def tuning_kernel(theta_0):
    theta_i = np.arange(0,2*np.pi,np.deg2rad(15))
    D = np.empty(2, dtype=np.ndarray)
    D[0] = np.cos(np.add(theta_i, theta_0))
    D[1] = np.sin(np.add(theta_i, theta_0))
    return D

def head_direction(v_in, theta_0=0):
    k = tuning_kernel(theta_0)
    return np.dot(v_in, k)

x = np.array([5, 5])
np.random.seed(12)
num_pcs = 11
num_gcs = 11
center = np.zeros((num_pcs**2, 2))

for a in range(num_pcs**2):
    x1, y1 = a//num_pcs, a%num_pcs
    center[a] = [x1, y1]

p = np.exp(-(np.linalg.norm(center-x.T, axis=-1))**2/(1))[:, np.newaxis]
S = np.load('w_gc_pc.npy')
g = np.load('w_gc_gc.npy')
# g = np.zeros_like(S)
k1 = np.load('k1.npy')
# np.fill_diagonal(g, np.load('g.npy'))
# plot.imshow(g)
# plot.show()
w_g_hd = np.load('w_gc_hdv.npy')

heading = 0
dx = np.cos(heading)
dy = np.sin(heading)
x_prime = x + np.array([dx, dy])

hd = head_direction([dx, dy])
# g = k1  #np.dot(k1, (np.dot(w_g_hd, hd[:, np.newaxis])))

plot.imshow(g)
plot.show()

dp = np.exp(-(np.linalg.norm(center-x_prime.T, axis=-1))**2/(1))[:, np.newaxis] - p
p += dp

dp_prime = np.dot(np.linalg.inv(S), np.dot(g, np.dot(S, dp)))

plot.subplot(2, 2, 1)
plot.imshow(p.reshape(num_pcs, num_pcs))
plot.title("p")
plot.subplot(2, 2, 2)
plot.imshow(dp.reshape(num_pcs, num_pcs))
plot.title("dp")
plot.subplot(2, 2, 3)
plot.imshow(dp_prime.reshape(num_pcs, num_pcs))
plot.title("dp'")
plot.subplot(2, 2, 4)
plot.imshow((p+dp_prime).reshape(num_pcs, num_pcs))
plot.title("p'")
plot.show()