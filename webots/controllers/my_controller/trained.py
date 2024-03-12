import numpy as np
import matplotlib.pyplot as plot
from scipy import signal

theta_inc = 15

def tuning_kernel(theta_0):
    theta_i = np.arange(0,2*np.pi,np.deg2rad(15))
    D = np.empty(2, dtype=np.ndarray)
    D[0] = np.cos(np.add(theta_i, theta_0))
    D[1] = np.sin(np.add(theta_i, theta_0))
    return D

def head_direction(v_in, theta_0=0):
    k = tuning_kernel(theta_0)
    return np.dot(v_in, k)

# plot.stem(np.rad2deg(np.arange(0,2*np.pi,np.deg2rad(30))), head_direction([1, 0]))
# plot.show()

np.random.seed(12)
num_pcs = 20
num_gcs = 5
# num_cgs = 10
center = np.zeros((num_pcs**2, 2))
w_gc_gc = np.zeros((num_gcs**2, num_gcs**2))
# gc_center = np.zeros((num_gcs**2, 2))
M = np.zeros((num_pcs**2, num_pcs**2))
sig = 1
w_row = signal.ricker(num_gcs, sig)

marr_wavelet = np.zeros((num_gcs, num_gcs))
for i in range(num_gcs):
    for j in range(num_gcs):
        r_squared = (i-num_gcs//2)**2 + (j-num_gcs//2)**2
        marr_wavelet[i, j] = 1/(np.pi*sig**4) * (1 - 1/2 * r_squared/sig**2) * np.exp(-r_squared/(2*sig**2))

for i in range(num_gcs**2):
    y = i%num_gcs
    x = i//num_gcs
    w_gc_gc[i] = 1*np.roll(marr_wavelet, (x-num_gcs//2, y-num_gcs//2), (0, 1)).flatten()
    # plot.imshow(w_gc_gc[i].reshape(num_gcs, num_gcs))
    # plot.title(i)
    # plot.show()
    # for j in range(num_gcs):
    # w_gc_gc[i] = np.roll(w_row, num_gcs//2 + i+1)  # = 2/(np.sqrt(3*sig)*np.pi**(1/4)) * (1 - (t/sig)**2) * np.exp(-(t**2)/(2*sig**2))
# b = np.random.binomial(1, .2, size=(num_gcs, num_gcs))
# w_gc_gc = (b + b.T)/2
# plot.imshow(w_gc_gc[0:20, :])
# plot.figure()
np.fill_diagonal(w_gc_gc, 0)
# plot.subplot(2, 1, 1)
# plot.imshow(w_gc_gc[0].reshape(num_gcs, num_gcs))
# plot.subplot(2, 1, 2)
# plot.imshow(w_gc_gc[1].reshape(num_gcs, num_gcs))
# plot.figure()
# plot.imshow(w_gc_gc)
# plot.show()

# w_gc_pc = np.random.random((num_gcs**2, num_pcs**2))
# w_gc_pc /= np.linalg.norm(w_gc_pc)
w_gc_hdv = np.random.random((num_gcs**2, len(np.arange(0,2*np.pi,np.deg2rad(theta_inc)))))
w_gc_hdv /= np.linalg.norm(w_gc_hdv)
w_gc_v = np.random.random((num_gcs**2, 2))
w_gc_gc_c = np.copy(w_gc_gc)
w_intra_gc = np.random.random((num_gcs**2, num_gcs**2))
w_intra_gc_rev = np.random.random((num_gcs**2, num_gcs**2))
gc_c = np.zeros((num_gcs**2, 1))
gc = np.zeros((num_gcs**2, 1))

x1, x2, y1, y2 = np.meshgrid(np.arange(0, num_pcs), np.arange(0, num_pcs), np.arange(0, num_pcs), np.arange(0, num_pcs), indexing='ij')

for a in range(num_pcs):
    for b in range(num_pcs):
        for c in range(num_pcs):
            for d in range(num_pcs):
                ix = num_pcs*a + b
                iy = num_pcs*c + d
                # M[ix, iy] = np.exp(-np.linalg.norm((np.array([x1[a, b, c, d], x2[a, b, c, d]]) - np.array([y1[a, b, c, d], y2[a, b, c, d]])))**2/10)
                center[ix] = [x1[a, b, c, d], x2[a, b, c, d]]

x = num_pcs//2 
y = num_pcs//2
v = np.zeros((num_pcs**2, 1))

k1 = np.linalg.inv(np.eye(num_gcs**2)-w_gc_gc)
k2 = np.linalg.inv(np.eye(num_gcs**2)-w_gc_gc_c)

w_gc_gc = np.load("w_gc_gc.npy")
w_gc_hdv = np.load("w_gc_hdv.npy")
w_gc_pc = np.load("w_gc_pc.npy")
print(w_gc_pc.shape)
print(gc.shape)

v = np.exp(-(np.linalg.norm(center-np.array([x, y]).T, axis=-1))**2/1)[:, np.newaxis]
heading = np.pi
dx = np.cos(heading)
dy = np.sin(heading)
hdv = head_direction([dx, dy])[:, np.newaxis]


t = 50000
x = num_pcs//2 # * np.ones((t, 1))
y = num_pcs//2 # * np.ones((t, 1))
# v = np.zeros((num_pcs**2, 1))
# heading = 0
k1 = np.linalg.inv(np.eye(num_gcs**2)-w_gc_gc)
k2 = np.linalg.inv(np.eye(num_gcs**2)-w_gc_gc_c)

hmap = np.zeros((num_pcs+1, num_pcs+1))

# for i in range(1, t):
#     val = np.random.randint(1, 5)
#     if val == 1:
#         x[i] = x[i-1] + 1
#         y[i] = y[i-1]
#         heading = 0
#     elif val == 2:
#         x[i] = x[i-1] - 1
#         y[i] = y[i-1]
#         heading = np.pi
#     elif val == 3:
#         x[i] = x[i-1]
#         y[i] = y[i-1] + 1
#         heading = np.pi/2
#     else:
#         x[i] = x[i-1]
#         y[i] = y[i-1] - 1
#         heading = 3*np.pi/2
#     x[i] = max(0, min(num_pcs, x[i]))
#     y[i] = max(0, min(num_pcs, y[i]))

#     # heading += np.random.normal(0, np.pi/4)
#     # heading = heading % (2*np.pi)
#     dx = np.cos(heading)
#     dy = np.sin(heading)

#     hdv = head_direction([dx, dy])[:, np.newaxis]  

#     gc = np.dot(k1, np.dot(w_gc_hdv, hdv)) # np.dot(k1, np.dot(w_gc_pc, v) + np.dot(w_gc_hdv, hdv))   #1 * (-gc + np.dot(k1, 1*np.dot(w_gc_pc, v) + 0*np.dot(w_gc_gc, gc) + 0*))
#     gc = np.maximum(0, gc) 
#     hmap[int(x[i]), int(y[i])] = gc[0]

# for h in np.arange(0, 2*np.pi, np.pi/4):
#     dx = np.cos(h)
#     dy = np.sin(h)
#     hdv = head_direction([dx, dy])[:, np.newaxis]
#     v = np.exp(-(np.linalg.norm(center-np.array([x, y]).T, axis=-1))**2/1)[:, np.newaxis]
#     plot.imshow(v.reshape(num_pcs, num_pcs))    # np.dot(w_gc_pc, v).reshape(num_gcs, num_gcs))
#     plot.title("pc initial activity")
#     # plot.figure()
#     gc = np.dot(w_gc_pc, v)
#     # plot.imshow(gc.reshape(num_gcs, num_gcs))
#     # plot.title("GC activity")
#     # plot.figure()
#     # plot.subplot(2, 1, 1)
#     # plot.imshow(np.maximum(0, np.dot(k1, np.dot(w_gc_hdv, hdv))).reshape(num_gcs, num_gcs))
#     # plot.title("HDV generated GC activity with k")
#     # plot.subplot(2, 1, 2)
#     gc = np.maximum(0, np.dot(k1, np.dot(w_gc_hdv, hdv)))
#     # plot.imshow(gc.reshape(num_gcs, num_gcs))
#     # plot.title("HDV generated GC activity w/o k")
#     plot.figure()
#     v = np.maximum(0, np.dot(w_gc_pc.T, gc))
#     plot.imshow(v.reshape(num_pcs, num_pcs))
#     plot.title("GC transformed to PC space "+str(np.rad2deg(h)))
#     plot.show()
plot.subplot(1, 2, 1)
plot.imshow(v.reshape(num_pcs, num_pcs) + np.dot(w_gc_pc.T, w_gc_pc)[:, 10].reshape(num_pcs, num_pcs))
plot.subplot(1, 2, 2)
plot.imshow(v.reshape(num_pcs, num_pcs))
plot.show()