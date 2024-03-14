# todo: fix mexican hat
 
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

np.random.seed(12)
num_pcs = 11
num_gcs = 11
center = np.zeros((num_pcs**2, 2))
w_gc_gc = np.zeros((num_gcs**2, num_gcs**2))
M = np.zeros((num_pcs**2, num_pcs**2))
sig = 1
w_row = signal.ricker(num_gcs, sig)

marr_wavelet = np.zeros((num_gcs, num_gcs))
for i in range(num_gcs):
    for j in range(num_gcs):
        r_squared = (i-num_gcs//2)**2 + (j-num_gcs//2)**2
        # marr_wavelet[i, j] = 1/(np.pi*sig**4) * (0*1 - 1/2 * r_squared/sig**2) * np.exp(-r_squared/(2*sig**2))
        marr_wavelet[i, j] = -np.exp(-np.linalg.norm((np.array([r_squared]))**2/2))

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

w_gc_pc = np.random.random((num_gcs**2, num_pcs**2))
w_gc_pc /= np.linalg.norm(w_gc_pc)
w_gc_hdv = np.random.random((num_gcs**2, len(np.arange(0,2*np.pi,np.deg2rad(theta_inc)))))
w_gc_hdv /= np.linalg.norm(w_gc_hdv)
gc = np.zeros((num_gcs**2, 1))
gc_old = np.zeros_like(gc)

for a in range(num_pcs**2):
    x1, y1 = a//num_pcs, a%num_pcs
    center[a] = [x1, y1]


t = 500000
x = num_pcs//2 * np.ones((t, 1))
y = num_pcs//2 * np.ones((t, 1))
v = np.zeros((num_pcs**2, 1))
heading = 0
# k1 = np.linalg.inv(np.eye(num_gcs**2)-w_gc_gc)
g = np.zeros((num_gcs**2, 1))

for i in range(1, t):
    k1 = np.linalg.inv(np.eye(num_gcs**2)-w_gc_gc)
    val = np.random.randint(1, 5)
    if val == 1:
        x[i] = x[i-1] + 1
        y[i] = y[i-1]
        heading = 0
    elif val == 2:
        x[i] = x[i-1] - 1
        y[i] = y[i-1]
        heading = np.pi
    elif val == 3:
        x[i] = x[i-1]
        y[i] = y[i-1] + 1
        heading = np.pi/2
    else:
        x[i] = x[i-1]
        y[i] = y[i-1] - 1
        heading = 3*np.pi/2
    x[i] = max(0, min(num_pcs, x[i]))
    y[i] = max(0, min(num_pcs, y[i]))

    # heading += np.random.normal(0, np.pi/4)
    # heading = heading % (2*np.pi)
    dx = np.cos(heading)
    dy = np.sin(heading)
    # x[i]= max(0, min(num_pcs, x[i-1] + dx))
    # if x[i] == x[i-1]:
    #     dx = 0
    # y[i] = max(0, min(num_pcs, y[i-1] + dy))
    # if y[i] == y[i-1]:
    #     dy = 0

    # disp = np.exp(-(np.linalg.norm(disp_center - [1, heading], axis=-1))**2)
    # plot.imshow(disp)
    # plot.show()
    hdv = head_direction([dx, dy])[:, np.newaxis]  #

    v_new = np.exp(-(np.linalg.norm(center-np.array([x[i], y[i]]).T, axis=-1))**2/(1))[:, np.newaxis]\
        # - np.exp(-(np.linalg.norm(center-np.array([x[i], y[i]]).T, axis=-1))**2/(2*2**2))[:, np.newaxis]
    act =  v_new - v
    

    # print(v.shape, hdv.shape)
    # z = (np.dot(w_gc_hdv, hdv)/np.sum(np.dot(w_gc_hdv, hdv)))**2
    # z = (np.dot(w_gc_pc, act)/np.sum(np.dot(w_gc_pc, act)))**2
    # gc = np.dot(w_gc_v, [dx, dy])   # np.dot(w_gc_gc, z)    
    # np.maximum(0, np.dot(w_gc_hdv, hdv) + np.dot(w_gc_gc, gc))    #np.maximum(0, np.dot(k, np.dot(w_gc_hdv, hdv)))    # 
    # plot.stem(gc)
    # plot.show()
    # np.dot(k, np.dot(w_gc_pc, v)))   #

    # gc_c = np.maximum(0, np.dot(k2, np.dot(w_intra_gc, gc) + 0*np.dot(w_gc_gc_c, gc_c) + np.dot(w_gc_hdv, hdv)))
    # gc = np.maximum(0, np.dot(k1, np.dot(w_gc_pc, v) + 0*np.dot(w_gc_gc, gc) + np.dot(w_intra_gc_rev, gc_c)))  # * 

    gc = np.dot(k1, np.dot(w_gc_pc, act) + 0*np.dot(w_gc_hdv, hdv)) #  + np.dot(w_gc_hdv, hdv)) # np.dot(k1, np.dot(w_gc_pc, v) + np.dot(w_gc_hdv, hdv))   #1 * (-gc + np.dot(k1, 1*np.dot(w_gc_pc, v) + 0*np.dot(w_gc_gc, gc) + 0*))
    gc = np.maximum(0, gc) 
    # print(gc.shape, g.shape)
    # g += 1/100 * (gc - np.median(gc)) * gc_old

    # plot.subplot(3, 1, 1)
    # plot.stem(gc-gc.mean())
    # plot.subplot(3, 1, 2)
    # plot.stem(gc_old-gc_old.mean())
    # plot.subplot(3, 1, 3)
    # plot.stem(dg)
    # plot.show()
    w_gc_pc += 1/(100+i//10000) * gc * (act.T - gc * w_gc_pc) 
    w_gc_hdv += 1/(100+i//10000) * gc * (hdv.T -  gc * w_gc_hdv)
    # M += 1/(100) * v * (v.T -  v * M)
    # M += .01 * v * (v.T - v * M)
    v = v_new
    w_gc_gc -= 1/(100+i//10000) * gc * (gc_old.T - gc * w_gc_gc)   #  + .5 * np.absolute(w_gc_gc)) # gc * (gc.T - gc * w_gc_gc)
    # plot.imshow(np.dot(-gc, gc.T) + w_gc_gc)
    # plot.show()
    # np.fill_diagonal(w_gc_gc, 0)
    # plot.imshow(w_gc_gc)
    # plot.show()
    # plot.imshow(.01 * gc * (hdv.T -  gc * w_gc_hdv))
    # plot.show()

    # M = np.maximum(0, M)
    gc_old = gc
    w_gc_pc = np.maximum(0, w_gc_pc)
    w_gc_hdv = np.maximum(0, w_gc_hdv)

plot.imshow(v.reshape(num_pcs, num_pcs))
plot.title("Place Cell Activation")

plot.figure()
plot.plot(x, y, linewidth=.5)
plot.title(str(x.mean())+" "+str(y.mean()))
plot.figure()

for i in range(num_gcs**2):
    ax = plot.subplot(num_gcs, num_gcs, i+1)    #, projection='polar')
    # ax.set_theta_direction('clockwise')
    # ax.set_theta_zero_location('N')
    plot.stem(np.arange(0,2*np.pi,np.deg2rad(theta_inc)), w_gc_hdv[i])
    plot.title(i)
# plot.show()
# plot.imshow(w_gc_pc[0].reshape(10, 10))
# plot.show()
# w, v = np.linalg.eig(M)
# print(w.shape, v.shape)
plot.figure()
# for i in range(10):
#     plot.subplot(5, 2, i+1)
for i in range(num_gcs**2):
    plot.subplot(num_gcs, num_gcs, i+1)
    plot.imshow(w_gc_pc[i].reshape(num_pcs, num_pcs), origin='lower')
    plot.title(i)

plot.figure()
plot.imshow(M, origin='lower')
plot.show()

np.save('w_gc_pc.npy', w_gc_pc)
np.save('w_gc_hdv.npy', w_gc_hdv)
np.save('w_gc_gc.npy', w_gc_gc)
np.save('centers.npy', center)
np.save('g.npy', g)
np.save('k1.npy', k1)
# plot.figure()
# plot.imshow(np.dot(v, np.dot(w*np.eye(100), v.T)))
# # proj = np.dot(w*np.eye(10), v)
# # print(w.shape)
# # print(v.shape)
# # plot.imshow(proj)
# plot.show()


