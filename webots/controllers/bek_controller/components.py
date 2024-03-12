# todo: fix mexican hat
 
import numpy as np
import matplotlib.pyplot as plot
from scipy import signal

num_dir = 3
np.random.seed(12)
pref_dir = np.linspace(0, 2*np.pi, num_dir, endpoint=False)

def normal_curve(x, mean=0, c=1):
    return np.exp(-(x-mean)**2/(2*c**2))

def head_direction(theta):
    pref_dir = np.linspace(0, 2*np.pi, num_dir, endpoint=False)
    y = np.cos(pref_dir-theta)
    return y

num_gcs = num_dir**2

num_pcs = 20
center = np.zeros((num_pcs**2, 2))
w_gc_hdv = np.random.random((num_gcs, num_dir))
w_gc_hdv /= np.linalg.norm(w_gc_hdv)
# w_gc_gc = np.zeros((num_gcs, num_gcs))
# w_row = -normal_curve(np.array(range(num_dir)), (num_dir-1)/2)
# for row in range(w_gc_gc.shape[0]):
#     w_gc_gc[row] = w_row.shift()
# plot.stem(w_row)
# plot.show()
w_gc_pc = np.random.random((num_gcs, num_pcs**2))
w_gc_pc /= np.linalg.norm(w_gc_pc)
gc = np.zeros((num_dir, num_dir))

for a in range(num_pcs**2):
    x1, y1 = a//num_pcs, a%num_pcs
    center[a] = [x1, y1]


t = 500000
x = num_pcs//2 * np.ones((t, 1))
y = num_pcs//2 * np.ones((t, 1))
v = np.zeros((num_pcs**2, 1))
heading = 0

for i in range(1, t):
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

    hdv = head_direction(heading).reshape(num_dir)  #

    v_new = np.exp(-(np.linalg.norm(center-np.array([x[i], y[i]]).T, axis=-1))**2/(1))[:, np.newaxis]\
        # - np.exp(-(np.linalg.norm(center-np.array([x[i], y[i]]).T, axis=-1))**2/(2*2**2))[:, np.newaxis]
    act =  v_new - v
    z = np.maximum(0, hdv * np.dot(w_gc_pc, v_new).reshape(num_dir, num_dir)).flatten()[:, np.newaxis]
    # if np.sum(z):
    #     continue
    gc = z.flatten()[:, np.newaxis]
    # plot.subplot(1, 3, 1)
    # plot.imshow(z)
    # plot.subplot(1, 3, 2)
    # plot.imshow(gc.reshape(num_dir, num_dir))
    # plot.subplot(1, 3, 3)
    # plot.imshow(w_gc_pc)
    # plot.show()

    w_gc_pc += 1/(100+i//10000) * gc * (v_new.T - gc * w_gc_pc) 
    w_gc_pc = np.maximum(0, w_gc_pc)
    w_gc_hdv += 1/(100+i//10000) * gc * (hdv.T -  gc * w_gc_hdv)
    w_gc_hdv = np.maximum(0, w_gc_hdv)
#     # M += 1/(100) * v * (v.T -  v * M)
#     # M += .01 * v * (v.T - v * M)
    v = v_new
#     w_gc_gc -= 1/(100+i//10000) * gc * (gc_old.T - gc * w_gc_gc)   #  + .5 * np.absolute(w_gc_gc)) # gc * (gc.T - gc * w_gc_gc)
#     # plot.imshow(np.dot(-gc, gc.T) + w_gc_gc)
#     # plot.show()
#     # np.fill_diagonal(w_gc_gc, 0)
#     # plot.imshow(w_gc_gc)
#     # plot.show()
#     # plot.imshow(.01 * gc * (hdv.T -  gc * w_gc_hdv))
#     # plot.show()

#     # M = np.maximum(0, M)
#     gc_old = gc
    
#     

plot.imshow(v.reshape(num_pcs, num_pcs))
plot.title("Place Cell Activation")

plot.figure()
plot.plot(x, y, linewidth=.5)
plot.title(str(x.mean())+" "+str(y.mean()))
plot.figure()

# for i in range(num_gcs**2):
#     ax = plot.subplot(num_gcs, num_gcs, i+1)    #, projection='polar')
#     # ax.set_theta_direction('clockwise')
#     # ax.set_theta_zero_location('N')
#     plot.stem(np.arange(0,2*np.pi,np.deg2rad(theta_inc)), w_gc_hdv[i])
#     plot.title(i)
# # plot.show()

# # w, v = np.linalg.eig(M)
# # print(w.shape, v.shape)
# plot.figure()
# # for i in range(10):
# #     plot.subplot(5, 2, i+1)
for i in range(num_gcs):
    plot.subplot(num_dir, num_dir, i+1)
    plot.imshow(w_gc_pc[i].reshape(num_pcs, num_pcs), origin='lower', cmap='gnuplot2')
    plot.title(i)

plot.figure()
for i in range(num_gcs):
    plot.subplot(num_dir, num_dir, i+1)
    plot.stem(w_gc_hdv[i])
    plot.title(i)
# plot.imshow(M, origin='lower')
# plot.show()

# np.save('w_gc_pc.npy', w_gc_pc)
# np.save('w_gc_hdv.npy', w_gc_hdv)
# np.save('w_gc_gc.npy', w_gc_gc)
# np.save('centers.npy', center)
# np.save('g.npy', g)
# np.save('k1.npy', k1)
# # plot.figure()
# # plot.imshow(np.dot(v, np.dot(w*np.eye(100), v.T)))
# # # proj = np.dot(w*np.eye(10), v)
# # # print(w.shape)
# # # print(v.shape)
# # # plot.imshow(proj)
plot.show()


