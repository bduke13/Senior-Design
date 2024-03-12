import _pickle as pickle
import plotly
import glob
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import cm, rcParams
from numpy.random import default_rng
import tensorflow as tf
import os
import subprocess
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.tri as tri
from scipy.interpolate import griddata
cmap = cm.get_cmap('jet') # 'plasma')
rng = default_rng()
contxt = 0
rcParams.update({'font.size': 12})

with open('pcn.pkl', 'rb') as f:
    pc_net = pickle.load(f)
with open('rcn.pkl', 'rb') as f:
    rc_net = pickle.load(f)
probed = 243 #, 524, 665, 818
top_3 = tf.math.top_k(rc_net.w_in[contxt], k=40).indices.numpy() # tf.math.argmax(, 1)

goal_r = .6
# exit()
# with open('gcn.pkl', 'rb') as f:
#     gc_net = pickle.load(f)

# plot.plot(tf.reduce_sum(pc_net.w_rec[:, :, 936], 1))
# for direction in range(8):
#     plot.subplot(2, 4, direction+1)
#     plot.plot(pc_net.w_rec[direction, :, 936])
# plot.show()
# exit()

neighbours = tf.math.argmax(pc_net.w_rec[:, probed], -1).numpy()
top_3_rnd = [top_3, rng.integers(0, 1000, 3), neighbours, (893, 79, 30)][0] 
print(top_3_rnd)
print(tf.reduce_max(pc_net.w_rec[:, probed], -1).numpy())
img = plot.imread("robot_arena_one.PNG")
goalLocation = [[-3,3], [-1.5, -3], [-3, 2.75], [-1, -1]][contxt] # [1, .9]

with open('hmap_x.pkl', 'rb') as f:
    hmap_x = pickle.load(f)[10:]
with open('hmap_y.pkl', 'rb') as f:
    hmap_y = pickle.load(f)[10:]
with open('hmap_z.pkl', 'rb') as f:
    hmap_z = np.asarray(pickle.load(f))[10:]
with open('hmap_g.pkl', 'rb') as f:
    hmap_g = np.asarray(pickle.load(f))

print(hmap_z.shape)

# print(tf.reduce_min(rc_net.w_in))
# print(rc_net.w_in.shape)

# plot.imshow(tf.reduce_max(pc_net.w_rec, 0))
# plot.show()


# exit()

# plot.stem(tf.reduce_max(pc_net.w_rec[:, 18], 0))
# plot.show()
# exit()

# plot.plot(hmap_z.T[probed])
# for i in neighbours:
#     plot.plot(hmap_z.T[i])
# legends = ["probed", *neighbours.numpy()]
# plot.legend(legends)
# plot.show()

# for d in range(8):
#     plot.subplot(2, 4, d+1)
#     plot.stem(pc_net.w_rec[d, :, probed])
# plot.show()



most_active = tf.reduce_max(hmap_z, 0).numpy().argsort()[-120:-80] # rc_net.w_in[contxt].numpy().argsort()[-20:] # 
reward_function = tf.tensordot(rc_net.w_in[contxt], tf.cast(hmap_z.T, tf.float32), 1)/tf.cast(tf.reduce_sum(hmap_z, -1), tf.float32)
total_activity = tf.reduce_sum(hmap_z, 1)


# plot.stem(rc_net.w_in[contxt])
# plot.show()

def rewardMapPlot():
    fig = plot.figure(dpi=200)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    params = {'mathtext.default': 'regular' }          
    plot.rcParams.update(params)

    # grid_z = griddata((hmap_x, hmap_y), np.nan_to_num(reward_function), (np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)), method='linear')
    # cntr = ax.tricontourf(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100), grid_z) # hmap_x, hmap_y, np.nan_to_num(reward_function), cmap=cmap, alpha=.6)
    ax.plot(hmap_x, hmap_y, linestyle='dashed', color='b', alpha=.8) # s=.5, c=np.nan_to_num(reward_function), cmap=cmap, alpha=.6, s=.5) #
    goal = plot.Circle((goalLocation), goal_r, color='g', alpha=.5) #  plot.Circle((goalLocation), goal_r, color='r', linewidth=2, fill=False) # 
    # cntr = ax.hexbin(hmap_x, hmap_y, reward_function, 50, alpha=.6, cmap=cmap)
    
    # plot_trisurf(hmap_x, hmap_y, reward_function, cmap=cmap)
    ax.add_patch(goal)
    ax.set_ylim(5, -5)
    ax.set_xlim(-5, 5)
    ax.imshow(img, extent=[-5, 5, -5, 5], origin='lower')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # fig.colorbar(cntr)
    # plot.title("$v_g^r$ shown over maze {i} for the goal shown in red".format(i=contxt+1)) # $  #$v_g^r$ shown over maze {i} for the goal shown in white".format(i=contxt+1)) #
    plot.savefig('photos_quick\C{c}Run4B.png'.format(c=contxt)) # Map.png'.format(c=contxt)) # 
    plot.show()   

def probeOne(probed):
    fig = plot.figure()
    ax = fig.add_subplot(111)   # , projection='3d')
    ax.set_aspect('equal')
    params = {'mathtext.default': 'regular' }          
    plot.rcParams.update(params)
    # print(np.argwhere(np.isnan(hmap_z[:, probed])))
    cntr = ax.hexbin(hmap_x, hmap_y, np.nan_to_num(hmap_z[:, probed].flatten()), 50, cmap=cmap, alpha=.6) #, s=.5) #, vmin=0, vmax=1) marker='*', 
    goal = plot.Circle((goalLocation), goal_r, color='r', linewidth=2, fill=False)
    # plot_trisurf(hmap_x, hmap_y, reward_function, cmap=cmap)
    ax.add_patch(goal)
    # ax.plot(hmap_x, hmap_y, linewidth=.5)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(5, -5)
    ax.set_xlim(-5, 5)
    ax.imshow(img, extent=[-5, 5, -5, 5], origin='lower')
    plot.colorbar(cntr)
    v = "v_{"+str(probed)+"}^p"
    plot.title("${v}$ shown over maze {i}".format(v=v, i=contxt+1))
    plot.savefig("photos_quick\C{c}PC{i}.png".format(c=contxt, i=probed))
    print("Reward value:", rc_net.w_in[contxt, probed])
    plot.show()

def probeState(state):
    fig = plot.figure()
    ax = fig.add_subplot(111)   # , projection='3d')
    ax.set_aspect('equal')
    params = {'mathtext.default': 'regular' }          
    plot.rcParams.update(params)
    # print(np.argwhere(np.isnan(hmap_z[:, probed])))
    cntr = ax.scatter(hmap_x, hmap_y, c=np.nan_to_num(tf.tensordot(hmap_z, state, 1).numpy().flatten()), marker='*', cmap=cmap, s=.5) #, vmin=0, vmax=1)
    goal = plot.Circle((goalLocation), goal_r, color='g', alpha=.25, fill=True)
    # plot_trisurf(hmap_x, hmap_y, reward_function, cmap=cmap)
    ax.add_patch(goal)
    # ax.plot(hmap_x, hmap_y, linewidth=.5)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(5, -5)
    ax.set_xlim(-5, 5)
    ax.imshow(img, extent=[-5, 5, -5, 5], origin='lower')
    plot.colorbar(cntr)
    v = "v_{"+str(probed)+"}^p"
    plot.title("${v}$ shown over maze {i}".format(v=v, i=contxt+1))
    plot.savefig("photos_quick\C{c}PC{i}.png".format(c=contxt, i=probed))
    print("Reward value:", rc_net.w_in[contxt, probed])
    plot.show()

def showTopThree():

    fig = plot.figure()

    ax = fig.add_subplot(131)
    ax.set_aspect('equal')
    ax.tricontourf(hmap_x, hmap_y, hmap_z[:, top_3_rnd[0]], cmap=cmap)  # 
    ax.set_ylim(5, -5)
    plot.title(str(rc_net.w_in[contxt, top_3_rnd[0]].numpy()) + ", " + str(hmap_z[:, top_3_rnd[0]].max()))

    ax = fig.add_subplot(132)
    ax.set_aspect('equal')
    ax.tricontourf(hmap_x, hmap_y, hmap_z[:, top_3_rnd[1]], cmap=cmap)  # 
    ax.set_ylim(5, -5)
    plot.title(str(rc_net.w_in[contxt, top_3_rnd[0]].numpy()) + ", " + str(hmap_z[:, top_3_rnd[1]].max()))

    ax = fig.add_subplot(133)
    ax.set_aspect('equal')
    ax.tricontourf(hmap_x, hmap_y, hmap_z[:, top_3_rnd[2]], cmap=cmap)
    ax.set_ylim(5, -5)
    plot.title(str(rc_net.w_in[contxt, top_3_rnd[2]].numpy()) + ", " + str(hmap_z[:, top_3_rnd[2]].max()))

    plot.show()

def getStatistics(probed: int):
    x_stats, y_stats = DescrStatsW(hmap_x, weights=hmap_z[:, probed]), DescrStatsW(hmap_y, weights=hmap_z[:, probed])
    print('Centered at: ({x_m}, {y_m}) with spread of ({x_v},{y_v})'.format(x_m=x_stats.mean, y_m=y_stats.mean, x_v=x_stats.std, y_v=y_stats.std))

def conjuctiveGCs(batch:int=0):
    n_gcs = 11  # int(np.sqrt(hmap_g[-1].size))
    # print(n_gcs)
    for x in range(n_gcs):
        for y in range(n_gcs):
            i = y + (x * n_gcs)
            ax = plot.subplot(n_gcs, n_gcs, i+1, projection='polar')
            ax.plot(np.linspace(0, 2*np.pi, 8, endpoint=False)[:, np.newaxis], gc_net.w_gc_hdv[(batch*n_gcs**2)+i])
            ax.set_aspect('equal')
            ax.set_title((batch*n_gcs**2)+i)
            ax.tick_params(
                which='both',      
                bottom=False,      
                top=False,         
                labelbottom=False) 

    plot.figure()
    for x in range(n_gcs):
        for y in range(n_gcs):
            i = y + (x * n_gcs)
            ax = plot.subplot(n_gcs, n_gcs, i+1)
            ax.tricontourf(hmap_x, hmap_y, np.nan_to_num(hmap_g[:, (batch*n_gcs**2)+i].flatten()), cmap=cmap)
            ax.set_aspect('equal')
            ax.set_title(tf.reduce_max(hmap_g[:, i]).numpy().round(2))
            ax.tick_params(
                which='both',      
                bottom=False,      
                top=False,         
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False)

def showAll():
    n_pcs = int(np.sqrt(hmap_z[-1].size))
    plot.figure()
    for x in range(n_pcs):
        for y in range(n_pcs):
            i = y + (x * n_pcs)
            ax = plot.subplot(n_pcs, n_pcs, i+1)
            ax.tricontourf(hmap_x, hmap_y, np.nan_to_num(hmap_z[:, i].flatten()), cmap=cmap)
            ax.set_aspect('equal')
            ax.set_title(tf.reduce_max(hmap_z[:, i]).numpy().round(2))
            ax.tick_params(
                which='both',      
                bottom=False,      
                top=False,         
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False)

def tuning_kernel(theta_0):
    theta_i = np.arange(0, 2*np.pi, np.deg2rad(360//8))
    D = np.empty(2, dtype=np.ndarray)
    D[0] = np.cos(np.add(theta_i, theta_0))
    D[1] = np.sin(np.add(theta_i, theta_0))
    return D

def head_direction(theta_0, v_in=[1, 1]):
    k = tuning_kernel(theta_0)
    return np.dot(v_in, k)  # np.maximum(0, np.dot(v_in, k))

def generate_frames(n_frames):
    folder = 'video'
    for _ in range(hmap_z.shape[0]):
        fig = plot.figure(1); plot.clf()
        ax = fig.add_subplot(121)
        # z = np.dot(hmap_z, tf.tensordot(tf.transpose(gc_net.w_gc_pc), tf.cast(img, tf.float32), 1))
        # pc_ac = z   # tf.linalg.normalize((z/tf.reduce_sum(z))**10)[0].numpy()
        # print(pc_ac.shape, hmap_x.shape, hmap_z.shape, z.shape, tf.tensordot(tf.transpose(gc_net.w_gc_pc), gc_net.v, 1).shape)
        # print(np.dot(hmap_z, hmap_z[j]).shape, hmap_x.shape, gc_net.w_gc_pc.numpy().T.shape, hmap_z.shape)
        ax.tricontourf(hmap_x, hmap_y, np.dot(hmap_z, hmap_z[_]), cmap=cmap)
        ax.scatter(hmap_x[_], hmap_y[_])
        ax.set_aspect('equal')
        ax = fig.add_subplot(122)
        ax.stem(hmap_z[_])
        ax.set_ylim(0, 1)
        plot.pause(.1)
        if _ >= n_frames:
            break
        plot.savefig(folder + "/file%02d.png" % _)

def generate_video():
    import cv2
    import os

    image_folder = 'video'
    video_name = 'video1.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 9, frameSize = (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def compute_path_length():
    l = 0
    for n in range(hmap_x.shape[0]-1):
        # print("x_2", np.array([hmap_y[n+1], hmap_x[n+1]]))
        # print("x_1", np.array([hmap_y[n], hmap_x[n]]))
        # print("d", np.array([hmap_y[n+1], hmap_x[n+1]])-np.array([hmap_y[n], hmap_x[n]]))
        l_n = np.linalg.norm(np.array([hmap_y[n+1], hmap_x[n+1]])-np.array([hmap_y[n], hmap_x[n]]))
        # print(l_n)
        l += l_n
        # break
    return l

if __name__ == "__main__":    
    # plot.stem(rc_net.w_in[contxt])
    # plot.title("r values")
    # plot.show()           
    plot.imshow(hmap_z.T)
    plot.show()

    rewardMapPlot()

    # s = tf.reduce_sum(hmap_z[:20], 0)
    # for t in range(10):
    #     probeState(s)
    #     plot.show()
    #     s = tf.tensordot(tf.reduce_max(pc_net.w_rec, 0), s, 1)
    # probeOne(probed)
    # for i in neighbours: # top_3_rnd: #most_active: #
    #     probeOne(i)
    # exit()
    # showTopThree()
    for n in most_active:
        probeOne(n)
    exit()
    # num_gcs = 99
    # for step in range(200):
    #     fig = plot.figure(1); plot.clf()
    #     ax = fig.add_subplot(211)
    #     ax.tricontourf(hmap_x, hmap_y, np.dot(hmap_z, pc_net.v), cmap=cmap)
    #     ax.scatter(hmap_x[-1], hmap_y[-1])
    #     ax.set_aspect('equal')
    #     # plot.imshow(gc_net.v.numpy().reshape((num_gcs, num_gcs)))
    #     ax = fig.add_subplot(212)
    #     # print(hmap_z.shape, gc_net.w_gc_pc.shape, gc_net.v.shape, tf.tensordot(tf.transpose(gc_net.w_gc_pc), gc_net.v, 1).shape)
    #     z = np.dot(hmap_z, tf.tensordot(tf.transpose(gc_net.w_gc_pc), gc_net.v, 1))
    #     pc_ac = tf.linalg.normalize((z/tf.reduce_sum(z))**10)[0].numpy()
    #     # print(pc_ac.shape, hmap_x.shape, hmap_z.shape, z.shape, tf.tensordot(tf.transpose(gc_net.w_gc_pc), gc_net.v, 1).shape)
    #     ax.tricontourf(hmap_x, hmap_y, np.nan_to_num(pc_ac.flatten()), cmap=cmap)
    #     ax.set_aspect('equal')
    #     rad = np.deg2rad(90)
    #     v = np.array([np.cos(rad), np.sin(rad)])
    #     hdv = head_direction(0, v)
    #     gc_net(hdv, tf.tensordot(tf.transpose(gc_net.w_gc_pc), gc_net.v, 1)[:, 0], False)
    #     plot.title(tf.reduce_max(gc_net.v).numpy())
    #     plot.pause(.1)

    # for i in range(50):
    #     plot.figure(1); plot.clf()
    #     plot.subplot(211)
    #     plot.imshow(gc_net.w_gc_gc[i].numpy().reshape(num_gcs, num_gcs))
    #     plot.subplot(212)
    #     plot.imshow(gc_net.w_gc_gc[:, i].numpy().reshape(num_gcs, num_gcs))
    #     plot.pause(.1)
    # exit()

    # print(hmap_g[0].shape)  #, tf.reduce_max(hmap_g[-1]))
    
    # j = 0
    # for img in hmap_g:
    #     fig = plot.figure(1); plot.clf()
    #     plot.subplot(211)
    #     plot.imshow(img.reshape((num_gcs, num_gcs)))
    #     plot.title('Number ' + str(j))
    

   


    
    # currPos = np.dot()
    # plot.imshow(gc_net.w_gc_gc[0].numpy().reshape((11, 11)))
    # plot.show()
    # exit()

    # plot.stem(pc_net.v)
    # ax = plot.subplot(111)
    # ax.tricontourf(hmap_x, hmap_y, np.nan_to_num(hmap_g[:, -1].flatten()), cmap=cmap)
    # ax.set_aspect('equal')
    # plot.show()
    # exit()
    # probeOne()
    # probeOne(1869)
    # exit()

    # conjuctiveGCs(0)
    plot.show()