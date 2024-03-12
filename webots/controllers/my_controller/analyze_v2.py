# seems std for no place field formation is 0. Verify 
 
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
from scipy.spatial.distance import jaccard, cdist

cmap = cm.get_cmap('jet') # 'plasma')
rng = default_rng()
contxt = 0
rcParams.update({'font.size': 20})

with open('pcn.pkl', 'rb') as f:
    pc_net = pickle.load(f)

img = plot.imread("maze_one.PNG") # ("maze_circle.png") # 

scales = (25, 50, 100)


class Record:

    def __init__(self, s):
        self.x = None
        self.y = None
        self.z = None
        self.s = s
        self.means = None
        self.stdev = None
        self.pf_formed = None
    
    def getStats(self, show=False, ax=None):
        # get individual pc stats to detrmine place field formation
        self.means = np.empty([self.z.shape[-1], 2])
        self.stdev = np.empty([self.z.shape[-1], 2])
        print("Means shape", self.means.shape)

        for i in range(self.z.shape[-1]):
            x, y = DescrStatsW(self.x, weights=self.z[:, i]), DescrStatsW(self.y, weights=self.z[:, i])
            self.means[i] = x.mean, y.mean
            self.stdev[i] = x.std, y.std

        self.pf_formed = np.logical_and(np.isfinite(self.stdev.max(-1)), np.less_equal(self.stdev.max(-1), 1))

        if show:
            if not ax:
                plot.figure()
                plot.hist2d(self.stdev[self.pf_formed][:, 0], self.stdev[self.pf_formed][:, 1], bins=50, alpha=.4)
                plot.colorbar()
                return
                fig = plot.figure()
                ax = fig.add_subplot(111) 
                ax.set_aspect('equal') 
            ax.hist2d(self.stdev[self.pf_formed][:, 0], self.stdev[self.pf_formed][:, 1], bins=50, alpha=.4)
            ax.colorbar()



    def probe(self, i=None, ax=None):
        if not i:
            i = np.random.choice(np.arange(self.z.shape[-1])[self.pf_formed])
        if not ax:
            fig = plot.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
        params = {'mathtext.default': 'regular' }          
        plot.rcParams.update(params)
        cntr = ax.hexbin(self.x, self.y, np.nan_to_num(self.z[:, i].flatten()), 50, cmap=cmap, alpha=.6)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_ylim(5.25, -5.25)
        ax.set_xlim(5.25, -5.25)
        ax.imshow(img, extent=[5.25, -5.25, 5.25, -5.25], origin='lower')

records = []
fig = plot.figure()
a = fig.add_subplot(111)
a.set_ylabel("Frequency")
a.set_xlabel("Place Field Size")

fig2 = plot.figure()
for s, i in zip(scales, range(3)):
    record = Record(s)
    with open('hmap_x{s}.pkl'.format(s=s), 'rb') as f:
        record.x = pickle.load(f)[10:]
    with open('hmap_y{s}.pkl'.format(s=s), 'rb') as f:
        record.y = pickle.load(f)[10:]
    with open('hmap_z{s}.pkl'.format(s=s), 'rb') as f:
        record.z = pickle.load(f)[10:]
    record.getStats(True)

    for j in range(1, 4):
        a2 = fig2.add_subplot(3, 3, (i*3)+j)
        record.probe(ax=a2)

a.legend(scales)
plot.show()
records.append(record)

# todo: use records instead of hmap and co

hmap_z = np.zeros((3, 1000, hmap_x.shape[0]))   # 3 contexts by 1000 pcs by number of timesteps
for s, c in zip((25, 50, 100), range(3)):
    with open('hmap_z{s}.pkl'.format(s=s), 'rb') as f:
        a = np.asarray(pickle.load(f))
        print(a.shape)
        continue
        hmap_z[c] = a[10:].T
exit()
def getStatistics(z):
    return DescrStatsW(hmap_x, weights=z), DescrStatsW(hmap_y, weights=z)


corr = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        corr[i, j] = jaccard(pf_formed[i], pf_formed[j])

print("Place fields formed: ", pf_formed.sum(-1))

# print("Correlation Coefficients:\n", corr)
# fig, ax = plot.subplots(dpi=200, figsize=(12,8))
# im = ax.imshow(corr, cmap="plasma", vmax=0, vmin=1)
# fig.colorbar(im)
# plot.xticks(range(3), [1, 2, 3])
# plot.yticks(range(3), [1, 2, 3])

# ax.set_ylabel("Context")
# ax.set_xlabel("Context")
# for (j, i), label in np.ndenumerate(corr):
#     ax.text(i, j, np.around(label, 2), ha='center', va='center', color='white', fontweight='bold')
# # plot.title("Jaccard Distance of Place Field Formation\nAcross the 3 Contexts")
# plot.savefig("jaccard_distances.png")

# f = plot.figure(dpi=200, figsize=(12,8))
# im = plot.imshow(pf_formed[:, :100], cmap='Greys', aspect='auto')
# f.colorbar(im)
# # plot.xticks(range(20))
# plot.xlabel("Place Cell Index")
# plot.yticks(range(3), [1, 2, 3])
# plot.xticks([20, 40, 5.250, 80])
# plot.ylabel("Context")
# # plot.title("Place Field Formation in the First 100 Neurons Across the 3 Contexts", pad=20)
# plot.savefig("place_field_formation.png")


# Calculate distances between pf centers where both pcs have pfs and then create histogram - 3*3 symmetric
def centers_dist(i, j, ax):
    both = pf_formed[[i, j]].all(0)
    i_c = means[i, both]
    j_c = means[j, both]
    dists = cdist(i_c, j_c, 'euclidean').diagonal()
    ax.hist(dists, edgecolor='black', bins=range(12))
    # ax.set_title("(Context {i} x Context {j}".format(i=i+1, j=j+1), fontdict={'fontsize':15})
    # plot.xlabel("Euclidean Distance (m$^2$)")
    # plot.ylabel("Frequency")

    
# f = plot.figure(figsize=(10, 10), dpi=200)
# a = f.add_subplot(111)
# # a.xaxis.set_visible(False)
# # a.yaxis.set_visible(False)
# a.set_frame_on(False)
# a.set_xticks([])
# a.set_yticks([])
# a.set_xlabel("Euclidean Distance (m$^2$)", labelpad=25)
# a.set_ylabel("Frequency", labelpad=40)
# ax1 = f.add_subplot(311)
# ax1.set_title("(a)", loc="left", fontdict={'fontsize':18})
# ax1.xaxis.set_visible(False)
# centers_dist(0, 1, ax1)
# ax2 = f.add_subplot(312)
# ax2.set_title("(b)", loc="left", fontdict={'fontsize':18})
# ax2.xaxis.set_visible(False)
# centers_dist(0, 2, ax2)
# ax3 = f.add_subplot(313)
# ax3.set_title("(c)", loc="left", fontdict={'fontsize':18})
# centers_dist(1, 2, ax3)
# plot.savefig("distances_histogram.png")

#  exit()

# # print(one_two.shape, one_two.sum())
# exit()
# for c in range(3):
#     pf_formed[c] = np.isfinite(stdev[c].max(-1))
#     plot.subplot(1, 3, c+1)
#     # plot.hist2d(means[c, p, 0], means[c, p, 1], range=[[-5, 5], [-5, 5]])# stdev[c].max(-1).tolist())
#     # plot.colorbar()
#     nans = np.isnan(stdev[c].max(-1))
#     plot.hist(stdev[c, ~nans].max(-1))
#     plot.title(1000-nans.sum())
#     # plot.hist(stdev[c].max(-1))
# plot.show()



def probeOne(probed, ax=None):
    if not ax:
        fig = plot.figure()
        ax = fig.add_subplot(111)   # , projection='3d')
        ax.set_aspect('equal')
        params = {'mathtext.default': 'regular' }          
        plot.rcParams.update(params)
    # print(np.argwhere(np.isnan(hmap_z[:, probed])))
    # ax.plot(hmap_x, hmap_y, color='b', alpha=.8)
    # ax.scatter(hmap_x, hmap_y, s=10*np.nan_to_num(hmap_z[:, probed].flatten()), c=np.nan_to_num(hmap_z[:, probed].flatten()), cmap=cmap, marker='x')
    cntr = ax.hexbin(hmap_x, hmap_y, np.nan_to_num(hmap_z[:, probed].flatten()), 50, cmap=cmap, alpha=.6)
    # , vmin=0, vmax=.35) # marker='*', 
    # goal = plot.Circle((goalLocation), goal_r, color='r', linewidth=2, fill=False)
    # plot_trisurf(hmap_x, hmap_y, reward_function, cmap=cmap)
    # ax.add_patch(goal)
    # ax.plot(hmap_x, hmap_y, linewidth=.5)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(5.25, -5.25)
    ax.set_xlim(5.25, -5.25)
    ax.imshow(img, extent=[5.25, -5.25, 5.25, -5.25], origin='lower')
    # plot.colorbar(cntr)
    # v = "v_{"+str(probed)+"}^p"
    # plot.title("${v}$ shown in context {i}".format(v=v, i=contxt+1))
    # v = "v_{"+str(probed)+"}^b"
    # plot.title("${v}$".format(v=v))
    # plot.savefig("bvc\C{c}BVC{i}.png".format(c=contxt, i=probed))
    # print("Reward value:", rc_net.w_in[contxt, probed])
    # plot.show()


ids = np.where(pf_formed[0]==True)[0]
print(ids)
for i in np.random.choice(ids, 20, False):
    probeOne(i)
plot.show()



f = plot.figure(figsize=(12, 4), dpi=200)
a = f.add_subplot(111)
a.set_frame_on(False)
a.set_aspect('equal')
a.xaxis.set_visible(False)
a.yaxis.set_visible(False)
p = 1
for n in (350, 18, 113):
    ax = f.add_subplot(1, 3, p, adjustable='box', aspect=1)
    # print(p%3, n//3)
    probeOne(n, ax)
    p+=1
plot.savefig("BVC_three_square.png")
plot.show()

# if __name__ == "__main__":    
#     p = [900, 439, 995.25, 719]
#     for n in np.random.randint(1000, size=10): # to_probe: #most_active:
#         probeOne(0, n)
#     plot.show()