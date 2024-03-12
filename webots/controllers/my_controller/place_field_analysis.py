import pickle
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
from matplotlib import cm, rcParams
from statsmodels.stats.weightstats import DescrStatsW
from pygeostat.statistics.utils import weighted_mean, weighted_skew, weighted_kurtosis
import plotly.express as px
import matplotlib.patches as patches
import pandas as pd
rcParams.update({
                    'legend.fontsize': 'x-large',
                    'axes.labelsize': 'x-large',
                    'axes.titlesize':'x-large',
                    'xtick.labelsize':'x-large',
                    'ytick.labelsize':'x-large'
                })

# sc = pd.read_csv('square_maze_centers.csv', index_col=0).dropna()
# cc = pd.read_csv('circle_maze_centers.csv', index_col=0).dropna()

# 

with open('hmap_x.pkl', 'rb') as f:
    hmap_x = pickle.load(f)[10:]
with open('hmap_y.pkl', 'rb') as f:
    hmap_y = pickle.load(f)[10:]
with open('hmap_z.pkl', 'rb') as f:
    hmap_z = np.asarray(pickle.load(f))[10:]


cmap = cm.get_cmap('jet')
img1 = plot.imread("robot_arena_four.PNG") #  
img2 = plot.imread("maze_circle.png") #

class Record:

    def __init__(self, s):
        self.x = None
        self.y = None
        self.z = None
        self.s = s
        self.means = None
        self.stdev = None
        self.skews = None
        self.kurtosis = None
        self.pf_formed = None
    
    def getStats(self, show=False, ax=None):
        # get individual pc stats to detrmine place field formation
        self.means = np.empty([self.z.shape[-1], 2])
        self.stdev = np.empty([self.z.shape[-1], 2])
        self.skews = np.empty([self.z.shape[-1], 2])
        self.kurtosis = np.empty([self.z.shape[-1], 2])

        for i in range(self.z.shape[-1]):
            x, y = DescrStatsW(self.x, weights=self.z[:, i]), DescrStatsW(self.y, weights=self.z[:, i])
            try:
                self.means[i] = weighted_mean(self.x, wts=self.z[:, i]), weighted_mean(self.y, wts=self.z[:, i]) # x.mean, y.mean
                self.skews[i] = weighted_skew(self.x, wts=self.z[:, i]), weighted_skew(self.y, wts=self.z[:, i])
                self.kurtosis[i] = weighted_kurtosis(self.x, wts=self.z[:, i]) - 3, weighted_kurtosis(self.y, wts=self.z[:, i]) - 3
            except:
                self.means[i] = np.nan, np.nan
                self.skews[i] = np.nan, np.nan
                self.kurtosis[i] = np.nan, np.nan
            self.stdev[i] = x.std, y.std

        # mod_skew = np.all(np.less_equal(np.abs(self.skews), 0.5), -1)
        # self.pf_formed = np.greater_equal(self.z.T.max(-1), .1) # np.logical_and(np.greater_equal(self.z.T.max(-1), .1), np.logical_and(np.isfinite(self.stdev.max(-1)), mod_skew)) # np.less_equal(self.stdev.max(-1), 1))

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



    def probe(self, i=None, ax=None, img=img1):
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
        # plot.colorbar(cntr)

r = Record(.25)
r.x = hmap_x
r.y = hmap_y
r.z = hmap_z
r.getStats()

records = []

with open('square_records_5_1.pickle', 'rb') as f:
    records.append(pickle.load(f))
# with open('square_records_5_2.pickle', 'rb') as f:
#     records.append(pickle.load(f))
# with open('circle_records_5_1.pickle', 'rb') as f:
#     records.append(pickle.load(f))
# with open('circle_records_5_2.pickle', 'rb') as f:
#     records.append(pickle.load(f))
with open('cross_records_5_1.pickle', 'rb') as f:
    records.append(pickle.load(f))
# with open('cross_records_5_2.pickle', 'rb') as f:
#     records.append(pickle.load(f))

pf_f_0 = pd.DataFrame(records[0].z).max().gt(.2)
pf_f_1 = pd.DataFrame(records[1].z).max().gt(.2)

# print("Square", pf_f_0.sum())
# print("Circle", pf_f_1.sum())
# exit()
# print("Squares", new_square_z.sum())
# print("Circles", new_circle_z.max().gt(.2).sum())
# exit()

# plot.figure(dpi=200, figsize=(7, 11))
# plot.subplot(211)
# plot.hist(new_square_z.max(), bins=20)
# plot.subplot(212)
# plot.hist(new_circle_z.max(), bins=20)
# plot.show()
# exit()
# pf_f_0 = pd.DataFrame(r.z).rolling(10).mean().max().gt(.2)
# print(pf_f_0.sum())
# # records.append(r)
# with open('cross_records_5_3.pickle', 'wb') as f:
#     # records = pickle.load(f)
#     pickle.dump(r, f)
# exit()

# pf_f_0 = np.greater_equal(records[0].z.T.max(-1), .2)
# pf_f_1 = np.greater_equal(records[1].z.T.max(-1), .2)
# columns=['x center', 'y center', 'x stdev', 'y stdev', 'x kurtosis', 'y kurtosis', 'x skewness', 'y skewness', 'Maze']

# # records[0].getStats(show=False)
# # records.append(r)

# sc = pd.DataFrame({"x_1":records[0].means[:, 0][pf_f_0], "y_1":records[0].means[:, 1][pf_f_0], "x_2":records[1].means[:, 0][pf_f_0], "y_2":records[1].means[:, 1][pf_f_0]})
# sc.dropna(inplace=True)

# cc = pd.DataFrame({"x_1":records[2].means[:, 0][pf_f_1], "y_1":records[2].means[:, 1][pf_f_1], "x_2":records[3].means[:, 0][pf_f_1], "y_2":records[3].means[:, 1][pf_f_1]})
# cc.dropna(inplace=True)

# f = plot.figure(dpi=200, figsize=(8,11))
# a = f.add_subplot(111)
# a.set_frame_on(False)
# a.set_xticks([])
# a.set_yticks([])
# a.set_ylabel("Count", labelpad=25)
# a.set_xlabel("Euclidean distance between place field peaks across runs", labelpad=40)
# ax1 = f.add_subplot(211)
# sns.histplot(np.linalg.norm(np.subtract(sc[['x_1', 'y_1']], sc[['x_2', 'y_2']]), axis=1), stat='count', bins=20, ax=ax1, binrange=(0,0.5))#, bins=np.arange(0,3,.1), binrange=(0,3))
# ax1.set_ylabel(None)
# ax2 = f.add_subplot(212)
# sns.histplot(np.linalg.norm(np.subtract(cc[['x_1', 'y_1']], cc[['x_2', 'y_2']]), axis=1), stat='count', bins=20, ax=ax2, binrange=(0,0.5)) #, bins=np.arange(0,3,.1), binrange=(0,3))
# ax2.set_ylabel(None)
# plot.savefig("both_stability.png")

# # plot.savefig("pc_shift.png")
# print("Square mean", np.linalg.norm(np.subtract(sc[['x_1', 'y_1']], sc[['x_2', 'y_2']]), axis=1).max())
# print("Circle mean", np.linalg.norm(np.subtract(cc[['x_1', 'y_1']], cc[['x_2', 'y_2']]), axis=1).max())
# exit()

# exit()
# records[0].getStats(show=False)
# records[1].getStats(show=False)

# f = plot.figure(dpi=200, figsize=(8,11))
# i = 1

# # for n, m in zip(np.random.choice(np.arange(1000)[pf_f_0], 3), np.random.choice(np.arange(1000)[pf_f_1], 3)):
# for n in np.random.choice(np.arange(1000)[pf_f_0], 3):
#     ax1 = f.add_subplot(3,1,i)
#     records[0].probe(n, ax=ax1, img=img1)
#     # ax2 = f.add_subplot(3,2,i+1)
#     # records[1].probe(m, ax=ax2, img=img2)
#     i += 1
# # plot.savefig("open_pfs.png")
# # exit()


# records.getStats(show=False)
# r_df = r_df.append()
# records[1].probe(646)
# plot.show()
# exit()

df_1 = pd.DataFrame({'x center':records[0].means[:, 0][pf_f_0], 'y center':records[0].means[:, 1][pf_f_0], 'x stdev':records[0].stdev[:, 0][pf_f_0], 'y stdev':records[0].stdev[:, 1][pf_f_0],
              'x kurtosis':records[0].kurtosis[:, 0][pf_f_0], 'y kurtosis':records[0].kurtosis[:, 1][pf_f_0], 'x skewness':records[0].skews[:, 0][pf_f_0], 'y skewness':records[0].skews[:, 1][pf_f_0],
               'Maze':'square'})
print(df_1.describe())
df_2 = pd.DataFrame({'x center':records[1].means[:, 0][pf_f_1], 'y center':records[1].means[:, 1][pf_f_1], 'x stdev':records[1].stdev[:, 0][pf_f_1], 'y stdev':records[1].stdev[:, 1][pf_f_1],
              'x kurtosis':records[1].kurtosis[:, 0][pf_f_1], 'y kurtosis':records[1].kurtosis[:, 1][pf_f_1], 'x skewness':records[1].skews[:, 0][pf_f_1], 'y skewness':records[1].skews[:, 1][pf_f_1],
               'Maze':'circle'})
print(df_2.describe())
pot_rep = (records[0].stdev[:, 0]>1) | (records[0].stdev[:, 1]>1) & pf_f_0
print(sum(pot_rep))
# r_df = df_1
# double = (327, 323, 946) # , 453)

# f=plot.figure(dpi=200, figsize=(8,12))
# np.random.seed(int(time.time()))
# i=1
# m_s = (857, 927, 344, 25)
# n_s = (537, 770, 642)
# for n, m in zip(n_s, m_s): # np.random.choice(np.arange(1000)[pf_f_0], 3, replace=False), np.random.choice(np.arange(1000)[pot_rep], 3, replace=False)):
#     print(n, m)
#     # for n in np.random.choice(np.arange(1000)[pf_f_0], 3):
#     ax1 = f.add_subplot(3,2,i)
#     records[0].probe(n, ax=ax1, img=img1)
#     ax2 = f.add_subplot(3,2,i+1)
#     records[0].probe(m, ax=ax2, img=img1)
#     i += 2
# plot.savefig("cross_pfs.png")
# exit()

# r_df = pd.concat([df_1, df_2], ignore_index=True)

# r.getStats(show=False)

# x_diff = pd.Series(r.means[:, 0])[pf_f_1] - df_2['x center']
# y_diff = pd.Series(r.means[:, 1])[pf_f_1] - df_2['y center']

# print(sum(pf_f))
# circle_maze_centers = pd.read_csv("square_maze_centers.csv")
# circle_maze_centers["x_3"] = records.means[:, 0][pf_f]
# circle_maze_centers["y_3"] = records.means[:, 1][pf_f] # , 'x_2':r.means[:, 0][pf_f], 'y_2':r.means[:, 1][pf_f] })
# circle_maze_centers.to_csv("square_maze_centers.csv")
# exit()
# dist = [np.linalg.norm(np.subtract((pd.Series(r.means[:, 0])[pf_f_1].iloc[i], pd.Series(r.means[:, 1])[pf_f_1].iloc[i]), (df_2['x center'].iloc[i], df_2['y center'].iloc[i]))) for i in range(pf_f_1.sum()) ]

# sns.histplot(x_diff)
# sns.histplot(y_diff)
# plot.figure()
# sns.histplot(dist)
# plot.show()
# exit()

print("Square good skew", df_1[(df_1['x skewness'].abs()<.8) & (df_1['y skewness'].abs()<.8)].shape)
print("Circle good skew", df_2[(df_2['x skewness'].abs()<.8) & (df_2['y skewness'].abs()<.8)].shape)
print("Good square", df_1[(df_1['x skewness'].abs()<.8) & (df_1['y skewness'].abs()<.8) & (df_1['x kurtosis'].abs()<3) & (df_1['y kurtosis'].abs()<3)].shape)
print("Good circle", df_2[(df_2['x skewness'].abs()<.8) & (df_2['y skewness'].abs()<.8) & (df_2['x kurtosis'].abs()<3) & (df_2['y kurtosis'].abs()<3)].shape)
print("Square good kurtosis", df_1[(df_1['x kurtosis'].abs()<3) & (df_1['y kurtosis'].abs()<3)].shape)
print("Circle good kurtosis", df_2[(df_2['x kurtosis'].abs()<3) & (df_2['y kurtosis'].abs()<3)].shape)

print(df_2[(df_2['x stdev']<0.5) & (df_2['y stdev']<0.5)].describe())
f=plot.figure(dpi=400, figsize=(8,11))
a=f.add_subplot(111)
a.set_frame_on(False)
a.set_xticks([])
a.set_yticks([])
a.set_ylabel('y-axis standard deviation')
a.set_xlabel('x-axis standard deviation', labelpad=30)
ax1=f.add_subplot(211)
ax1.set(aspect='equal', xlabel=None, ylabel=None)
plot.hist2d(x=df_2['x stdev'], y=df_2['y stdev'], cmap=cmap, bins=50)
plot.colorbar() #, range=((0,0.5), (0,0.5)))
ax2 =f.add_subplot(212)
ax2.set(aspect='equal', xlabel=None, ylabel=None)
plot.hist2d(x=df_2['x stdev'], y=df_2['y stdev'], cmap=cmap, bins=25, range=((0,0.5), (0,0.5)))
plot.colorbar()
plot.savefig('cross_stdev_zoom.png')
exit()

# r_df = r_df[(r_df['x skewness'].abs()<.8) & (r_df['y skewness'].abs()<.8) & (r_df['x kurtosis'].abs()<3) & (r_df['y kurtosis'].abs()<3)]
# print(df_2[(df_2.x skewness.abs()<.8) & (df_2.y skewness.abs()<.8) & (df_2.x kurtosis.abs()<3) & (df_2.y kurtosis.abs()<3)&(df_2.x stdev>1)])
# print(r_df.describe())
# exit()

f = plot.figure(figsize=(11,6), dpi=400)
# a = f.add_subplot(111)
# a.set_frame_on(False)
# a.set_xticks([])
# a.set_yticks([])
# a.set_xlabel("Maximum firing rates", labelpad=25)
# a.set_ylabel("Frequency", labelpad=40)
ax1 = f.add_subplot(111)
# ax1.set_title("(a)", loc="left", fontdict={'fontsize':18})
ax1.xaxis.set_visible(True)
ax1.hist(pd.DataFrame(records[0].z).max(), bins=np.arange(.8, step=.05), edgecolor='black')
ax1.axvline(0.2, c='r')
ax1.set_xlabel("Maximum firing rates")
ax1.set_ylabel("Frequency")
# plot.show()
# exit()
# ax2 = f.add_subplot(212)
# # ax2.set_title("(b)", loc="left", fontdict={'fontsize':18})
# ax2.xaxis.set_visible(True)
# ax2.hist(pd.DataFrame(records[1].z).max(), bins=np.arange(.8, step=.05), edgecolor='black')
# ax2.axvline(0.2, c='r')
plot.savefig("crossMaxRates.png")
# # exit()


fig = plot.figure(figsize=(6,6), dpi=400)
# a = fig.add_subplot(111)
# a.set_frame_on(False)
# a.set_xticks([])
# a.set_yticks([])
plot.title("Place Field Centers", pad=25)
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.scatter(df_1['x center'], df_1['y center'], s=8, c='darkblue', marker='x')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_ylim(5.25, -5.25)
ax.set_xlim(5.25, -5.25)
# ax.set_title("Place Field Centers")
ax.imshow(img1, extent=[5.25, -5.25, 5.25, -5.25], origin='lower')
# plot.savefig("cross_centers.png")
# ax = fig.add_subplot(212)
# ax.set_aspect('equal')
# ax.scatter(df_2['x center'], df_2['y center'], s=8, c='darkblue', marker='x')
# ax.imshow(img2, extent=[5.25, -5.25, 5.25, -5.25], origin='lower')
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
# ax.set_ylim(5.25, -5.25)
# ax.set_xlim(5.25, -5.25)
# plot.scatter(records[1].x, records[1].y, c='r', s=.2, alpha=.4)
# plot.colorbar()
# plot.show()
plot.savefig("cross_centers.png")


# plot.figure()
# plot.subplot(211)
# plot.hist(df_1['x stdev'], alpha=.6)
# plot.hist(df_1['y stdev'], alpha=.6)
# plot.legend(['x-axis standard deviation', 'y-axis standard deviation'])
# plot.subplot(212)
# plot.hist(df_2['x stdev'], alpha=.6)
# plot.hist(df_2['y stdev'], alpha=.6)
# plot.legend(['x-axis standard deviation', 'y-axis standard deviation'])
# plot.show()

# plot.figure()
# plot.subplot(211)
# plot.hist(df_1['x skewness'], alpha=.6)
# plot.hist(df_1['y skewness'], alpha=.6)
# plot.legend(['x-axis skew', 'y-axis skew'])
# plot.subplot(212)
# plot.hist(df_2['x skewness'], alpha=.6)
# plot.hist(df_2['y skewness'], alpha=.6)
# plot.legend(['x-axis skew', 'y-axis skew'])

# plot.figure()
# plot.subplot(211)
# plot.hist(df_1['x kurtosis'], alpha=.6)
# plot.hist(df_1['y kurtosis'], alpha=.6)
# plot.legend(['x-axis standard deviation', 'y-axis standard deviation'])
# plot.subplot(212)
# plot.hist(df_2['x kurtosis'], alpha=.6)
# plot.hist(df_2['y kurtosis'], alpha=.6)
# plot.legend(['x-axis kurtosis', 'y-axis kurtosis'])
# plot.show()

# # df_1.plot(x='x center', y='y center', kind="scatter")

# # jointplot(data=df_1, x='x center', y='y center')
# # jointplot(data=df_2, x='x center', y='y center')
# 

# g = sns.JointGrid()
# sns.scatterplot(data=r_df[r_df["Maze"]=='square'], x='x stdev', y='y stdev', s=100, linewidth=1.5, ax=g.ax_joint)
# sns.kdeplot(data=r_df[r_df["Maze"]=='square'], x='x stdev', linewidth=2, ax=g.ax_marg_x)
# sns.kdeplot(data=r_df[r_df["Maze"]=='square'], y='y stdev', linewidth=2, ax=g.ax_marg_y)

f = plot.figure(dpi=200, figsize=(8,8))
plot.title("Place Field Standard Deviations")
a = f.add_subplot(111)
# a.set_frame_on(False)
# a.set_xticks([])
# a.set_yticks([])
# ax1 = f.add_subplot(211)
# plot.scatter(x=df_1['x stdev'], y=df_1['y stdev'], s=15, alpha=.8)
# rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
# ax1.add_patch(rect)
# plot.colorbar()
# rect = patches.Rectangle((0, 0), .4, .4, linewidth=1, edgecolor='r', facecolor='none')
# ax1.add_patch(rect)
# # sns.kdeplot(data=r_df[r_df["Maze"]=='square'], x='x stdev', y='y stdev', ax=ax1)
a.set(aspect='equal', xlabel=None, ylabel=None)
# ax2 = f.add_subplot(212)
plot.hist2d(x=df_1['x stdev'], y=df_1['y stdev'], cmap=cmap, bins=100)
plot.colorbar()
# ax2.set(aspect='equal', xlabel=None, ylabel=None)
# a.set_xlabel('x-axis standard deviation',labelpad=20)
# a.set_ylabel('y-label standard deviation', labelpad=5)
# a.set_title('Distribution of standard deviations in open mazes', pad=20)
# plot.show()
plot.savefig("cross_stdev.png")
# g.plot_joint(sns.scatterplot, s=100, alpha=.5)
# g.plot_marginals(sns.histplot, kde=True)

# f = plot.figure(dpi=200, figsize=(8, 12))
# plot.title("Circular Maze Place Field Standard Deviations")
# a = f.add_subplot(111)
# a.set_frame_on(False)
# a.set_xticks([])
# a.set_yticks([])
# ax1 = f.add_subplot(211)
# sns.scatterplot(data=r_df[r_df["Maze"]=='circle'], x='x stdev', y='y stdev', ax=ax1, s=6, alpha=.8)
# rect = patches.Rectangle((0, 0), .4, .4, linewidth=1, edgecolor='r', facecolor='none')
# ax1.add_patch(rect)
# # ax1.axvline(0.4, c='r')
# # ax1.axhline(0.4, c='r')
# # ax1.axhline(0.05, c='r')
# # sns.kdeplot(data=r_df[r_df["Maze"]=='circle'], x='x stdev', y='y stdev', ax=ax1)
# ax1.set(aspect='equal', xlabel=None, ylabel=None)
# ax2 = f.add_subplot(212)
# sns.histplot(data=r_df[r_df["Maze"]=='circle'], x='x stdev', y='y stdev', thresh=None, cbar=True, ax=ax2, binrange=((0, .4),(0, .4)), cmap="viridis")
# ax2.set(aspect='equal', xlabel=None, ylabel=None)
# # plot.show()
# # plot.savefig("circle_stdev.png")
# # g.plot_joint(sns.scatterplot, s=100, alpha=.5)
# # g.plot_marginals(sns.histplot, kde=True)

# fig = plot.figure(dpi=200, figsize=(5,8))
# ax1 = fig.add_subplot(211)
# sns.jointplot(data=r_df[r_df["Maze"]=='square'], x='x stdev', y='y stdev', kde=True) #, levels=100) #, kind='hist')
# # plot.figure(dpi=200, figsize=(5,8))
# ax2 = fig.add_subplot(212)
# sns.jointplot(data=r_df[r_df["Maze"]=='circle'], x='x stdev', y='y stdev', kde=True)

# plot.show()


# plot.figure(dpi=200, figsize=(11,8))
# plot.subplot(212)
# plot.title("Circular Maze")
# sns.histplot(data=r_df[r_df["Maze"]=='circle'], x='x stdev', y='y stdev', fill=True) #, levels=100) #, kind='hist')
# plot.subplot(211)
# plot.title("Square Maze")
# sns.histplot(data=r_df[r_df["Maze"]=='square'], x='x stdev', y='y stdev', fill=True) #, kind='hist')
# plot.show()
# exit()
# # g = sns.JointGrid(data=r_df, x='x kurtosis', y='y kurtosis', hue='Maze')
# # g.plot_joint(sns.histplot)
# # g.plot_marginals(sns.boxplot)
f=plot.figure(dpi=200, figsize=(8,12))
a = f.add_subplot(111)
a.set_frame_on(False)
a.set_xticks([])
a.set_yticks([])
a.set_ylabel("y-axis kurtosis", labelpad=20)
a.set_xlabel("x-axis kurtosis", labelpad=20)
a.set_title("Distribution of excess kurtosis in non-trivial maze", pad=10)
ax2=f.add_subplot(212)
plot.hist2d(x=df_1['x kurtosis'], y=df_1['y kurtosis'], cmap=cmap, range=((-5,5), (-5,5)), bins=50)
plot.colorbar()
# # ax2 = sns.kdeplot(data=df_2, x='x kurtosis', y='y kurtosis', fill=True) #, kind='hist', xlim=(-5, 5), ylim=(-5, 5), alpha=.6)
# ax2.set(xlabel=None, ylabel=None, title=None, aspect='equal')
ax1=f.add_subplot(211)
plot.scatter(x=df_1['x kurtosis'], y=df_1['y kurtosis'], s=15, alpha=.8) # , bins=50)
rect = patches.Rectangle((-5, -5), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
ax1.add_patch(rect)
# # ax1 = sns.kdeplot(data=df_1, x='x kurtosis', y='y kurtosis', fill=True) #, kind='hist', xlim=(-5, 5), ylim=(-5, 5), alpha=.6)
ax1.set(xlabel=None, ylabel=None, aspect='equal')
# plot.colorbar()
plot.savefig("cross_kurtosis.png")

# g = s       ns.JointGrid(data=r_df, x='x skewness', y='y skewness', hue='Maze')
# g.plot_joint(sns.histplot)
# g.plot_marginals(sns.boxplot)
f=plot.figure(dpi=200, figsize=(6,6))
a = f.add_subplot(111)
a.set_frame_on(False)
a.set_xticks([])
a.set_yticks([])
a.set_ylabel("y-axis skewness", labelpad=20)
a.set_xlabel("x-axis skewness", labelpad=20)
a.set_title("Distribution of skewness in non-trivial maze", pad=60)
# ax_1 = f.add_subplot(212)
plot.hist2d(data=df_1, x='x skewness', y='y skewness', cmap=cmap, range=((-7, 7), (-7,7)), bins=50)
a.set_aspect('equal')
plot.colorbar()
# # ax2 = sns.kdeplot(data=df_2, x='x skewness', y='y skewness', fill=True) #, kind='hist', xlim=(-5, 5), ylim=(-5, 5), alpha=.6)
# ax2.set(xlabel=None, ylabel=None, title=None, aspect='equal')
# ax_2 = f.add_subplot(211)
# # # ax1 = sns.kdeplot(data=df_1, x='x skewness', y='y skewness', fill=True) #, kind='hist', xlim=(-5, 5), ylim=(-5, 5), alpha=.6)
# plot.scatter(data=df_1, x='x skewness', y='y skewness') #, cmap=cmap, bins=50)
# ax_2.set(xlabel=None, ylabel=None, aspect='equal')
# rect = patches.Rectangle((-7, -7), 14, 14, linewidth=1, edgecolor='r', facecolor='none')
# ax_2.add_patch(rect)
# plot.colorbar()
# plot.show()
plot.savefig('cross_skewness.png')
# plot.show()
exit()
# for r in records:
#     r.getStats(show=False)
#     pf_formed = np.greater_equal(r.z.T.max(-1), .2)

#     plot.figure()
#     plot.hist2d(r.means[:, 0][pf_formed], r.means[:, 1][pf_formed], vmin=0, bins=5, cmap=cmap)
#     plot.title("Place Field Center")
#     plot.colorbar()

#     plot.figure()
#     plot.scatter(r.stdev[:, 0][pf_formed], r.stdev[:, 1][pf_formed])
#     plot.title("Standard deviation")
#     # plot.colorbar()

#     plot.figure()
#     plot.scatter(r.kurtosis[:, 0][pf_formed] - 3, r.kurtosis[:, 1][pf_formed] - 3)
#     plot.title("Excess Kurtosis")
#     # plot.colorbar()

#     # plot.figure()
#     jointplot(x=r.skews[:, 0][pf_formed], y=r.skews[:, 1][pf_formed])#.set(title="Skewness")
#     # plot.title("Skewness")
#     # plot.colorbar()

#     ### max firing of place cells
#     # plot.figure()
#     # max_firing = r.z.T.max(-1)
#     # plot.hist(max_firing[pf_formed])
#     plot.show()

# # h, x, y, i = plot.hist2d(r.means.T[0][r.pf_formed], r.means.T[1][r.pf_formed])

# # fig = go.Figure(data=[go.Surface(z=h, x=x, y=y)])
# # fig.update_layout(title='Place Cell Distribution', autosize=False,
# #                   width=500, height=500,
# #                   margin=dict(l=65, r=50, b=65, t=90))
# # fig.show()