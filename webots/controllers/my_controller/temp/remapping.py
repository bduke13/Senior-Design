import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
from statsmodels.stats.weightstats import DescrStatsW
from pygeostat.statistics.utils import weighted_mean, weighted_skew, weighted_kurtosis
from matplotlib import cm, rcParams
rcParams.update({
                    'legend.fontsize': 'x-large',
                    'axes.labelsize': 'x-large',
                    'axes.titlesize':'x-large',
                    'xtick.labelsize':'x-large',
                    'ytick.labelsize':'x-large'
                })

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
        ax.imshow(img1, extent=[5.25, -5.25, 5.25, -5.25], origin='lower')
        plot.colorbar(cntr)


records = []
for n in (2,3):
    with open(f'square_records_{n}.pickle', 'rb') as f:
        r = pickle.load(f)
        r.getStats(show=False)
        records.append(r)

pf_f = pd.DataFrame(records[0].z).rolling(10).mean().max().gt(.2)
sc = pd.DataFrame({'x_1':records[0].means[:, 0][pf_f], 'y_1':records[0].means[:, 1][pf_f],
                   'x_2':records[1].means[:, 0][pf_f], 'y_2':records[1].means[:, 1][pf_f]})
cc = pd.read_csv('circle_maze_centers.csv', index_col=0).dropna()

f = plot.figure(dpi=200, figsize=(8,11))
a = f.add_subplot(111)
a.set_frame_on(False)
a.set_xticks([])
a.set_yticks([])
a.set_ylabel("Count", labelpad=25)
a.set_xlabel("Euclidean distance between place field peaks across runs", labelpad=40)
ax1 = f.add_subplot(211)
sns.histplot(np.linalg.norm(np.subtract(sc[['x_1', 'y_1']], sc[['x_2', 'y_2']]), axis=1), stat='count', kde=True, binwidth=.05, ax=ax1)
ax1.set_ylabel(None)
ax2 = f.add_subplot(212)
sns.histplot(np.linalg.norm(np.subtract(cc[['x_1', 'y_1']], cc[['x_2', 'y_2']]), axis=1), stat='count', kde=True, binwidth=.05, ax=ax2)
ax2.set_ylabel(None)
plot.savefig("pc_shift.png")
print("Square mean", np.linalg.norm(np.subtract(sc[['x_1', 'y_1']], sc[['x_2', 'y_2']]), axis=1).mean())
print("Circle mean", np.linalg.norm(np.subtract(cc[['x_1', 'y_1']], cc[['x_2', 'y_2']]), axis=1).mean())
exit()

