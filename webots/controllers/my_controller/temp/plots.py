import matplotlib.pyplot as plot 
import pandas as pd
import numpy as np
import seaborn as sns

# eff = pd.read_csv('efficiencies5.csv', header=0, index_col=0, delimiter='\t').T
# eff.columns = range(1, 5)
# print(eff.head())
# fig, axes = plot.subplots(dpi=200)

# sns.boxplot(data=eff, ax=axes)
# # axes.set_yscale("log")
# axes.set_title('Path Indirectness Over Runs in Maze 1')
# axes.yaxis.grid(which="Both")
# axes.set_xlabel('Run')
# axes.set_ylabel('Path Indirectness')
# plot.savefig("path_indirectness.png")
# plot.show()

dve = pd.read_csv('dve.csv', header=0, delimiter='\t')
print(dve)
print(dve.corr())
z = np.polyfit(x=dve.iloc[:, 0], y=dve.iloc[:, 1], deg=1)
p = np.poly1d(z)
dve['trendline'] = p(dve.iloc[:, 0])
plot.figure(dpi=200)
plot.scatter(dve.iloc[:, 0], dve.iloc[:, 1], marker='x')
# plot.plot(dve.iloc[:, 0], dve.iloc[:, 2])
plot.grid(True)
plot.xlabel("Distance Covered on First Run (m)")
plot.ylabel("Indirectness of Second Run")
plot.title("Distance Covered on Run 1 vs Indirectness of Run 2")
plot.savefig("distance_vs_indirectness.png")
plot.show()

