
import pyvisgraph as vg
from shapely.geometry import Polygon
import matplotlib.pyplot as plot
import folium
import numpy as np

def get_vertices(center, size, orientation=0):
    x0, x1 = center[0] - size[0]/2 * np.cos(orientation), center[0] + size[0]/2 * np.cos(orientation)
    y0, y1 = center[1] - size[0]/2 * np.sin(orientation), center[1] + size[0]/2 * np.sin(orientation)
    return [vg.Point(x0, y1), vg.Point(x1, y0)] # vg.Point(x0, y1), vg.Point(x1, y0)]

def plot_environment(polys: list):
    for poly in polys:
        plot.plot([point.x for point in poly], [point.y for point in poly], c='grey')
    plot.gca().invert_yaxis()

plot.figure()
polygon1 = Polygon([(0,5),
                    (1,1),
                    (3,0),
                    ])

polys = [get_vertices([0, -5], [10.3, 0.3], 0), get_vertices([0, 5], [10.3, 0.3], 0), 
            get_vertices([5, 0], [9.7, 0.3], 1.57), get_vertices([-5, 0], [ 9.7, 0.3], 1.57),
            get_vertices([2, 1.27], [3, .3], 0), get_vertices([-3.6, 0], [4, .3], .785)]

plot_environment(polys)
g = vg.VisGraph()
g.build(polys)

start_point, end_point = vg.Point( 1.93, -0.73), vg.Point(-1.5, -3)
shortest_path = g.shortest_path(start_point, end_point)

print(shortest_path) # [point for point in shortest_path])

# shortest distance
l = 0
for n in range(len(shortest_path)-1):
    l += np.linalg.norm(np.array([shortest_path[n+1].y, shortest_path[n+1].x])-np.array([shortest_path[n].y, shortest_path[n].x]))
print("Shortest distance {}".format(l-.6))
# compute using euclidean distance

# Plot of the path 
plot.plot([point.x for point in shortest_path], [point.y for point in shortest_path])
plot.show()
exit()