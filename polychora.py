#!/usr/bin/env python3

'''
Polychora: Python animation of uniform polychora in 4 and higher dimensions

Author: M. Patrick Kelly
Email: patrickyunen@gmail.com
Last Updated: 12-9-2020
'''

from itertools import product, permutations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

dimension = 4
views = {3:-4, 4:-3, 5:-3, 6:-3} #Camera positions for projection
#edge_width = 1 #Hypercube edge width; will be scaled for depth cueing

class Nhypercube():
# Stores vertices and edges of an N-dimensional hypercube
    def __init__(self, dimension=4):
        points = [-1,1]
        self.dimension = dimension
        self.vertices = np.asarray(list(product(points, repeat=dimension)))
        self.edges = [(j,k)
                      for j in range(2**dimension)
                      for k in range(j,2**dimension)
                      if sum(abs(self.vertices[j] - self.vertices[k])) == 2]

class Dodecaplex():
# Stores vertices and edges of a 120-cell or dodecaplex
# Coordinates calculated in advance and saved to file
    def __init__(self, scale=1):
        self.vertices = scale*np.load('dodecaplex.npy')
        self.edges = []
        with open('dodecaplex_edges.txt', 'r') as fh:
            for line in fh.readlines():
                my_tuple = tuple(int(line.split()[i]) for i in range(2))
                self.edges.append(my_tuple)

def get_random_rotation(N, theta):
# Returns N-dimensional rotation matrix of theta radians w/ random orientation

    # Two random vectors define a hyperplane of rotation
    v1 = np.array([np.random.uniform(-1, 1) for i in range(N)])
    v2 = np.array([np.random.uniform(-1, 1) for i in range(N)])

    # Use Gram-Schmidt to orthogonalize these vectors
    u2 = v2 - (np.dot(v1, v2) / np.dot(v1, v1)) * v1

    # Then normalize
    normed1 = v1 / np.sqrt((np.dot(v1, v1)))
    normed2 = u2 / np.sqrt((np.dot(u2, u2)))

    # Plug into the generalized N-dimensional Rodrigues rotation formula:
    # R = I + ((n2⨂n1)-(n1⨂n2))sin(⁡α) + ((n1⨂n1)+(n2⨂n2))(cos(⁡α)-1)
    M1 = np.identity(N)
    M2 = np.sin(theta) * (np.outer(normed2, normed1) - np.outer(normed1, normed2))
    M3 = (np.cos(theta) - 1) * (np.outer(normed1, normed1) + np.outer(normed2, normed2))
    return M1 + M2 + M3


def project_from_N(D, N, vecs):
# Project from dimension N to dimension N-1 w.r.t. camera at D

    #Array indexing starts at 0, so decrement dimension by 1
    N -= 1

    #Convert to homogeneous coordinates
    quotient = vecs[:,N] - D
    newvecs = (((vecs.T) / quotient).T)

    #Restore the original array values from index N on for depth cueing
    newvecs[:,N:] = vecs[:,N:]
    return newvecs


def plot_line(axes, point1, point2, edge_width):
# Plot line from pt1 to pt2, w/ line thickness scaled for depth cueing
    width1 = edge_width / (point1[-1] - views[list(views.keys())[-1]])
    width2 = edge_width / (point2[-1] - views[list(views.keys())[-1]])
    width_range = np.linspace(width1, width2, 1000)
    del_x = np.linspace(point1[0], point2[0], 1000)
    del_y = np.linspace(point1[1], point2[1], 1000)
    lwidths = width_range
    points = np.array([del_x,del_y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    axes.add_collection(LineCollection(segments, linewidths=lwidths,color='w', alpha=1.0))
    return axes

def plot_vertex(axes, vertex, size=0.01):
# Plot vertex node, w/ node size scaled for depth cueing
    c0 = Circle((vertex[0], vertex[1]), size/(vertex[-1] - views[list(views.keys())[-1]]),
                fc='r',  ec='r', zorder=10)
    axes.add_patch(c0)
    return axes


# Set up the figure in matplotlib
fig, ax = plt.subplots(1, 3)
fig.set_figheight(6)
fig.set_figwidth(18)

# Set up three axes
for k in range(3):
    ax[k].set(adjustable='box', aspect='equal')
    ax[k].axes.xaxis.set_visible(False)
    ax[k].axes.yaxis.set_visible(False)
    ax[k].set_facecolor((0.5, 0.5, 0.5))

# Instantiate a 4-hypercube, a dodecaplex, and a 6-hypercube
hc4 = Nhypercube(4)
ddp = Dodecaplex()
hc6 = Nhypercube(6)

# Cycle through two full rotations
steps = 2
counter = 0

for step in range(steps):
    frames = 100
    d_theta = 2 * np.pi / (frames)
    rot_hc4 = get_random_rotation(4, d_theta)
    rot_ddp = get_random_rotation(4, d_theta)
    rot_hc6 = get_random_rotation(6, d_theta)

    for ind in range(frames):
        print(f'{counter}/{steps*frames}') #To indicate progress
        # Clear the axes after each frame
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        ax[0].set_title('Hypercube (4-dimensional)', fontsize=14, fontweight='bold')
        ax[1].set_title('Dodecaplex (4-dimensional)', fontsize=14, fontweight='bold')
        ax[2].set_title('Hypercube (6-dimensional)', fontsize=14, fontweight='bold')


        # Project from dimension N to dimension N - 1.
        # Repeat until we reach dimension 2 (the display screen).
        pts_hc4 = hc4.vertices
        for k in range(4, 2, -1):
            pts_hc4 = project_from_N(views[k], k, pts_hc4)

        pts_ddp = ddp.vertices
        for k in range(4, 2, -1):
            pts_ddp = project_from_N(views[k], k, pts_ddp)

        pts_hc6 = hc6.vertices
        for k in range(6,2,-1):
            pts_hc6 = project_from_N(views[k], k, pts_hc6)


        # Plot the vertices (only for the 4-hypercube)
        for vertex in range(hc4.vertices.shape[0]):  # Circles representing the anchor points and the bobs
            plot_vertex(ax[0], pts_hc4[vertex])


        # Plot the edges
        for j, k in hc4.edges:
            plot_line(ax[0], pts_hc4[j,:], pts_hc4[k,:], 5)

        for j, k in ddp.edges:
            plot_line(ax[1], pts_ddp[j,:], pts_ddp[k,:], 1)

        for j, k in hc6.edges:
            plot_line(ax[2], pts_hc6[j,:], pts_hc6[k,:], 3)


        # Set axes limits
        ax[0].axis('equal')
        ax[0].axis([-0.4, 0.4, -0.4, 0.4])
        ax[1].axis('equal')
        ax[1].axis([-1.8, 1.8, -1.8, 1.8])
        ax[2].axis('equal')
        ax[2].axis([-0.06, 0.06, -0.06, 0.06])


        # Create & save the image
        plt.savefig('frames/anim{:04d}.png'.format(counter), dpi=100)
        counter += 1


        # Incremental rotation of each object
        hc4.vertices = np.dot(hc4.vertices, rot_hc4)
        ddp.vertices = np.dot(ddp.vertices, rot_ddp)
        hc6.vertices = np.dot(hc6.vertices, rot_hc6)



