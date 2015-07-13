import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from pylab import *

dpi = 150
fps = 30
lag = 5000  # length of the tail

def ani_frame(pos, t1, t2):
    fig, ax = plt.subplots()

    fig.patch.set_facecolor('green')
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Push axis slightly off teh edge of the screen to remove any border
    ax.set_position([-0.01, -0.01, 1.02, 1.02])
    ax.set_axis_bgcolor('green')
    ax.xaxis.label.set_color('green')
    ax.yaxis.label.set_color('green')

    # Make it nice and square, the final resolution is this times the DPI
    fig.set_size_inches([3,3])

    east = pos['East'].squeeze()
    north = pos['North'].squeeze()
    time = pos['time']

    # Plot the whole path as a faint color
    bg_idx = np.where((time >= t1) & (time <= t2))
    col = (0.2, 0.2, 0.2)
    background1, = ax.plot(east[bg_idx], north[bg_idx], color=col, linewidth=3)
    col = (0.8, 0.8, 0.8)
    background2, = ax.plot(east[bg_idx], north[bg_idx], color=col)


    # Get the samples that are in this time window
    def getIdx(t):
        idx = np.where((time > (t - lag)) & (time <= t))
        return idx

    # Create the plot of the recent data
    idx = getIdx(t1)
    col = (0.1, 0.1, 0.8)
    
    x = east[idx]
    y = north[idx]
    t = (t1 - time[idx]) / lag

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, array=t, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0,1), lw=2)
    lc.set_array(t)
    lc.set_linewidth(3)
    lc.set_zorder(20)  # make sure trail is on top
    ax.add_collection(lc)

    # Mask out the path
    def init():
        lc.set_segments([])
        lc.set_array([])
        lc.set_linewidth(0)
        return lc,

    # Plot a segment of the path
    def update_img(t):
        idx = getIdx(t)
        x = east[idx]
        y = north[idx]

        col = (double) (t - time[idx]) / lag

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc.set_segments(segments)
        lc.set_array(col)
        lc.set_linewidth(3)
        ax.add_collection(lc)
        ax.autoscale()
        plt.draw()

        return lc,

    ani = animation.FuncAnimation(fig,update_img,np.arange(t1,t2,step=1000/fps),interval=0,blit=False)

    writer = animation.writers['ffmpeg'](fps=30)
    ani.save('demo.mp4',dpi=dpi,fps=30,writer=writer)

    return lc
