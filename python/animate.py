import matplotlib.animation as animation
import numpy as np
from pylab import *

dpi = 250
fps = 30

def ani_frame(pos, t1, t2):
	fig, ax = plt.subplots()

	#ax.set_aspect('equal')
	
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	fig.set_size_inches([5,5])
	fig.patch.set_facecolor('green')
	#tight_layout()

	east = pos['East'].squeeze()
	north = pos['North'].squeeze()
	time = pos['time']

	lag = 5000

	# Plot the whole path as a faint color
	bg_idx = np.where((time >= t1) & (time <= t2))
	col = (0.8, 0.8, 0.8)
	background, = ax.plot(east[bg_idx], north[bg_idx], color=col)

    # Get the samples that are in this time window
	def getIdx(t):
		idx = np.where((time > (t - lag)) & (time <= t))
		return idx

	# Create the plot of the recent data
	idx = getIdx(t1)
	col = (0.2, 0.2, 0.2)
	path, = ax.plot(east[idx], north[idx], color=col)

	# Mask out the path
	def init():
		fig.patch.set_facecolor('green')
		ax.set_axis_bgcolor('green')
		path.set_xdata(np.ma.array(east, mask=True))
		path.set_ydata(np.ma.array(north, mask=True))
		return path,

	# Plot a segment of the path
	def update_img(t):
		idx = getIdx(t)
		path.set_xdata(np.ma.array(east[idx], mask=False))
		path.set_ydata(np.ma.array(north[idx], mask=False))
		return path,

	ani = animation.FuncAnimation(fig,update_img,np.arange(t1,t2,step=1000/fps),init_func=init,interval=10,blit=True)
	writer = animation.writers['ffmpeg'](fps=fps)
	ani.save('demo.mp4',writer=writer,dpi=dpi)

	return ani
