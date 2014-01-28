import matplotlib.animation as animation
import numpy as np
from pylab import *

dpi = 150
fps = 30

def ani_frame(pos, t1, t2):
	fig, ax = plt.subplots()

	#ax.set_aspect('equal')
	
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

	lag = 5000

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
	path, = ax.plot(east[idx], north[idx], color=col, linewidth=3)

	fig.show()

	# Mask out the path
	def init():
		fig.patch.set_facecolor('green')
		path.set_xdata(np.ma.array(east, mask=True))
		path.set_ydata(np.ma.array(north, mask=True))
		return path,

	# Plot a segment of the path
	def update_img(t):
		idx = getIdx(t)
		path.set_xdata(np.ma.array(east[idx], mask=False))
		path.set_ydata(np.ma.array(north[idx], mask=False))
		return path,

	
	#t2 = t1 + 33 * 200

	ani = animation.FuncAnimation(fig,update_img,np.arange(t1,t2,step=1000/fps),init_func=init,interval=0,blit=False)
	#ani.save('demo.mp4',dpi=dpi,fps=30,extra_args=['-vcodec', 'libx264'])

	writer = animation.writers['ffmpeg'](fps=30)
	ani.save('demo.mp4',dpi=dpi,fps=30,writer=writer)

	return ani
