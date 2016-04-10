
# call this from shell.py with
# exec('lqg/replay_log.py')

# the expected gyro noise, this can be measured from real flights
SA = 1000.0

from lqg.test import rtkf

rtkf.init()
rtkf.configure(tau=-4.0, gain=numpy.array([ 9.2516613 ,  9.22609043,  6.64852238], dtype=numpy.float64))

# configure filter parameters
  # qw is the amount of process noise in the rate parameter. this parameter also
  #    interacts with the gyro noise parameter in that if we expect little noise
  #    from the gyro then the estimate will be pulled more by the measurements
  #    regardless of this parameter. so a low value pushes the rate to trust the
  #    torque measurement more and a high value the gyro measurement more
  # qu is the amount of drift in the torque measurement, so a low value will make
  #    the torque more reflect the smoothed actuator input (and really slowly
  #    changing bias)
rtkf.configure(qw=1e0, qu=1e-5, qbias=1e-12, sa=SA)

# to save time do not reparse variables if already done in parent
# namespace
try:
	actuator
	ad
	gyros
except NameError:
	actuator = uavo_list.as_numpy_array(UAVO_ActuatorDesired)
	ad = uavo_list.as_numpy_array(UAVO_ActuatorDesired)
	gyros = uavo_list.as_numpy_array(UAVO_Gyros)

# find the time of the last gyro update before the last actuator update
# to ensure we always have valid data for running
t_start = actuator['time'][0]
t_end = actuator['time'][-1]
g_idx_end = find(gyros['time'] < t_end)[-1]
g_idx_start = find(gyros['time'] > t_start)[0]

g_idx = np.arange(g_idx_start, g_idx_end)

STEPS = len(g_idx)
NUMX = 9

history = numpy.zeros((STEPS,NUMX))
times = numpy.zeros((STEPS,))

ll = numpy.zeros((STEPS,3))
err = numpy.zeros((STEPS,3))

dT = mean(diff(gyros['time']))

old_state = numpy.zeros(9,)
for idx in range(STEPS):

	i = g_idx[idx] # look up the gyro index for this time step
	t = gyros['time'][i]
	c_idx = find(actuator['time'] < t)[-1]  # find index for the most recent control before this gyro
	g = numpy.array([gyros['x'][i,0], gyros['y'][i,0], gyros['z'][i,0]], dtype=numpy.float64)
	c = numpy.array([actuator['Roll'][c_idx,0],actuator['Pitch'][c_idx,0],actuator['Yaw'][c_idx,0]], dtype=numpy.float64)

	state = rtkf.advance(g, c, dT)

	# store the log likelihood of this time step observation, use the old
	# state to give a "cross validated" attempt at predicting the future
	# since the gyro should be fairly smooth in reality
	g_err = (g - old_state[0:3])
	old_state = state
	ll[idx,:] = (-pow(g_err, 2) / 2 / SA) - log(sqrt(SA * 2 * pi))
	err[idx,:] = g_err

	history[idx,:] = state
	times[idx] = t

if True:
	clf()

	t1 = 90
	t2 = 91

	def get_idx(uavo):
		global t1
		global t2

		from matplotlib.mlab import find
		from numpy import logical_and
		return find(logical_and(uavo['time'] > t1, uavo['time'] < t2))

	g_idx = get_idx(gyros)
	ad_idx = get_idx(ad)

	kf_idx = find(logical_and(times > t1, times < t2))

	clf()
	plot(gyros['time'][g_idx],gyros['z'][g_idx,0], times[kf_idx], history[kf_idx,2], times[kf_idx], history[kf_idx,5]*20, ad['time'][ad_idx], ad['Yaw'][ad_idx,0]*100)
	legend(['Gyro','Rate','Torque','Actuator'])

	# ax1 = subplot(2,3,1)
	# plot(gyros['time'][g_idx],gyros['x'][g_idx,0],  times, history[:,0])
	# title('Rate')

	# ax2 = subplot(2,3,2,sharex=ax1)
	# plot(actuator['time'],actuator['Roll'], times, history[:,3])
	# title('Torque')

	# ax3 = subplot(2,3,3,sharex=ax1)
	# plot(times, history[:,7])
	# title('Bias')

	# ax4 = subplot(2,3,4,sharex=ax1)
	# plot(gyros['time'][g_idx],gyros['y'][g_idx,0],  times, history[:,1])
	# title('Rate')

	# ax5 = subplot(2,3,5,sharex=ax1)
	# plot(actuator['time'],actuator['Pitch'], times, history[:,4])
	# title('Torque')

	# ax6 = subplot(2,3,6,sharex=ax1)
	# plot(times, history[:,8])
	# title('Bias')

	# xlim(t_start, t_end)

print `mean(ll[:,0])` + " " + `mean(ll[:,1])`