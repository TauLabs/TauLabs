
# call this from shell.py with
# exec('lqg/replay_log.py')

from lqg.test import rtkf

rtkf.init()
rtkf.configure(tau=-3.05, gain=numpy.array([ 9.71691132,  9.64305401,  4.78812265], dtype=numpy.float64))

# to save time do not reparse variables if already done in parent
# namespace
try:
	gyros
except NameError:
	gyros = uavo_list.as_numpy_array(UAVO_Gyros)

try:
	actuator
except NameError:
	actuator = uavo_list.as_numpy_array(UAVO_ActuatorDesired)

# find the time of the last gyro update before the last actuator update
# to ensure we always have valid data for running
t_start = actuator['time'][0]
t_end = actuator['time'][-1]
g_idx_end = find(gyros['time'] < t_end)[-1]
g_idx_start = find(gyros['time'] > t_start)[0]

g_idx = np.arange(g_idx_start, g_idx_end)

# stop early for testing
#g_idx = g_idx[range(5000)]

STEPS = len(g_idx)
NUMX = 9

history = numpy.zeros((STEPS,NUMX))
times = numpy.zeros((STEPS,1))

dT = mean(diff(gyros['time']))
for idx in range(STEPS):

	i = g_idx[idx] # look up the gyro index for this time step
	t = gyros['time'][i]
	c_idx = find(actuator['time'] < t)[-1]  # find index for the most recent control before this gyro
	g = numpy.array([gyros['x'][i,0], gyros['y'][i,0], gyros['z'][i,0]], dtype=numpy.float64)
	c = numpy.array([actuator['Roll'][c_idx,0],actuator['Pitch'][c_idx,0],actuator['Yaw'][c_idx,0]], dtype=numpy.float64)


	state = rtkf.advance(g, c, dT)

	history[idx,:] = state
	times[idx] = t

clf()
ax1 = subplot(1,3,1)
plot(gyros['time'][g_idx],gyros['x'][g_idx,0],  times, history[:,0])
title('Rate')

ax2 = subplot(1,3,2,sharex=ax1)
plot(actuator['time'],actuator['Roll'], times, history[:,3])
title('Torque')

ax3 = subplot(1,3,3,sharex=ax1)
plot(times, history[:,6])
title('Bias')

xlim(t_start, t_end)