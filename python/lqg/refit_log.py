
# call this from shell.py with
# exec('lqg/replay_log.py')

from lqg.test import rtsi

rtsi.init()

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

try:
	fs
except NameError:
	fs = uavo_list.as_numpy_array(UAVO_FlightStatus)

# find the time of the last gyro update before the last actuator update
# to ensure we always have valid data for running
#t_start = actuator['time'][0]
#t_start = fs['time'][find(fs['FlightMode']==10)[0]]
t_start = actuator['time'][find(actuator['Throttle']>0)[0]]
t_end = actuator['time'][-1]
g_idx_end = find(gyros['time'] < t_end)[-1]
g_idx_start = find(gyros['time'] > t_start)[0]

g_idx = np.arange(g_idx_start, g_idx_end)

STEPS = len(g_idx)
NUMX = 14

history = numpy.zeros((STEPS,NUMX))
times = numpy.zeros((STEPS,))

ll = numpy.zeros((STEPS,3))
err = numpy.zeros((STEPS,3))

dT = mean(diff(gyros['time']))

old_state = numpy.zeros(14,)
for idx in range(STEPS):

	i = g_idx[idx] # look up the gyro index for this time step
	t = gyros['time'][i]
	c_idx = find(actuator['time'] < t)[-1]  # find index for the most recent control before this gyro
	g = numpy.array([gyros['x'][i,0], gyros['y'][i,0], gyros['z'][i,0]], dtype=numpy.float64)
	c = numpy.array([actuator['Roll'][c_idx,0],actuator['Pitch'][c_idx,0],actuator['Yaw'][c_idx,0]], dtype=numpy.float64)

	state = rtsi.predict(g, c, dT)


	history[idx,:] = state
	times[idx] = t

if True:
	clf()

	ax1 = subplot(311)
	plot(gyros['time'], gyros['x'], times,history[:,0])
	ylabel('Rate (deg/s)')

	subplot(312, sharex=ax1)
	plot(times,history[:,1], actuator['time'],actuator['Roll'],'k')
	ylabel('Torque (deg/s^2)')

	subplot(313,sharex=ax1)
	plot(times,history[:,9:])
	legend(['r','p','y1','y2','tau'])
	ylabel('Parameters');
	xlabel('Time')

	xlabel('Time (s)')
	#xlim(95,97)

