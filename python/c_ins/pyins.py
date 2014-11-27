
from sympy import symbols, lambdify, sqrt
from sympy import MatrixSymbol, Matrix
from numpy import cos, sin, power
from sympy.matrices import *
from quaternions import *
import numpy
import ins

class PyINS:

	GRAV = 9.81

	def __init__(self):
		""" Creates the INS14 class and prepares the equations. 

		Important variables are
		  * X  - the vector of state variables
		  * Xd - the vector of state derivatives for state and inputs
		  * Y  - the vector of outputs for current state value
		"""

		self.state = []

	def prepare(self):
		""" prepare the C INS wrapper
		"""

	def predict(self, U=[0,0,0,0,0,0], dT = 1.0/666.0):
		""" Perform the prediction step
		"""

		gyros = numpy.array(U[0:3])
		accels = numpy.array(U[3:])
		self.state = ins.prediction(gyros, accels, dT)

		#print `self.state`


	def correction(self, pos=None, vel=None, mag=None, baro=None):
		""" Perform the INS correction based on the provided corrections
		"""

		sensors = 0
		Z = numpy.zeros((10,1))

		# the masks must match the values in insgps.h

		if pos is not None:
			sensors = sensors | 0x0003
			Z[0] = pos[0]
			Z[1] = pos[1]

		if vel is not None:
			sensors = sensors | 0x0018
			Z[3] = vel[0]
			Z[4] = vel[1]
			Z[5] = vel[2]

		if mag is not None:
			sensors = sensors | 0x00C0
			Z[6] = mag[0]
			Z[7] = mag[1]
			Z[8] = mag[2]

		if baro is not None:
			sensors = sensors | 0x0200
			Z[9] = baro

		ins.correction(Z, sensors)

def run_uavo_list(uavo_list):

	import taulabs

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,2)

	gyros = uavo_list.as_numpy_array(taulabs.uavo.UAVO_Gyros)
	accels = uavo_list.as_numpy_array(taulabs.uavo.UAVO_Accels)
	ned = uavo_list.as_numpy_array(taulabs.uavo.UAVO_NEDPosition)
	vel = uavo_list.as_numpy_array(taulabs.uavo.UAVO_GPSVelocity)
	mag = uavo_list.as_numpy_array(taulabs.uavo.UAVO_Magnetometer)
	baro = uavo_list.as_numpy_array(taulabs.uavo.UAVO_BaroAltitude)

	STEPS = gyros['time'].shape[0]
	history = numpy.zeros((STEPS,14))
	times = numpy.zeros((STEPS,1))

	steps = 0
	t = gyros['time'][0]
	gyro_idx = 0
	accel_idx = 0
	ned_idx = 0
	vel_idx = 0
	mag_idx = 0
	baro_idx = 0

	sim = PyINS()
	sim.prepare()

	dT = 1 / 666.0
	tlast = t - dT

	while(True):
		if (gyros['time'][gyro_idx] < t) & (accels['time'][accel_idx] < t):
			# once the time marker is past both of these sample that were used
			# then find the next one in the future for each
			while gyros['time'][gyro_idx] < t:
				gyro_idx = gyro_idx + 1
			while accels['time'][accel_idx] < t:
				accel_idx = accel_idx + 1
			U = [gyros['x'][gyro_idx],gyros['y'][gyro_idx],gyros['z'][gyro_idx],accels['x'][accel_idx],accels['y'][accel_idx],accels['z'][accel_idx]]

			dT = dT*0.99 + numpy.double(gyros['time'][gyro_idx] - t) * 0.01
			
			sim.predict(U=U)

			if ned['time'][ned_idx] < t:
				sim.correction(pos=[ned['North'][ned_idx], ned['East'][ned_idx], ned['Down'][ned_idx]])
				ned_idx = ned_idx + 1

			if vel['time'][vel_idx] < t:
				sim.correction(vel=[vel['North'][vel_idx], vel['East'][vel_idx], vel['Down'][vel_idx]])
				vel_idx = vel_idx + 1

			if mag['time'][mag_idx] < t:
				sim.correction(mag=[mag['x'][mag_idx], mag['y'][mag_idx], mag['z'][mag_idx]])
				mag_idx = mag_idx + 1

			if baro['time'][baro_idx] < t:
				sim.correction(baro=baro['Altitude'][baro_idx])
				baro_idx = baro_idx + 1

			history[steps,:] = ins.r_X.T
			times[steps] = t

			steps = steps + 1

			if steps % 50 == 0:
				print "dT: " + `dT`

				ax[0][0].cla()
				ax[0][0].plot(times[0:steps],history[0:steps,0:3])
				ax[0][0].plot(ned['times'],ned['North'],'.',ned['times'],ned['East'])
				ax[0][1].cla()
				ax[0][1].plot(times[0:steps],history[0:steps,3:6])
				ax[0][0].plot(vel['times'],vel['North'],'.',vel['times'],vel['East'])
				ax[1][0].cla()
				ax[1][0].plot(times[0:steps],history[0:steps,6:10])
				ax[1][1].cla()
				ax[1][1].plot(times[0:steps],history[0:steps,10:])

				plt.draw()
				fig.show()

		t = t + 0.001 # advance 1ms
	return ins

def test():
	""" test the INS with simulated data
	"""

	from numpy import cos, sin

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,2)

	sim = PyINS()
	sim.prepare()

	dT = 1.0 / 666.0
	STEPS = 100000

	history = numpy.zeros((STEPS,16))
	history_rpy = numpy.zeros((STEPS,3))
	times = numpy.zeros((STEPS,1))

	for k in range(STEPS):
		ROLL = 0.1
		YAW  = 0.2
		sim.predict(U=[0,0,YAW, 0, PyINS.GRAV*sin(ROLL), -PyINS.GRAV*cos(ROLL) - 0.0], dT=dT)

		history[k,:] = sim.state
		history_rpy[k,:] = quat_rpy(sim.state[6:10])
		times[k] = k * dT

		angle = 0*numpy.pi/3 + YAW * dT * k # radians 
		height = 1.0 * k * dT

		if True and k % 60 == 59:
			sim.correction(pos=[[10],[5],[-height]])

		if True and k % 60 == 59:
			sim.correction(vel=[[0],[0],[-1]])

		if k % 20 == 8:
			sim.correction(baro=[height])

		if True and k % 20 == 15:
			sim.correction(mag=[[400 * cos(angle)], [-400 * sin(angle)], [1600]])

		if k % 1000 == 0:

			ax[0][0].cla()
			ax[0][0].plot(times[0:k:4],history[0:k:4,0:3])
			ax[0][0].set_title('Position')
			ax[0][1].cla()
			ax[0][1].plot(times[0:k:4],history[0:k:4,3:6])
			ax[0][1].set_title('Velocity')
			plt.sca(ax[0][1])
			plt.ylim(-2,2)
			ax[1][0].cla()
			ax[1][0].plot(times[0:k:4],history_rpy[0:k:4,:])
			ax[1][0].set_title('Attitude')
			ax[1][1].cla()
			ax[1][1].plot(times[0:k:4],history[0:k:4,10:])
			ax[1][1].set_title('Biases')

			plt.draw()
			fig.show()

	plt.show()

if  __name__ =='__main__':
	test()