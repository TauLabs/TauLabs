
from sympy import symbols, lambdify, sqrt
from sympy import MatrixSymbol, Matrix
from numpy import cos, sin, power
from sympy.matrices import *
from quaternions import *
import numpy
import ins

# this is the set of (currently) recommend INS settings. modified from
# https://raw.githubusercontent.com/wiki/TauLabs/TauLabs/files/htfpv-sparky-nav_20130527.uav
default_mag_var = numpy.array([10.0, 10.0, 100.0])
default_gyro_var = numpy.array([1e-5, 1e-5, 1e-4])
default_accel_var = numpy.array([0.01, 0.01, 0.01])
default_baro_var = 0.1
default_gps_var=numpy.array([1e-3,1e-2,10])

class PyINS:

	GRAV = 9.805

	def __init__(self):
		""" Creates the INS14 class and prepares the equations. 

		Important variables are
		  * X  - the vector of state variables
		  * Xd - the vector of state derivatives for state and inputs
		  * Y  - the vector of outputs for current state value
		"""

		self.state = []

	def configure(self, mag_var=None, gyro_var=None, accel_var=None, baro_var=None, gps_var=None):
		""" configure the INS parameters """

		if mag_var is not None:
			ins.configure(mag_var=mag_var)
		if gyro_var is not None:
			ins.configure(gyro_var=gyro_var)
		if accel_var is not None:
			ins.configure(accel_var=accel_var)
		if baro_var is not None:
			ins.configure(baro_var=baro_var)
		if gps_var is not None:
			ins.configure(gps_var=gps_var)

	def prepare(self):
		""" prepare the C INS wrapper
		"""
		ins.init()
		self.configure(
			mag_var=default_mag_var,
			gyro_var=default_gyro_var,
			accel_var=default_accel_var,
			baro_var=default_baro_var,
			gps_var=default_gps_var
			)

	def predict(self, gyros, accels, dT = 1.0/666.0):
		""" Perform the prediction step
		"""

		self.state = ins.prediction(gyros, accels, dT)

	def correction(self, pos=None, vel=None, mag=None, baro=None):
		""" Perform the INS correction based on the provided corrections
		"""

		sensors = 0
		Z = numpy.zeros((10,),numpy.float64)

		# the masks must match the values in insgps.h

		if pos is not None:
			sensors = sensors | 0x0003
			Z[0] = pos[0]
			Z[1] = pos[1]

		if vel is not None:
			sensors = sensors | 0x0038
			Z[3] = vel[0]
			Z[4] = vel[1]
			Z[5] = vel[2]

		if mag is not None:
			sensors = sensors | 0x01C0
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

	attitude = uavo_list.as_numpy_array(taulabs.uavo.UAVO_AttitudeActual)
	gyros = uavo_list.as_numpy_array(taulabs.uavo.UAVO_Gyros)
	accels = uavo_list.as_numpy_array(taulabs.uavo.UAVO_Accels)
	ned = uavo_list.as_numpy_array(taulabs.uavo.UAVO_NEDPosition)
	vel = uavo_list.as_numpy_array(taulabs.uavo.UAVO_GPSVelocity)
	mag = uavo_list.as_numpy_array(taulabs.uavo.UAVO_Magnetometer)
	baro = uavo_list.as_numpy_array(taulabs.uavo.UAVO_BaroAltitude)

	STEPS = gyros['time'].size
	history = numpy.zeros((STEPS,16))
	history_rpy = numpy.zeros((STEPS,3))
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

	dT = numpy.mean(numpy.diff(gyros['time']))

	for gyro_idx in numpy.arange(STEPS):

		t = gyros['time'][gyro_idx]
		steps = gyro_idx
		accel_idx = (numpy.abs(accels['time']-t)).argmin()
			
		gyros_dat = numpy.array([gyros['x'][gyro_idx],gyros['y'][gyro_idx],gyros['z'][gyro_idx]]).T[0]
		accels_dat = numpy.array([accels['x'][accel_idx],accels['y'][accel_idx],accels['z'][accel_idx]]).T[0]

		sim.predict(gyros_dat,accels_dat,dT)

		if (ned_idx < ned['time'].size) and (ned['time'][ned_idx] < t):
			sim.correction(pos=[ned['North'][ned_idx,0], ned['East'][ned_idx,0], ned['Down'][ned_idx,0]])
			ned_idx = ned_idx + 1

		if (vel_idx < vel['time'].size) and (vel['time'][vel_idx] < t):
			sim.correction(vel=[vel['North'][vel_idx,0], vel['East'][vel_idx,0], vel['Down'][vel_idx,0]])
			vel_idx = vel_idx + 1

		if (mag_idx < mag['time'].size) and (mag['time'][mag_idx] < t):
			sim.correction(mag=[mag['x'][mag_idx,0], mag['y'][mag_idx,0], mag['z'][mag_idx,0]])
			mag_idx = mag_idx + 1

		if (baro_idx < baro['time'].size) and (baro['time'][baro_idx] < t):
			sim.correction(baro=baro['Altitude'][baro_idx,0])
			baro_idx = baro_idx + 1

		history[steps,:] = sim.state
		history_rpy[steps,:] = quat_rpy(sim.state[6:10])
		times[steps] = t

	ax[0][0].cla()
	ax[0][0].plot(ned['time'],ned['North'],'k',ned['time'],ned['East'],'k',ned['time'],ned['Down'],'k', baro['time'], -baro['Altitude'],'k')
	ax[0][0].plot(times,history[:,0:3])
	ax[0][0].set_title('Position')
	plt.sca(ax[0][0])
	plt.ylabel('m')
	ax[0][1].cla()
	ax[0][1].plot(vel['time'],vel['North'],'k',vel['time'],vel['East'],'k')
	ax[0][1].plot(times,history[:,3:6])
	ax[0][1].set_title('Velocity')
	plt.sca(ax[0][1])
	plt.ylabel('m/s')
	#plt.ylim(-2,2)
	ax[1][0].cla()
	ax[1][0].plot(attitude['time'],attitude['Roll'],'k',attitude['time'],attitude['Pitch'],'k',attitude['time'],attitude['Yaw'],'k')
	ax[1][0].plot(times,history_rpy[:,:])
	ax[1][0].set_title('Attitude')
	plt.sca(ax[1][0])
	plt.ylabel('Angle (Deg)')
	plt.xlabel('Time (s)')
	#plt.ylim(-1.1,1.1)
	ax[1][1].cla()
	ax[1][1].plot(times,history[:,10:])
	ax[1][1].set_title('Biases')
	plt.sca(ax[1][1])
	plt.ylabel('Bias (rad/s)')
	plt.xlabel('Time (s)')

	plt.draw()
	plt.show()

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