
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

class CINS:

	GRAV = 9.805

	def __init__(self):
		""" Creates the CINS class. 

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
		self.state = ins.init()
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

		self.state = ins.correction(Z, sensors)

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