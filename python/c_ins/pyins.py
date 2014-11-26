
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
			sensors = sensors | 0x0038
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

		print "Applying correction: " + hex(sensors)
		ins.correction(Z, sensors)

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
	times = numpy.zeros((STEPS,1))

	for k in range(STEPS):
		ROLL = 0.5
		YAW  = 0.5
		sim.predict(U=[0,0,YAW, 0, PyINS.GRAV*sin(ROLL),-PyINS.GRAV*cos(ROLL) - 0.1], dT=0.0015)

		history[k,:] = sim.state
		times[k] = k * dT

		angle = numpy.pi/3 + YAW * dT * k # radians 
		height = 1.0 * k * dT

		if True and k % 60 == 59:
			sim.correction(pos=[[10],[5],[-height]])

		if True and k % 60 == 59:
			sim.correction(vel=[[0],[0],[0]])

		if k % 20 == 8:
			sim.correction(baro=[height])

		if True and k % 20 == 15:
			sim.correction(mag=[[400 * cos(angle)], [-400 * sin(angle)], [1600]])

		if k % 50 == 0:
			print `k` + " Att: " + `quat_rpy_display(sim.state[6:10])`

			ax[0][0].cla()
			ax[0][0].plot(times[0:k:4],history[0:k:4,0:3])
			ax[0][1].cla()
			ax[0][1].plot(times[0:k:4],history[0:k:4,3:6])
			ax[1][0].cla()
			ax[1][0].plot(times[0:k:4],history[0:k:4,6:10])
			ax[1][1].cla()
			ax[1][1].plot(times[0:k:4],history[0:k:4,10:])

			plt.draw()
			fig.show()

if  __name__ =='__main__':
	test()