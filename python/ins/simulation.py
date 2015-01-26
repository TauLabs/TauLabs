import numpy
import matplotlib.pyplot as plt
from quaternions import *

DEG2RAD = 3.1415 / 180.0
GRAVITY = 9.805
MAG = numpy.array([400,0,1600])

class Simulation:
	""" Simple inertial navigation testing code that simulates the basic
	physics and outputs the sensor values. It has no control schemes and
	will just generate a simple set of paths.
	"""

	def __init__(self):
		""" Creates the Simulation """

		z3 = numpy.zeros((3,))
		self.q = numpy.array([1.0, 0, 0,0])
		self.pos = z3
		self.vel = z3
		self.vel[0] = -2.85

		self.last_accel = z3
		self.last_gyro = z3

		self.T = 0

	def advance(self, gyros, accels, dT=1/500.0):
		""" Advance the simulation one time step with the given inputs """

		self.last_gyro = gyros
		self.last_accel = accels

		# unpack the data into individual variables
		q = self.q
		q0,q1,q2,q3 = self.q
		ax,ay,az = accels

		# allocate variables for derivatives
		qdot = numpy.zeros((4,))
		vdot = numpy.zeros((3,))
		pdot = self.vel

		# work out the derivatives of the quaternion
		qdot[0] = (-q[1] * gyros[0] - q[2] * gyros[1] - q[3] * gyros[2])
		qdot[1] = (q[0] * gyros[0] - q[3] * gyros[1] + q[2] * gyros[2])
		qdot[2] = (q[3] * gyros[0] + q[0] * gyros[1] - q[1] * gyros[2])
		qdot[3] = (-q[2] * gyros[0] + q[1] * gyros[1] + q[0] * gyros[2])

		# predict velocity for next step
		vdot[0] = (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * ax + 2.0 * (q1 * q2 - q0 * q3) * ay + 2 * (q1 * q3 + q0 * q2) * az;
		vdot[1] = 2.0 * (q1 * q2 + q0 * q3) * ax + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * ay + 2 * (q2 * q3 - q0 * q1) * az;
		vdot[2] = 2.0 * (q1 * q3 - q0 * q2) * ax + 2 * (q2 * q3 + q0 * q1) * ay + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * az + GRAVITY;

		self.q = self.q + qdot * dT * DEG2RAD / 2
		self.vel = self.vel + vdot * dT
		self.pos = self.pos + self.vel * dT

		self.q = quat_norm(self.q)

	def rock_and_turn(self, dT=1/500.0):
		""" simulate yawing while rocking forward and backwards """

		from numpy import sin, pi

		accels = numpy.array([0.0,0.0,-GRAVITY])
		gyros = numpy.array([0.0,0.0,5.0])

		if self.T == 0:
			self.vel[0] = 0

		# take off and accelerate at first
		if self.T < 5:
			accels[0] = 0.0
			accels[1] = 0
			accels[2] = -GRAVITY + 0.5

		# level off
		if self.T > 20 and self.T < 25:
			accels[2] = -GRAVITY - 0.5

		if self.T > 30:
			gyros[1] = gyros[1] + 10.0 * sin(self.T * 2 * pi * 0.1)

		self.advance(gyros, accels, dT)
		self.T = self.T + dT

	def fly_circle(self, dT=1/500.0):
		accels = numpy.array([0.0,1,-GRAVITY])
		gyros = numpy.array([0,0,-20])

		# take off and accelerate at first
		if self.T < 5:
			accels[0] = 0.0
			accels[1] = 0
			accels[2] = -GRAVITY + 0.5
			gyros[2] = 0

		# level off
		if self.T > 20 and self.T < 25:
			accels[2] = -GRAVITY - 0.5

		self.advance(gyros, accels, dT)
		self.T = self.T + dT

	def get_q(self):
		return self.q

	def get_rpy(self):
		return quat_rpy(self.q)

	def get_accel(self):
		return self.last_accel

	def get_gyro(self):
		return self.last_gyro

	def get_mag(self):
		Rbe = quat_rbe(self.q)
		return Rbe.dot(MAG)

	def get_pos(self):
		return self.pos

	def get_vel(self):
		return self.vel

if  __name__ =='__main__':

	s = Simulation()

	STEP = 100000

	pos = numpy.zeros((STEP,3))
	vel = numpy.zeros((STEP,3))
	rpy = numpy.zeros((STEP,3))
	t = numpy.zeros((STEP,))

	for i in numpy.arange(STEP):
		rpy[i,:] = quat_rpy(s.get_q())
		pos[i,:] = s.get_pos()
		vel[i,:] = s.get_vel()
		t[i] = i / 500.0
		s.fly_circle()

	plt.plot(vel[:,0],vel[:,1])
	plt.draw()
	plt.show()