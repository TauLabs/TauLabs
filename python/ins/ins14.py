from sympy import symbols, lambdify
from sympy import MatrixSymbol, Matrix
from sympy.matrices import *
import numpy

class INS14:

	def __init__(self):
		""" Creates the INS14 class and prepares the equations. 

		Important variables are
		  * X  - the vector of state variables
		  * Xd - the vector of state derivatives for state and inputs
		  * Y  - the vector of outputs for current state value
		"""

		# Create state variable
		Px, Py, Pz = symbols('Px Py Pz')
		Vx, Vy, Vz = symbols('Vx Vy Vz')
		q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
		wbx, wby, wbz = symbols('wbx wby wbz')
		abz = symbols('abz')
		# Helper variables
		P = Matrix([Px, Py, Pz])
		V = Matrix([Vx, Vy, Vz])
		q = Matrix([q0, q1, q2, q3])
		wb = Matrix([wbx, wby, wbz])
		X = Matrix([Px, Py, Pz, Vx, Vy, Vz, q0, q1, q2, q3, wbx, wby, wbz, abz])

		# strap down equation matrix
		Q = Matrix([[-q1, -q2, -q3],[q0, -q3, q2],[q3, q0, -q1],[-q2, q1, q0]])

		# Inputs to system
		wx, wy, wz = symbols('wx, wy, wz')
		ax, ay, az = symbols('ax, ay, az')
		U = [wx, wy, wz, ax, ay, az]

		# Noise inputs
		nwx, nwy, nwz = symbols('nwx, nwy, nwz')       # gyro noise
		nax, nay, naz = symbols('nax, nay, naz')       # accel noise
		nwbx, nwby, nwbz = symbols('nwbx, nwby, nwbz') # gyro bias drift
		nabz = symbols('nabz')                         # z-accel bias drift
		n = [nwx, nwy, nwz, nax, nay, naz, nwbx, nwby, nwbz, nabz]

		# State equations

		Rbe = Matrix([[(q0*q0+q1*q1-q2*q2-q3*q3), 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
		              [2*(q1*q2-q0*q3), (q0*q0-q1*q1+q2*q2-q3*q3),  2*(q2*q3+q0*q1)],
		              [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), (q0*q0-q1*q1-q2*q2+q3*q3)]])
		Reb = Rbe.T

		atrue = Matrix([ax-nax,ay-nay,az-naz-abz])
		Pd = V
		Vd = (Reb * atrue) + Matrix([0,0,9.81])
		qd = Q/2 * Matrix([wx-nwx-wbx,wy-nwy-wby,wz-nwz-wbz])  # measured - noise - bias
		wbd = Matrix([nwbx, nwby, nwbz])
		abd = nabz

		# combine all the derivatives.
		Xd = Matrix([Pd[0], Pd[1], Pd[2], Vd[0], Vd[1], Vd[2], qd[0], qd[1], qd[2], qd[3], wbd[0], wbd[1], wbd[2], abd])
		Xd_nn = Xd.subs(nwbx,0).subs(nwby,0).subs(nwbz,0).subs(nwx,0).subs(nwy,0).subs(nwz,0).subs(nabz,0)  # version without noise inputs

		# Compute jacobians of state equations.
		wrt = [Px, Py, Pz, Vx, Vy, Vz, q0, q1, q2, q3, wbx, wby, wbz, abz, wx, wy, wz]
		Xdl = lambdify(wrt,Xd_nn)
		F = Xd_nn.jacobian(X)
		G = Xd.jacobian(n)

		# Output equations.
		#Be = MatrixSymbol('Be',3,1)                    # mag near home location
		Be = Matrix([400, 0, 1400])
		Bb = Rbe * Matrix(Be);                         # predicted mag in body frame
		Y  = Matrix([P, V, Bb, Matrix([-Pz])])         # predicted outputs
		H  = Y.jacobian(X)

		# Store the useful functions.
		self.X = X      # vector of state variables
		self.Xd = Xd    # vector of derivatives in state space as a function of state and inputs
		self.Y = Y      # vector of functions that predict the output for the current state space output
		self.U = U      # vector of inputs
		self.noise = n

		# Functions to linearize derivatives around current point.
		self.F = F
		self.G = G
		self.H = H

		# These are populated in the prepare method
		self.r_X = []
		self.r_P = []
		self.Q = []
		self.R = []
		self.l_Xd = []
		self.l_F = []
		self.l_G = []
		self.l_H = []
		self.l_Y = []

	def prepare(self):
		""" Prepare to run data through the INS14

		Initializes the state variables and covariance. Compiles equations to run faster.

		 * self.R is the measurement noise (pos, vel, mag, baro)
		 * self.Q is the process noise (gyro, accel, gyro bias, accel bias)
		"""

		self.r_X = numpy.matrix([0,0,0,0,0,0,1,0,0,0,0,0,0,0]).T
		self.r_P = numpy.diagflat([25,25,25,5,5,5,1e-5,1e-5,1e-5,1e-5,1e-9,1e-9,1e-9,1e-9])
		self.R = numpy.diagflat([0.001, 0.001, 10, 0.01, 0.01, 0.01, 10, 10, 100, 0.01])
		self.Q = numpy.diagflat([1e-5,1e-5,1e-4,0.003,0.003,0.003,2e-5,2e-5,2e-5,1e-5])

		# the noise inputs to the system are not used in the prediction (or assume their mean of zero)
		Xd = self.Xd
		for i in self.noise:
			Xd = Xd.subs(i,0)

		if True:
			from sympy.utilities.autowrap import ufuncify

			self.l_Xd = lambdify([self.X, self.U],Xd, "numpy")

			# Jacobian of state derivatives  as a function of state and input
			self.l_F  = lambdify([self.X, self.U],Xd.jacobian(self.X), "numpy")

			# Jacobian of state derivatives with process noise as a function of state
			# note this uses the original Xd with noise terms still included
			self.l_G  = lambdify([self.X], self.Xd.jacobian(self.noise), "numpy")

			# Jacobian of ouputs versus state
			self.l_H  = lambdify([self.X], self.Y.jacobian(self.X), "numpy")

			# Functional form to predict outputs
			self.l_Y  = lambdify([self.X], self.Y, "numpy")
		else:
			
			self.l_Xd = lambdify([self.X, self.U],Xd)

			# Jacobian of state derivatives  as a function of state and input
			self.l_F  = lambdify([self.X, self.U],Xd.jacobian(self.X))

			# Jacobian of state derivatives with process noise as a function of state
			# note this uses the original Xd with noise terms still included
			self.l_G  = lambdify([self.X], self.Xd.jacobian(self.noise))

			# Jacobian of ouputs versus state
			self.l_H  = lambdify([self.X], self.Y.jacobian(self.X))

			# Functional form to predict outputs
			self.l_Y  = lambdify([self.X], self.Y)
	
	def normalize(self):
		""" Make sure the quaternion state stays normalized
		"""

		q = self.r_X[6:10]
		qn = sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
		for i in range(4):
			q[i] = q[i] / qn
		self.r_X[6:10] = q

		self.r_X[10:] = 0

	def attitude_swing_twist(self):
		""" Decompose the attitude into a yaw and non-yaw component
		    with the swing twist-algorithm: 
			( https://stackoverflow.com/a/22401169 )
		"""

		q  = self.r_X[6:10]
		print "Attitude: " + `quat_rpy_display(q)`
		ra = Matrix([q[1], q[2], q[3]])
		di = Matrix([0, 0, 1])
		p  = (ra.T * di)[0,0] * di
		twist = Matrix([q[0,0], p[0], p[1], p[2]])
		twist = twist / sqrt((twist.T * twist)[0,0])
		print "Normed twist: " + `quat_rpy_display(twist)`
		swing = quat_multiply(q, quat_inv(twist))
		print "Remainder: " + `quat_rpy_display(swing)`


	def predict(self, U=[0,0,0,0,0,0], dT = 1.0/666.0):
		""" Perform the prediction step
		"""

		# fourth-order runga kuta state prediction
		k1 = self.l_Xd(self.r_X.tolist(), U) 
		k2 = self.l_Xd((self.r_X + 0.5*dT*k1).tolist(), U)
		k3 = self.l_Xd((self.r_X + 0.5*dT*k2).tolist(), U)
		k4 = self.l_Xd((self.r_X + dT*k3).tolist() , U)
		d = (k1+2*k2+2*k3+k4) / 6

		f = self.l_F(self.r_X.tolist(), U)
		g = self.l_G(self.r_X.T.tolist())

		self.r_X = self.r_X + dT * d

		P = self.r_P
		#self.r_P = P + dT * (numpy.f*P + P*f.T) + (dT**2) * g * diag(self.Q) * g.T
		#self.r_P = (eye(NUMX)+F*T)*Pplus*(eye(NUMX)+F*T)' + T^2*G*diag(Q)*G'
		I = numpy.matrix(numpy.identity(14))
		self.r_P = (I + f * dT) * P * (I + f * dT).T + (dT**2) * g * diag(self.Q) * g.T

		self.normalize()

	def correction(self, pos=None, vel=None, mag=None, baro=None):
		""" Perform the INS correction based on the provided corrections
		"""

		P = self.r_P

		Y = self.l_Y([self.r_X.tolist()])
		H = self.l_H([self.r_X.tolist()])

		idx = []
		Z = []

		if pos is not None:
			idx.extend((0,1))
			Z.extend(pos[0:2])

		if vel is not None:
			idx.extend((3,4))
			Z.extend(vel[0:2])

		if mag is not None:
			idx.extend((6,7))
			Z.extend(mag[0:2])

		if baro is not None:
			idx.append(9)
			Z.append(baro)

		# construct appropriately sized predictions based on provided inputs
		Z = numpy.matrix(Z)
		Y = Y[idx]
		H = H[idx,:]
		R = self.R[idx,:][:,idx]

		# calculate Kalman gain matrix
		# K = P*H.T/(H*P*H.T + self.R);
		K = numpy.linalg.lstsq(numpy.matrix(H*P*H.T + R), numpy.matrix(P*H.T).T)[0].T

		self.r_X = self.r_X + K*(Z-Y)

		self.r_P = P - K*H*P;

		self.normalize()

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

	ins = INS14()
	ins.prepare()

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
			print "dT: " + `dT`
			ins.predict(U=U)

			if ned['time'][ned_idx] < t:
				print `t` + " ned: " + `ned['time'][ned_idx]`
				ins.correction(pos=[ned['North'][ned_idx], ned['East'][ned_idx], ned['Down'][ned_idx]])
				ned_idx = ned_idx + 1

			if vel['time'][vel_idx] < t:
				print `t` + " vel: " + `vel['time'][vel_idx]`
				ins.correction(vel=[vel['North'][vel_idx], vel['East'][vel_idx], vel['Down'][vel_idx]])
				vel_idx = vel_idx + 1

			if mag['time'][mag_idx] < t:
				print `t` + " mag: " + `mag['time'][mag_idx]`
				ins.correction(mag=[mag['x'][mag_idx], mag['y'][mag_idx], mag['z'][mag_idx]])
				mag_idx = mag_idx + 1

			if baro['time'][baro_idx] < t:
				print `t` + " baro: " + `baro['time'][baro_idx]` + " " + `baro['Altitude'][baro_idx]`
				ins.correction(baro=baro['Altitude'][baro_idx])
				baro_idx = baro_idx + 1

			history[steps,:] = ins.r_X.T
			times[steps] = t

			steps = steps + 1

			if steps % 50 == 0:
				ax[0][0].cla()
				ax[0][0].plot(times[0:steps],history[0:steps,0:3])
				ax[0][1].cla()
				ax[0][1].plot(times[0:steps],history[0:steps,3:6])
				ax[1][0].cla()
				ax[1][0].plot(times[0:steps],history[0:steps,6:10])
				ax[1][1].cla()
				ax[1][1].plot(times[0:steps],history[0:steps,10:])

				ax[0][2].imshow(ins.r_P)

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

	ins = INS14()
	ins.prepare()

	dT = 1.0 / 666.0

	STEPS = 10000
	history = numpy.zeros((STEPS,14))
	times = numpy.zeros((STEPS,1))

	for k in range(STEPS):
		ins.predict(U=[0,0,1,0,0,-9.81+0.01])

		history[k,:] = ins.r_X.T
		times[k] = k * dT

		angle = 1 * dT * k # radians 
		height = 0.3 * k * dT

		if k % 60 == 0:
			ins.correction(pos=[[0],[0],[-height]])

		if k % 60 == 5:
			ins.correction(vel=[[0],[0],[0]])

		if k % 20 == 8:
			ins.correction(baro=[height])

		if k % 20 == 15:
			ins.correction(mag=[[400 * cos(angle)], [-400 * sin(angle)], [1600]])

		if k % 250 == 0:
			print `k`

			ax[0][0].cla()
			ax[0][0].plot(times[0:k:4],history[0:k:4,0:3])
			ax[0][1].cla()
			ax[0][1].plot(times[0:k:4],history[0:k:4,3:6])
			ax[1][0].cla()
			ax[1][0].plot(times[0:k:4],history[0:k:4,6:10])
			ax[1][1].cla()
			ax[1][1].plot(times[0:k:4],history[0:k:4,10:])

			#print `ins.r_P`
			#ax[0][2].imshow(ins.r_P)

			plt.draw()
			fig.show()

def main():

	import sys
	from IPython.core import ultratb
	sys.excepthook = ultratb.FormattedTB(mode='Verbose',color_scheme='Linux', call_pdb=1)	

	import matplotlib.pyplot as plt
	fig, ax1 = plt.subplots()

	i = INS14()
	i.prepare()

	STEPS = 1000
	history = numpy.zeros((STEPS,14))
	for k in range(STEPS):

		i.predict()

		if k % 10 == 0:
			i.correction(vel=[0,0,0],baro=0)

		history[k,:] = i.r_X.T

		if k % 50 == 0:
			print `k`
			ax1.cla()
			ax1.plot(numpy.arange(0,k),history[0:k,0])
			plt.draw()
			fig.show()


if  __name__ =='__main__':
    test()


