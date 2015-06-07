from sympy import symbols, lambdify, sqrt
from sympy import MatrixSymbol, Matrix
from numpy import cos, sin, power
from sympy.matrices import *
from quaternions import *
import numpy

# this is the set of (currently) recommend INS settings. modified from
# https://raw.githubusercontent.com/wiki/TauLabs/TauLabs/files/htfpv-sparky-nav_20130527.uav
default_mag_var = numpy.array([10.0, 10.0, 100.0])
default_gyro_var = numpy.array([1e-5, 1e-5, 1e-4])
default_accel_var = numpy.array([0.01, 0.01, 0.01])
default_baro_var = 0.1
default_gps_var=numpy.array([1e-3,1e-2,10])

class PyINS:

	GRAV = 9.805
	MAG_HEADING = True # use the compass for heading only

	def __init__(self):
		""" Creates the PyINS class and prepares the equations. 

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
		Vd = (Reb * atrue) + Matrix([0,0,self.GRAV])
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
		Be = Matrix([400, 0, 1600])
		if self.MAG_HEADING:
			# Project the mag vector to the body frame by only the rotational component
			k1 = sqrt(power(q0**2 + q1**2 - q2**2 - q3**2,2) + power(q0*q3*2 +  q1*q2*2,2))
			Rbh = Matrix([[(q0**2 + q1**2 - q2**2 - q3**2)/k1,              (2*q0*q3 + 2*q1*q2)/k1      ,  0],
				          [-(2*q0*q3 + 2*q1*q2)/k1           ,       (q0**2 + q1**2 - q2**2 - q3**2)/k1 ,  0],
				          [0,0,1]]);
			Bb = Rbh * Matrix(Be);                         # predicted mag in body frame
		else:
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
		self.r_X = numpy.matrix([0,0,0,0,0,0,1.0,0,0,0,0,0,0,0]).T
		self.r_P = []
		self.Q = []
		self.R = []
		self.l_Xd = []
		self.l_F = []
		self.l_G = []
		self.l_H = []
		self.l_Y = []

		# state format used by common code
		self.state = numpy.zeros((16))
		self.state[0:14] = self.r_X[0:14].T
		self.state[-1] = self.r_X[-1]

	def prepare(self):
		""" Prepare to run data through the PyINS

		Initializes the state variables and covariance. Compiles equations to run faster.

		 * self.R is the measurement noise (pos, vel, mag, baro)
		 * self.Q is the process noise (gyro, accel, gyro bias, accel bias)
		"""

		Q = [default_gyro_var[0], default_gyro_var[1], default_gyro_var[2],
		     default_accel_var[0], default_accel_var[1], default_accel_var[2],
		     5e-7, 5e-7, 2e-6, 5e-4]
		R = [default_gps_var[0], default_gps_var[0], default_gps_var[2], 
		     default_gps_var[1], default_gps_var[1], default_gps_var[2],
		     default_mag_var[0], default_mag_var[1], default_mag_var[2], default_baro_var]

		self.r_X = numpy.matrix([0,0,0,0,0,0,1.0,0,0,0,0,0,0,0]).T
		self.r_P = numpy.diagflat([25,25,25,
			                       5,5,5,
			                       1e-5,1e-5,1e-5,1e-5,
			                       1e-9,1e-9,1e-9,
			                       1e-7])
		self.R = numpy.diagflat(R)
		self.Q = numpy.diagflat(Q)

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

		q = self.r_X[6:10,0]
		qnew = numpy.zeros((4,1))
		qn = sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
		for i in range(4):
			qnew[i] = q[i] / qn
		self.r_X[6:10] = qnew

	def suppress_bias(self):
		self.r_X[10:] = 0
		self.r_P[10:,:] = zeros(4,14)
		self.r_P[:,10:] = zeros(14,4)

	def configure(self, mag_var=None, gyro_var=None, accel_var=None, baro_var=None, gps_var=None):
		""" configure the INS parameters """

		Q = [default_gyro_var[0], default_gyro_var[1], default_gyro_var[2],
		     default_accel_var[0], default_accel_var[1], default_accel_var[2],
		     5e-7, 5e-7, 2e-6, 5e-4]
		R = [default_gps_var[0], default_gps_var[0], default_gps_var[2], 
		     default_gps_var[1], default_gps_var[1], default_gps_var[2],
		     default_mag_var[0], default_mag_var[1], default_mag_var[2], default_baro_var]

		if mag_var is not None:
			self.R[6,6] = mag_var[0]
			self.R[7,7] = mag_var[1]
			self.R[8,8] = mag_var[2]
		if gyro_var is not None:
			self.Q[0,0] = gyro_var[0]
			self.Q[1,1] = gyro_var[1]
			self.Q[2,2] = gyro_var[2]
		if accel_var is not None:
			self.Q[3,3] = accel_var[0]
			self.Q[4,4] = accel_var[1]
			self.Q[5,5] = accel_var[2]
		if baro_var is not None:
			self.R[9,9] = baro_var
		if gps_var is not None:
			self.R[0,0] = gps_var[0]
			self.R[1,1] = gps_var[0]
			self.R[3,3] = gps_var[1]
			self.R[4,4] = gps_var[1]
			self.R[2,2] = gps_var[2]
			self.R[5,5] = gps_var[2]

	def predict(self, gyros, accels, dT = 1.0/666.0):
		""" Perform the prediction step
		"""

		U = numpy.concatenate((gyros, accels))

		# fourth-order runga kuta state prediction
		k1 = self.l_Xd(self.r_X.tolist(), U) 
		k2 = self.l_Xd((self.r_X + 0.5*dT*k1).tolist(), U)
		k3 = self.l_Xd((self.r_X + 0.5*dT*k2).tolist(), U)
		k4 = self.l_Xd((self.r_X + dT*k3).tolist() , U)
		d = (k1 + 2*k2 + 2*k3 + k4) / 6

		f = self.l_F(self.r_X.tolist(), U)
		g = self.l_G(self.r_X.T.tolist())

		self.r_X = self.r_X + dT * d

		P = self.r_P
		#self.r_P = P + dT * (numpy.f*P + P*f.T) + (dT**2) * g * diag(self.Q) * g.T
		#self.r_P = (eye(NUMX)+F*T)*Pplus*(eye(NUMX)+F*T)' + T^2*G*diag(Q)*G'
		I = numpy.matrix(numpy.identity(14))
		self.r_P = (I + f * dT) * P * (I + f * dT).T + (dT**2) * g * diag(self.Q) * g.T

		self.normalize()

		self.state[0:14] = self.r_X[0:14].T
		self.state[-1] = self.r_X[-1]

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
			Z.extend([[pos[0]]])
			Z.extend([[pos[1]]])

		if vel is not None:
			idx.extend((3,4,5))
			Z.extend([[vel[0]]])
			Z.extend([[vel[1]]])
			Z.extend([[vel[2]]])

		if mag is not None:

			if self.MAG_HEADING:
				# Remove the influence of attitude
				q0,q1,q2,q3 = self.r_X[6:10]
				k1 = power( (q0*q1*2.0 + q2*q3*2.0)**2 + (q0*q0-q1*q1-q2*q2+q3*q3)**2, -0.5)
				k2 = sqrt(1.0 - (q0*q2*2.0  - q1*q3*2.0)**2)

				Rbh = numpy.zeros((3,3))
				Rbh[0,0] = k2
				Rbh[0,2] = q0*q2*-2.0+q1*q3*2.0
				Rbh[1,0] = k1*(q0*q1*2.0+q2*q3*2.0)*(q0*q2*2.0-q1*q3*2.0)
				Rbh[1,1] = k1*(q0*q0-q1*q1-q2*q2+q3*q3)
				Rbh[1,2] = k1*sqrt(-power(q0*q2*2.0-q1*q3*2.0, 2.0)+1.0)*(q0*q1*2.0+q2*q3*2.0)
				Rbh[2,0] = k1*(q0*q2*2.0-q1*q3*2.0)*(q0*q0-q1*q1-q2*q2+q3*q3)
				Rbh[2,1] = -k1*(q0*q1*2.0+q2*q3*2.0)
				Rbh[2,2] = k1*k2*(q0*q0-q1*q1-q2*q2+q3*q3)

				print "Here: " + `Rbh.shape` + " " + `mag.shape`
				print `Rbh.dot(mag).shape`
				mag = Rbh.dot(mag)
				Z.extend([[mag[0]],[mag[1]]])
			else:
				# Use full mag shape
				Z.extend(mag[0:2])				

			idx.extend((6,7))

		if baro is not None:
			idx.append(9)
			Z.append(baro)

		# construct appropriately sized predictions based on provided inputs
		# this method of creating a matrix from a list is really ugly and
		# sensitive to how the elements are formatted in the list to create
		# the correct shape of the matrix
		Z = numpy.matrix(Z)
		Y = Y[idx]
		H = H[idx,:]
		R = self.R[idx,:][:,idx]

		# calculate Kalman gain matrix
		# K = P*H.T/(H*P*H.T + R);
		A = numpy.matrix(P*H.T)
		B = numpy.matrix(H*P*H.T + R)
		K = numpy.linalg.lstsq(B, A.T)[0].T

		self.normalize()

		self.r_X = self.r_X + K*(Z-Y)

		self.r_P = P - K*H*P;

		self.state[0:14] = self.r_X[0:14].T
		self.state[-1] = self.r_X[-1]

def test():
	""" test the INS with simulated data
	"""

	from numpy import cos, sin

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,2)

	ins = PyINS()
	ins.prepare()

	dT = 1.0 / 666.0

	STEPS = 100000
	history = numpy.zeros((STEPS,14))
	times = numpy.zeros((STEPS,1))

	ins.r_X[6:10] = numpy.matrix([1/sqrt(2),0,0,1/sqrt(2)]).T
	for k in range(STEPS):
		ROLL = 0.5
		YAW  = 0.5
		ins.predict([0,0,YAW], [0,PyINS.GRAV*sin(ROLL),-PyINS.GRAV*cos(ROLL) - 0.1], dT=0.0015)

		history[k,:] = ins.r_X.T
		times[k] = k * dT

		angle = numpy.pi/3 + YAW * dT * k # radians 
		height = 1.0 * k * dT

		if True and k % 60 == 59:
			ins.correction(pos=[[10],[5],[-height]])

		if True and k % 60 == 59:
			ins.correction(vel=[[0],[0],[0]])

		if k % 20 == 8:
			ins.correction(baro=[height])

		if True and k % 20 == 15:
			ins.correction(mag=[[400 * cos(angle)], [-400 * sin(angle)], [1600]])

		ins.normalize()
		if k < 200:
			ins.suppress_bias()

		if k % 50 == 0:
			print `k` + " Att: " + `quat_rpy_display(ins.r_X[6:10])` + " norm: " + `Matrix(ins.r_X[6:10]).norm()`

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

if  __name__ =='__main__':
    test()


