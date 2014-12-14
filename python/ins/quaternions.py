
from sympy.matrices import *
import numpy

def quat_multiply(q,r):
	""" Compute the quaternion multiplication t = q * t """

	q0,q1,q2,q3 = q
	r0,r1,r2,r3 = r
	t = Matrix([r0*q0-r1*r1-r2*r2-r3*q3,
		r0*q1+r1*q0-r2*q3+r3*q2,
		r0*q2+r1*q3+r2*q0-r3*q1,
		r0*q3-r1*q2+r2*q1+r3*q0])
	return t

def quat_inv(q):
	""" Return the quaternion inverse """
	return Matrix([q[0],-q[1],-q[2],-q[3]])

def quat_norm(q):
	return q / numpy.linalg.norm(q)

def quat_rot_vec(q,v):
	v1 = [0,v[0],v[1],v[2]]
	return quat_multiply(quat_multiply(q,p),quat_inv(q))

def quat_rpy(q):
	RAD2DEG = 180 / numpy.pi;

	q0,q1,q2,q3 = q
	q0s,q1s,q2s,q3s = [q0**2, q1**2, q2**2, q3**2]

	R13 = numpy.double(2 * (q1 * q3 - q0 * q2))
	R11 = numpy.double(q0s + q1s - q2s - q3s)
	R12 = numpy.double(2 * (q1 * q2 + q0 * q3))
	R23 = numpy.double(2 * (q2 * q3 + q0 * q1))
	R33 = numpy.double(q0s - q1s - q2s + q3s)

	rpy = [0,0,0]
	rpy[1] = RAD2DEG * numpy.arcsin(-R13)
	rpy[2] = RAD2DEG * numpy.arctan2(R12, R11)
	rpy[0] = RAD2DEG * numpy.arctan2(R23, R33)

	return rpy

def quat_rpy_display(q):
	return "Quaternion: " + `q.T.tolist()[0]` + " RPY: " + `quat_rpy(q)`

def quat_rbe(q):

	q0s = q[0] * q[0]
	q1s = q[1] * q[1]
	q2s = q[2] * q[2]
	q3s = q[3] * q[3]

	Rbe = numpy.zeros((3,3))

	Rbe[0][0] = q0s + q1s - q2s - q3s;
	Rbe[0][1] = 2 * (q[1] * q[2] + q[0] * q[3]);
	Rbe[0][2] = 2 * (q[1] * q[3] - q[0] * q[2]);
	Rbe[1][0] = 2 * (q[1] * q[2] - q[0] * q[3]);
	Rbe[1][1] = q0s - q1s + q2s - q3s;
	Rbe[1][2] = 2 * (q[2] * q[3] + q[0] * q[1]);
	Rbe[2][0] = 2 * (q[1] * q[3] + q[0] * q[2]);
	Rbe[2][1] = 2 * (q[2] * q[3] - q[0] * q[1]);
	Rbe[2][2] = q0s - q1s - q2s + q3s;

	return Rbe