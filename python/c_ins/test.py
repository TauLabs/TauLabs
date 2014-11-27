from pyins import PyINS
import unittest

from sympy import symbols, lambdify, sqrt
from sympy import MatrixSymbol, Matrix
from numpy import cos, sin, power
from sympy.matrices import *
from quaternions import *
import numpy
import ins

VISUALIZE = True

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.sim = PyINS()
        self.sim.prepare()

    def run_static(self, accel=[0,0,-PyINS.GRAV], gyro=[0,0,0], mag=[400,0,1600], pos=[0,0,0], vel=[0,0,0], STEPS=200000):
        """ simulate a static set of inputs and measurements
        """

        sim = self.sim

        dT = 1.0 / 666.0

        history = numpy.zeros((STEPS,16))
        times = numpy.zeros((STEPS,1))

        U = gyro
        U.extend(accel)

        for k in range(STEPS):
            sim.predict(U=U, dT=dT)

            history[k,:] = sim.state
            times[k] = k * dT

            height = 1.0 * k * dT

            if True and k % 60 == 59:
                sim.correction(pos=pos)

            if True and k % 60 == 59:
                sim.correction(vel=vel)

            if k % 20 == 8:
                sim.correction(baro=-pos[2])

            if True and k % 20 == 15:
                sim.correction(mag=mag)

        if VISUALIZE:
            from numpy import cos, sin

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2,2)

            k = STEPS

            ax[0][0].cla()
            ax[0][0].plot(times[0:k:4],history[0:k:4,0:3])
            ax[0][0].set_title('Position')
            ax[0][1].cla()
            ax[0][1].plot(times[0:k:4],history[0:k:4,3:6])
            ax[0][1].set_title('Velocity')
            plt.sca(ax[0][1])
            plt.ylim(-2,2)
            ax[1][0].cla()
            ax[1][0].plot(times[0:k:4],history[0:k:4,6:10])
            ax[1][0].set_title('Attitude')
            ax[1][1].cla()
            ax[1][1].plot(times[0:k:4],history[0:k:4,10:])
            ax[1][1].set_title('Biases')

            plt.show()


        return sim.state, history, times

    def assertState(self, state, pos=[0,0,0], vel=[0,0,0], rpy=[0,0,0], bias=[0,0,0,0,0,0]):
        """ check that the state is near a desired position
        """

        # check position
        self.assertAlmostEqual(state[0],pos[0],places=1)
        self.assertAlmostEqual(state[1],pos[1],places=1)
        self.assertAlmostEqual(state[2],pos[2],places=1)

        # check velocity
        self.assertAlmostEqual(state[3],vel[0],places=1)
        self.assertAlmostEqual(state[4],vel[1],places=1)
        self.assertAlmostEqual(state[5],vel[2],places=1)

        # check attitude (in degrees)
        s_rpy = quat_rpy(state[6:10])
        self.assertAlmostEqual(s_rpy[0],rpy[0],places=0)
        self.assertAlmostEqual(s_rpy[1],rpy[1],places=0)
        self.assertAlmostEqual(s_rpy[2],rpy[2],places=0)

        # check bias terms (gyros and accels)
        self.assertAlmostEqual(state[10],bias[0],places=2)
        self.assertAlmostEqual(state[11],bias[1],places=2)
        self.assertAlmostEqual(state[12],bias[2],places=2)
        self.assertAlmostEqual(state[13],bias[3],places=2)
        self.assertAlmostEqual(state[14],bias[4],places=2)
        self.assertAlmostEqual(state[15],bias[5],places=2)

    def test_face_west(self):
        """ test convergence at origin
        """

        mag = [0,-400,1600]
        state, history, times = self.run_static(mag=mag, STEPS=5000)
        self.assertState(state,rpy=[0,0,90])

    def test_pos_offset(self):
        """ test convergence at origin
        """

        self.sim.configure(gps_var=numpy.array([1e-7,1e-7,1e-3]),baro_var=1e-3)

        pos = [10,5,7]
        state, history, times = self.run_static(pos=pos, STEPS=5000)
        self.assertState(state,pos=pos)

    def test_origin(self):
        """ test convergence at origin
        """

        state, history, times = self.run_static(STEPS=5000)
        self.assertState(state)

if __name__ == '__main__':
    unittest.main()