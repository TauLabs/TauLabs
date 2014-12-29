from cins import CINS
from pyins import PyINS
import unittest

from sympy import symbols, lambdify, sqrt
from sympy import MatrixSymbol, Matrix
from numpy import cos, sin, power
from sympy.matrices import *
from quaternions import *
import numpy
import math
import ins

VISUALIZE = False

class CompareFunctions(unittest.TestCase):

    def setUp(self):

        self.c_sim = CINS()
        self.py_sim = PyINS()

        self.c_sim.prepare()
        self.py_sim.prepare()

    def run_static(self, accel=[0.0,0.0,-PyINS.GRAV],
        gyro=[0.0,0.0,0.0], mag=[400,0,1600],
        pos=[0,0,0], vel=[0,0,0],
        noise=False, STEPS=200000):
        """ simulate a static set of inputs and measurements
        """

        c_sim = self.c_sim
        py_sim = self.py_sim

        dT = 1.0 / 666.0

        numpy.random.seed(1)

        c_history = numpy.zeros((STEPS,16))
        c_history_rpy = numpy.zeros((STEPS,3))
        py_history = numpy.zeros((STEPS,16))
        py_history_rpy = numpy.zeros((STEPS,3))
        times = numpy.zeros((STEPS,1))

        for k in range(STEPS):
            print `k`
            ng = numpy.zeros(3,)
            na = numpy.zeros(3,)
            np = numpy.zeros(3,)
            nv = numpy.zeros(3,)
            nm = numpy.zeros(3,)

            if noise:
                ng = numpy.random.randn(3,) * 1e-3
                na = numpy.random.randn(3,) * 1e-3
                np = numpy.random.randn(3,) * 1e-3
                nv = numpy.random.randn(3,) * 1e-3
                nm = numpy.random.randn(3,) * 10.0

            c_sim.predict(gyro+ng, accel+na, dT=dT)
            py_sim.predict(gyro+ng, accel+na, dT=dT)

            times[k] = k * dT
            c_history[k,:] = c_sim.state
            c_history_rpy[k,:] = quat_rpy(c_sim.state[6:10])
            py_history[k,:] = py_sim.state
            py_history_rpy[k,:] = quat_rpy(py_sim.state[6:10])

            if False and k % 60 == 59:
                c_sim.correction(pos=pos+np)
                py_sim.correction(pos=pos+np)

            if False and k % 60 == 59:
                c_sim.correction(vel=vel+nv)
                py_sim.correction(vel=vel+nv)

            if True and k % 20 == 8:
                c_sim.correction(baro=-pos[2]+np[2])
                py_sim.correction(baro=-pos[2]+np[2])

            if True and k % 20 == 15:
                c_sim.correction(mag=mag+nm)
                py_sim.correction(mag=mag+nm)

            self.assertState(c_sim.state, py_sim.state)

        if VISUALIZE:
            from numpy import cos, sin

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2,2)

            k = STEPS

            ax[0][0].cla()
            ax[0][0].plot(times[0:k:4],c_history[0:k:4,0:3])
            ax[0][0].set_title('Position')
            plt.sca(ax[0][0])
            plt.ylabel('m')
            ax[0][1].cla()
            ax[0][1].plot(times[0:k:4],c_history[0:k:4,3:6])
            ax[0][1].set_title('Velocity')
            plt.sca(ax[0][1])
            plt.ylabel('m/s')
            #plt.ylim(-2,2)
            ax[1][0].cla()
            ax[1][0].plot(times[0:k:4],c_history_rpy[0:k:4,:])
            ax[1][0].set_title('Attitude')
            plt.sca(ax[1][0])
            plt.ylabel('Angle (Deg)')
            plt.xlabel('Time (s)')
            #plt.ylim(-1.1,1.1)
            ax[1][1].cla()
            ax[1][1].plot(times[0:k:4],c_history[0:k:4,10:])
            ax[1][1].set_title('Biases')
            plt.sca(ax[1][1])
            plt.ylabel('Bias (rad/s)')
            plt.xlabel('Time (s)')

            plt.suptitle(unittest.TestCase.shortDescription(self))
            plt.show()


        return sim.state, history, times

    def assertState(self, c_state, py_state):
        """ check that the state is near a desired position
        """

        # check position
        self.assertAlmostEqual(c_state[0],py_state[0],places=1)
        self.assertAlmostEqual(c_state[1],py_state[1],places=1)
        self.assertAlmostEqual(c_state[2],py_state[2],places=1)

        # check velocity
        self.assertAlmostEqual(c_state[3],py_state[3],places=1)
        self.assertAlmostEqual(c_state[4],py_state[4],places=1)
        self.assertAlmostEqual(c_state[5],py_state[5],places=1)

        # check attitude
        self.assertAlmostEqual(c_state[0],py_state[0],places=0)
        self.assertAlmostEqual(c_state[1],py_state[1],places=0)
        self.assertAlmostEqual(c_state[2],py_state[2],places=0)
        self.assertAlmostEqual(c_state[3],py_state[3],places=0)

        # check bias terms (gyros and accels)
        self.assertAlmostEqual(c_state[10],py_state[10],places=2)
        self.assertAlmostEqual(c_state[11],py_state[11],places=2)
        self.assertAlmostEqual(c_state[12],py_state[12],places=2)
        self.assertAlmostEqual(c_state[13],py_state[13],places=2)
        self.assertAlmostEqual(c_state[14],py_state[14],places=2)
        self.assertAlmostEqual(c_state[15],py_state[15],places=2)

    def test_face_west(self):
        """ test convergence to face west
        """

        mag = [0,-400,1600]
        state, history, times = self.run_static(mag=mag, STEPS=50000)
        self.assertState(state,rpy=[0,0,90])


if __name__ == '__main__':
    selected_test = None

    if selected_test is not None:
        VISUALIZE = True
        suite = unittest.TestSuite()
        suite.addTest(CompareFunctions(selected_test))
        unittest.TextTestRunner().run(suite)
    else:
        unittest.main()