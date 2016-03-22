
import unittest
import numpy
import math

import rtkf

VISUALIZE = True

class StaticTestFunctions(unittest.TestCase):

    def setUp(self):
        rtkf.init()

    def run_static(self,
        gyro=[0.0,0.0,0.0], control=[0.0,0.0,0.0],
        STEPS=200000, dT = 1.0/400.0):
        """ simulate a static set of inputs and measurements
        """

        NUMX = 9

        gyro = numpy.array(gyro, dtype=numpy.float64)
        control = numpy.array(control, dtype=numpy.float64)

        history = numpy.zeros((STEPS,NUMX))
        times = numpy.zeros((STEPS,1))

        for k in range(STEPS):
            state = rtkf.advance(gyro, control, dT)

            history[k,:] = state
            times[k] = k * dT


        if VISUALIZE:
            from numpy import cos, sin

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2,2,sharex=True)

            k = STEPS

            ax[0][0].cla()
            ax[0][0].plot(times[0:k:4],history[0:k:4,0:3])
            ax[0][0].set_title('Rate')
            plt.sca(ax[0][0])
            plt.ylabel('deg/s')

            ax[0][1].cla()
            ax[0][1].plot(times[0:k:4],history[0:k:4,3:6])
            ax[0][1].set_title('Torque')
            plt.sca(ax[0][1])
            plt.ylabel('deg/s/s')

            ax[1][0].cla()
            ax[1][0].plot(times[0:k:4],history[0:k:4,7:9])
            ax[1][0].set_title('Bias')
            plt.sca(ax[1][0])
            plt.ylabel('deg/s/s')
            plt.xlabel('Time (s)')

            plt.suptitle(unittest.TestCase.shortDescription(self))
            plt.show()

        return state, history, times

    def check_gyro(self, gyro_expected, state):
        self.assertAlmostEqual(gyro_expected[0],state[0],places=1)
        self.assertAlmostEqual(gyro_expected[1],state[1],places=1)
        self.assertAlmostEqual(gyro_expected[2],state[2],places=1)

    def test_static(self):
        """ test stability no input """

        dT = 1. / 400.
        STEPS = int(5./dT)
        gyro = [10,-5,1]
        state, history, times = self.run_static(gyro=gyro, control=[0, 0, 0], STEPS=STEPS, dT=dT)

        self.check_gyro(gyro, state)

if __name__ == '__main__':
    unittest.main()