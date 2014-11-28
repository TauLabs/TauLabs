from pyins import PyINS
import unittest

from sympy import symbols, lambdify, sqrt
from sympy import MatrixSymbol, Matrix
from numpy import cos, sin, power
from sympy.matrices import *
from quaternions import *
import numpy
import ins

VISUALIZE = False

class StaticTestFunctions(unittest.TestCase):

    def setUp(self):
        self.sim = PyINS()
        self.sim.prepare()

    def run_static(self, accel=[0.0,0.0,-PyINS.GRAV],
        gyro=[0.0,0.0,0.0], mag=[400,0,1600],
        pos=[0,0,0], vel=[0,0,0],
        noise=False, STEPS=200000):
        """ simulate a static set of inputs and measurements
        """

        sim = self.sim

        dT = 1.0 / 666.0

        #numpy.random.seed(1)

        history = numpy.zeros((STEPS,16))
        history_rpy = numpy.zeros((STEPS,3))
        times = numpy.zeros((STEPS,1))

        for k in range(STEPS):
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

            sim.predict(gyro+ng, accel+na, dT=dT)

            history[k,:] = sim.state
            history_rpy[k,:] = quat_rpy(sim.state[6:10])
            times[k] = k * dT

            if True and k % 60 == 59:
                sim.correction(pos=pos+np)

            if True and k % 60 == 59:
                sim.correction(vel=vel+nv)

            if k % 20 == 8:
                sim.correction(baro=-pos[2]+np[2])

            if True and k % 20 == 15:
                sim.correction(mag=mag+nm)

        if VISUALIZE:
            from numpy import cos, sin

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2,2)

            k = STEPS

            ax[0][0].cla()
            ax[0][0].plot(times[0:k:4],history[0:k:4,0:3])
            ax[0][0].set_title('Position')
            plt.sca(ax[0][0])
            plt.ylabel('m')
            ax[0][1].cla()
            ax[0][1].plot(times[0:k:4],history[0:k:4,3:6])
            ax[0][1].set_title('Velocity')
            plt.sca(ax[0][1])
            plt.ylabel('m/s')
            #plt.ylim(-2,2)
            ax[1][0].cla()
            ax[1][0].plot(times[0:k:4],history_rpy[0:k:4,:])
            ax[1][0].set_title('Attitude')
            plt.sca(ax[1][0])
            plt.ylabel('Angle (Deg)')
            plt.xlabel('Time (s)')
            #plt.ylim(-1.1,1.1)
            ax[1][1].cla()
            ax[1][1].plot(times[0:k:4],history[0:k:4,10:])
            ax[1][1].set_title('Biases')
            plt.sca(ax[1][1])
            plt.ylabel('Bias (rad/s)')
            plt.xlabel('Time (s)')

            plt.suptitle(unittest.TestCase.shortDescription(self))
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
        """ test convergence to face west
        """

        mag = [0,-400,1600]
        state, history, times = self.run_static(mag=mag, STEPS=50000)
        self.assertState(state,rpy=[0,0,90])

    def test_pos_offset(self):
        """ test convergence to location away from origin
        """

        pos = [10,5,7]
        state, history, times = self.run_static(pos=pos, STEPS=50000)
        self.assertState(state,pos=pos)

    def test_accel_bias(self):
        """ test convergence with biased accelerometers
        """

        bias = -0.2
        state, history, times = self.run_static(accel=[0,0,-PyINS.GRAV+bias], STEPS=50000)
        self.assertState(state,bias=[0,0,0,0,0,bias])

    def test_gyro_bias(self):
        """ test convergence with biased gyros
        """

        state, history, times = self.run_static(gyro=[0.1,-0.05,0.06], STEPS=150000)
        self.assertState(state,bias=[0.1,-0.05,0.06,0,0,0])

    def test_init_100m(self):
        """ test convergence to origin when initialized 100m away
        """

        ins.set_state(pos=numpy.array([100.0,100.0,50]))
        state, history, times = self.run_static(STEPS=50000)
        self.assertState(state)

    def test_init_bad_q(self):
        """ test convergence with bad initial attitude
        """

        ins.set_state(q=numpy.array([0.7, 0.7, 0, 0]))
        state, history, times = self.run_static() #STEPS=5000)
        self.assertState(state)

    def test_init_bad_bias(self):
        """ test convergence with bad initial gyro biases
        """

        ins.set_state(gyro_bias=numpy.array([0.05,0.05,-0.05]))
        state, history, times = self.run_static(STEPS=100000)
        self.assertState(state)

    def test_init_bad_q_bias(self):
        """ test convergence with bad initial attitude and biases
        """

        ins.set_state(q=numpy.array([0.7, 0.7, 0, 0]),gyro_bias=numpy.array([0.05,0.05,-0.05]))
        state, history, times = self.run_static(STEPS=100000)
        self.assertState(state)

    def test_mag_offset(self):
        """ test convergence with an incorrectly scaled mag
        """

        mag = [300,0,900]
        state, history, times = self.run_static(mag=mag,STEPS=50000)
        self.assertState(state)

    def test_origin(self):
        """ test convergence at origin
        """

        state, history, times = self.run_static(STEPS=50000)
        self.assertState(state)

    def test_stable(self):
        """ test stability at origin with noise
        """

        state, history, times = self.run_static(STEPS=50000, noise=True)
        self.assertState(state)

class ReplayFlightFunctions(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ReplayFlightFunctions, self).__init__(*args, **kwargs)
        self.uavolist = []

    def setUp(self):
        self.sim = PyINS()
        self.sim.prepare()

        # add python directory to search path so the modules can be loaded
        import sys,os
        sys.path.insert(1, os.path.dirname(sys.path[0]))

        import taulabs
        import cPickle

        filename = "/Users/Cotton/Documents/TauLabs/Logs/20140818_nav_testing/log_20140327_183850.dat.pickle"
        filename = "/Users/Cotton/Documents/TauLabs/Logs/20140820_navigation/log_20140820_232224.dat.pickle"

        # load data from file
        handle = open(filename, 'rb')
        githash = cPickle.load(handle)
        uavo_parsed = cPickle.load(handle)
        handle.close()

        # prepare the parser
        uavo_defs = taulabs.uavo_collection.UAVOCollection()
        uavo_defs.from_git_hash(githash)
        parser = taulabs.uavtalk.UavTalk(uavo_defs)

        print "Converting log records into python objects"
        uavo_list = taulabs.uavo_list.UAVOList(uavo_defs)
        for obj_id, data, timestamp in uavo_parsed:
            obj = uavo_defs[obj_id]
            u = obj.instance_from_bytes(data, timestamp)
            uavo_list.append(u)

        # We're done with this (potentially very large) variable, delete it.
        del uavo_parsed

        self.uavo_list = uavo_list

    def test_replay(self):
        """ replay a logfile """

        from pyins import run_uavo_list
        run_uavo_list(self.uavo_list)


if __name__ == '__main__':
    selected_test = None

    if selected_test is not None:
        VISUALIZE = True
        suite = unittest.TestSuite()
        suite.addTest(StaticTestFunctions(selected_test))
        unittest.TextTestRunner().run(suite)
    else:
        unittest.main()