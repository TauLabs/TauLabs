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

C_IMP = True

class StaticTestFunctions(unittest.TestCase):

    def setUp(self):
        if C_IMP:
            self.sim = CINS()
        else:
            self.sim = PyINS()
        self.sim.prepare()

    def run_static(self, accel=[0.0,0.0,-CINS.GRAV],
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
            fig, ax = plt.subplots(2,2,sharex=True)

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

            ax[1][0].cla()
            ax[1][0].plot(times[0:k:4],history_rpy[0:k:4,:])
            ax[1][0].set_title('Attitude')
            plt.sca(ax[1][0])
            plt.ylabel('Angle (Deg)')
            plt.xlabel('Time (s)')

            ax[1][1].cla()
            ax[1][1].plot(times[0:k:4],history[0:k:4,10:13],label="Gyro")
            ax[1][1].plot(times[0:k:4],history[0:k:4,-1],label="Accel")
            ax[1][1].set_title('Biases')
            plt.sca(ax[1][1])
            plt.ylabel('Bias (rad/s)')
            plt.xlabel('Time (s)')
            #plt.legend()

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
        self.assertTrue(abs(math.fmod(s_rpy[2]-rpy[2],360)) < 1)

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

        bias = -0.20
        state, history, times = self.run_static(accel=[0,0,-PyINS.GRAV+bias], STEPS=150000)
        self.assertState(state,bias=[0,0,0,0,0,bias])

    def test_gyro_bias(self):
        """ test convergence with biased gyros
        """

        state, history, times = self.run_static(gyro=[0.1,-0.05,0.06], STEPS=150000)
        self.assertState(state,bias=[0.1,-0.05,0.06,0,0,0])

    def test_gyro_bias_xy_rate(self):
        """ test gyro xy bias converges within 10% in a fixed time
        """

        TIME = 10 # seconds
        FS = 666  # sampling rate
        MAX_ERR = 0.1
        BIAS = 10.0 * math.pi / 180.0

        state, history, times = self.run_static(gyro=[BIAS,-BIAS,0], STEPS=TIME*FS)

        self.assertAlmostEqual(state[10], BIAS, delta=BIAS*MAX_ERR)
        self.assertAlmostEqual(state[11], -BIAS, delta=BIAS*MAX_ERR)

    def test_gyro_bias_z_rate(self):
        """ test gyro z bias converges within 10% in a fixed time
        """

        TIME = 10 # seconds
        FS = 666  # sampling rate
        MAX_ERR = 0.1
        BIAS = 10.0 * math.pi / 180.0

        state, history, times = self.run_static(gyro=[0,0,BIAS], STEPS=TIME*FS)

        self.assertAlmostEqual(state[12], BIAS, delta=BIAS*MAX_ERR)

    def test_accel_bias_z_rate(self):
        """ test accel z bias converges within 10% in a fixed time
        """

        TIME = 30 # seconds
        FS = 666  # sampling rate
        MAX_ERR = 0.1
        BIAS = 1

        state, history, times = self.run_static(accel=[0,0,-PyINS.GRAV+BIAS], STEPS=TIME*FS)

        self.assertAlmostEqual(state[15], BIAS, delta=BIAS*MAX_ERR)

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

class StepTestFunctions(unittest.TestCase):

    def setUp(self):
        if C_IMP:
            self.sim = CINS()
        else:
            self.sim = PyINS()
        self.sim.prepare()

    def run_step(self, accel=[0.0,0.0,-CINS.GRAV],
        gyro=[0.0,0.0,0.0], mag=[400,0,1600],
        pos=[0,0,0], vel=[0,0,0],
        noise=False, STEPS=200000,
        CHANGE=6660*3):
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

            if k < CHANGE:
                sim.predict(ng, numpy.array([0,0,-CINS.GRAV])+na, dT=dT)
            else:
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
            fig, ax = plt.subplots(2,2,sharex=True)

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
            ax[1][1].plot(times[0:k:4],history[0:k:4,10:13],label="Gyro")
            ax[1][1].plot(times[0:k:4],history[0:k:4,-1],label="Accel")
            ax[1][1].set_title('Biases')
            plt.sca(ax[1][1])
            plt.ylabel('Bias (rad/s)')
            plt.xlabel('Time (s)')
            plt.legend()

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
        self.assertTrue(abs(math.fmod(s_rpy[2]-rpy[2],360)) < 1)

        # check bias terms (gyros and accels)
        self.assertAlmostEqual(state[10],bias[0],places=2)
        self.assertAlmostEqual(state[11],bias[1],places=2)
        self.assertAlmostEqual(state[12],bias[2],places=2)
        self.assertAlmostEqual(state[13],bias[3],places=2)
        self.assertAlmostEqual(state[14],bias[4],places=2)
        self.assertAlmostEqual(state[15],bias[5],places=2)

    def test_accel_bias(self):
        """ test convergence with biased accelerometers
        """

        bias = -0.20
        state, history, times = self.run_step(accel=[0,0,-PyINS.GRAV+bias], STEPS=90*666)
        self.assertState(state,bias=[0,0,0,0,0,bias])

    def test_gyro_bias(self):
        """ test convergence with biased gyros
        """

        state, history, times = self.run_step(gyro=[0.1,-0.05,0.06], STEPS=90*666)
        self.assertState(state,bias=[0.1,-0.05,0.06,0,0,0])

    def test_gyro_bias_xy_rate(self):
        """ test gyro xy bias converges within 10% in a fixed time
        """

        TIME = 90 # seconds
        FS = 666  # sampling rate
        MAX_ERR = 0.1
        BIAS = 10.0 * math.pi / 180.0

        state, history, times = self.run_step(gyro=[BIAS,-BIAS,0], STEPS=TIME*FS)

        self.assertAlmostEqual(state[10], BIAS, delta=BIAS*MAX_ERR)
        self.assertAlmostEqual(state[11], -BIAS, delta=BIAS*MAX_ERR)

    def test_gyro_bias_z_rate(self):
        """ test gyro z bias converges within 10% in a fixed time
        """

        TIME = 90 # seconds
        FS = 666  # sampling rate
        MAX_ERR = 0.1
        BIAS = 10.0 * math.pi / 180.0

        state, history, times = self.run_step(gyro=[0,0,BIAS], STEPS=TIME*FS)

        self.assertAlmostEqual(state[12], BIAS, delta=BIAS*MAX_ERR)

    def test_accel_bias_z_rate(self):
        """ test accel z bias converges within 10% in a fixed time
        """

        TIME = 90 # seconds
        FS = 666  # sampling rate
        MAX_ERR = 0.1
        BIAS = 1

        state, history, times = self.run_step(accel=[0,0,-PyINS.GRAV+BIAS], STEPS=TIME*FS)

        self.assertAlmostEqual(state[15], BIAS, delta=BIAS*MAX_ERR)

class SimulatedFlightTests(unittest.TestCase):

    def setUp(self):
        if C_IMP:
            self.sim = CINS()
        else:
            self.sim = PyINS()
        self.sim.prepare()

        import simulation
        self.model = simulation.Simulation()

    def assertState(self, state, pos=[0,0,0], vel=[0,0,0], rpy=[0,0,0], bias=[0,0,0,0,0,0]):
        """ check that the state is near a desired position
        """

        # check position
        self.assertAlmostEqual(state[0],pos[0],places=0)
        self.assertAlmostEqual(state[1],pos[1],places=0)
        self.assertAlmostEqual(state[2],pos[2],places=0)

        # check velocity
        self.assertAlmostEqual(state[3],vel[0],places=1)
        self.assertAlmostEqual(state[4],vel[1],places=1)
        self.assertAlmostEqual(state[5],vel[2],places=1)

        # check attitude (in degrees)
        s_rpy = quat_rpy(state[6:10])
        self.assertAlmostEqual(s_rpy[0],rpy[0],places=0)
        self.assertAlmostEqual(s_rpy[1],rpy[1],places=0)
        self.assertTrue(abs(math.fmod(s_rpy[2]-rpy[2],360)) < 5)

        # check bias terms (gyros and accels)
        self.assertAlmostEqual(state[10],bias[0],places=0)
        self.assertAlmostEqual(state[11],bias[1],places=0)
        self.assertAlmostEqual(state[12],bias[2],places=0)
        self.assertAlmostEqual(state[13],bias[3],places=0)
        self.assertAlmostEqual(state[14],bias[4],places=0)
        self.assertAlmostEqual(state[15],bias[5],places=1)

    def plot(self, times, history, history_rpy, true_pos, true_vel, true_rpy):
        from numpy import cos, sin

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,2,sharex=True)

        k = times.size

        ax[0][0].cla()
        ax[0][0].plot(times[0:k:4],true_pos[0:k:4,:],'k--')
        ax[0][0].plot(times[0:k:4],history[0:k:4,0:3])
        ax[0][0].set_title('Position')
        plt.sca(ax[0][0])
        plt.ylabel('m')

        ax[0][1].cla()
        ax[0][1].plot(times[0:k:4],true_vel[0:k:4,:],'k--')
        ax[0][1].plot(times[0:k:4],history[0:k:4,3:6])
        ax[0][1].set_title('Velocity')
        plt.sca(ax[0][1])
        plt.ylabel('m/s')

        ax[1][0].cla()
        ax[1][0].plot(times[0:k:4],true_rpy[0:k:4,:],'k--')
        ax[1][0].plot(times[0:k:4],history_rpy[0:k:4,:])
        ax[1][0].set_title('Attitude')
        plt.sca(ax[1][0])
        plt.ylabel('Angle (Deg)')
        plt.xlabel('Time (s)')

        ax[1][1].cla()
        ax[1][1].plot(times[0:k:4],history[0:k:4,10:13])
        ax[1][1].plot(times[0:k:4],history[0:k:4,-1])
        ax[1][1].set_title('Biases')
        plt.sca(ax[1][1])
        plt.ylabel('Bias (rad/s)')
        plt.xlabel('Time (s)')

        plt.suptitle(unittest.TestCase.shortDescription(self))
        plt.show()

    def test_circle(self, STEPS=50000):
        """ test that the INS gets a good fit for a simulated flight of 
        circles
        """

        sim = self.sim
        model = self.model

        dT = 1.0 / 666.0

        numpy.random.seed(1)

        history = numpy.zeros((STEPS,16))
        history_rpy = numpy.zeros((STEPS,3))

        true_pos = numpy.zeros((STEPS,3))
        true_vel = numpy.zeros((STEPS,3))
        true_rpy = numpy.zeros((STEPS,3))

        times = numpy.zeros((STEPS,1))

        for k in range(STEPS):

            model.fly_circle(dT=dT)

            true_pos[k,:] = model.get_pos()
            true_vel[k,:] = model.get_vel()
            true_rpy[k,:] = model.get_rpy()

            ng = numpy.random.randn(3,) * 1e-3
            na = numpy.random.randn(3,) * 1e-3
            np = numpy.random.randn(3,) * 1e-3
            nv = numpy.random.randn(3,) * 1e-3
            nm = numpy.random.randn(3,) * 10.0

            # convert from rad/s to deg/s
            gyro = model.get_gyro() / 180.0 * math.pi
            accel = model.get_accel()

            sim.predict(gyro+ng, accel+na, dT=dT)

            if True and k % 60 == 59:
                sim.correction(pos=true_pos[k,:]+np)

            if True and k % 60 == 59:
                sim.correction(vel=true_vel[k,:]+nv)

            if k % 20 == 8:
                sim.correction(baro=-true_pos[k,2]+np[2])

            if True and k % 20 == 15:
                sim.correction(mag=model.get_mag()+nm)

            history[k,:] = sim.state
            history_rpy[k,:] = quat_rpy(sim.state[6:10])
            times[k] = k * dT

            if k > 100:
                numpy.testing.assert_almost_equal(sim.state[0:3], true_pos[k,:], decimal=0)
                numpy.testing.assert_almost_equal(sim.state[3:6], true_vel[k,:], decimal=0)
                # only test roll and pitch because of wraparound issues
                numpy.testing.assert_almost_equal(history_rpy[k,0:2], true_rpy[k,0:2], decimal=1)

        if VISUALIZE:
            self.plot(times, history, history_rpy, true_pos, true_vel, true_rpy)

        return sim.state, history, times

    def test_gyro_bias_circle(self):
        """ test that while flying a circle the location converges
        """

        STEPS=100000

        sim = self.sim
        model = self.model

        dT = 1.0 / 666.0

        history = numpy.zeros((STEPS,16))
        history_rpy = numpy.zeros((STEPS,3))

        true_pos = numpy.zeros((STEPS,3))
        true_vel = numpy.zeros((STEPS,3))
        true_rpy = numpy.zeros((STEPS,3))

        times = numpy.zeros((STEPS,1))

        numpy.random.seed(1)

        for k in range(STEPS):

            model.fly_circle(dT=dT)

            true_pos[k,:] = model.get_pos()
            true_vel[k,:] = model.get_vel()
            true_rpy[k,:] = model.get_rpy()

            ng = numpy.random.randn(3,) * 1e-3
            na = numpy.random.randn(3,) * 1e-3
            np = numpy.random.randn(3,) * 1e-3
            nv = numpy.random.randn(3,) * 1e-3
            nm = numpy.random.randn(3,) * 10.0

            # convert from rad/s to deg/s
            gyro = model.get_gyro() / 180.0 * math.pi
            accel = model.get_accel()

            # add simulated bias
            gyro = gyro + numpy.array([0.1,-0.05,0.15])

            sim.predict(gyro+ng, accel+na, dT=dT)

            pos = model.get_pos()

            if k % 60 == 59:
                sim.correction(pos=pos+np)
            if k % 60 == 59:
                sim.correction(vel=model.get_vel()+nv)
            if k % 20 == 8:
                sim.correction(baro=-pos[2]+np[2])
            if k % 20 == 15:
                sim.correction(mag=model.get_mag()+nm)

            history[k,:] = sim.state
            history_rpy[k,:] = quat_rpy(sim.state[6:10])
            times[k] = k * dT

        if VISUALIZE:
            self.plot(times, history, history_rpy, true_pos, true_vel, true_rpy)

        self.assertState(sim.state, pos=model.get_pos(), vel=model.get_vel(), rpy=model.get_rpy(), bias=[0,0,0,0,0,0])

    def test_accel_bias_circle(self):
        """ test that while flying a circle the location converges
        """

        STEPS=100000

        sim = self.sim
        model = self.model

        dT = 1.0 / 666.0

        history = numpy.zeros((STEPS,16))
        history_rpy = numpy.zeros((STEPS,3))

        true_pos = numpy.zeros((STEPS,3))
        true_vel = numpy.zeros((STEPS,3))
        true_rpy = numpy.zeros((STEPS,3))

        times = numpy.zeros((STEPS,1))

        numpy.random.seed(1)

        for k in range(STEPS):

            model.fly_circle(dT=dT)

            true_pos[k,:] = model.get_pos()
            true_vel[k,:] = model.get_vel()
            true_rpy[k,:] = model.get_rpy()

            ng = numpy.random.randn(3,) * 1e-3
            na = numpy.random.randn(3,) * 1e-3
            np = numpy.random.randn(3,) * 1e-3
            nv = numpy.random.randn(3,) * 1e-3
            nm = numpy.random.randn(3,) * 10.0

            # convert from rad/s to deg/s
            gyro = model.get_gyro() / 180.0 * math.pi
            accel = model.get_accel()

            # add simulated bias
            accel = accel + numpy.array([0.0,0,0.2])

            sim.predict(gyro+ng, accel+na, dT=dT)

            pos = model.get_pos()

            if k % 60 == 59:
                sim.correction(pos=pos+np)
            if k % 60 == 59:
                sim.correction(vel=model.get_vel()+nv)
            if k % 20 == 8:
                sim.correction(baro=-pos[2]+np[2])
            if k % 20 == 15:
                sim.correction(mag=model.get_mag()+nm)

            history[k,:] = sim.state
            history_rpy[k,:] = quat_rpy(sim.state[6:10])
            times[k] = k * dT

        if VISUALIZE:
            self.plot(times, history, history_rpy, true_pos, true_vel, true_rpy)

        self.assertState(sim.state, pos=model.get_pos(), vel=model.get_vel(), rpy=model.get_rpy(), bias=[0,0,0,0,0,0.2])


    def test_bad_init_q(self):
        """ test that while flying a circle the location converges with a bad initial attitude
        """

        sim = self.sim
        model = self.model

        dT = 1.0 / 666.0

        STEPS= 60 * 666

        history = numpy.zeros((STEPS,16))
        history_rpy = numpy.zeros((STEPS,3))

        true_pos = numpy.zeros((STEPS,3))
        true_vel = numpy.zeros((STEPS,3))
        true_rpy = numpy.zeros((STEPS,3))

        times = numpy.zeros((STEPS,1))

        numpy.random.seed(1)

        ins.set_state(q=numpy.array([math.sqrt(2),math.sqrt(2),0,0]))

        for k in range(STEPS):

            model.fly_circle(dT=dT)

            true_pos[k,:] = model.get_pos()
            true_vel[k,:] = model.get_vel()
            true_rpy[k,:] = model.get_rpy()

            ng = numpy.random.randn(3,) * 1e-3
            na = numpy.random.randn(3,) * 1e-3
            np = numpy.random.randn(3,) * 1e-3
            nv = numpy.random.randn(3,) * 1e-3
            nm = numpy.random.randn(3,) * 10.0

            # convert from rad/s to deg/s
            gyro = model.get_gyro() / 180.0 * math.pi
            accel = model.get_accel()

            sim.predict(gyro+ng, accel+na, dT=dT)

            pos = model.get_pos()

            if k % 60 == 59:
                sim.correction(pos=pos+np)
            if k % 60 == 59:
                sim.correction(vel=model.get_vel()+nv)
            if k % 20 == 8:
                sim.correction(baro=-pos[2]+np[2])
            if k % 20 == 15:
                sim.correction(mag=model.get_mag()+nm)

            history[k,:] = sim.state
            history_rpy[k,:] = quat_rpy(sim.state[6:10])
            times[k] = k * dT

        if VISUALIZE:
            self.plot(times, history, history_rpy, true_pos, true_vel, true_rpy)

        self.assertState(sim.state, pos=model.get_pos(), vel=model.get_vel(), rpy=model.get_rpy(), bias=[0,0,0,0,0,0])

    def rock_and_turn(self, STEPS=50000):
        """ test the biases work when rocking and turning. this tests the
            mag attitude compensation """

        sim = self.sim
        model = self.model

        dT = 1.0 / 666.0

        numpy.random.seed(1)

        history = numpy.zeros((STEPS,16))
        history_rpy = numpy.zeros((STEPS,3))

        true_pos = numpy.zeros((STEPS,3))
        true_vel = numpy.zeros((STEPS,3))
        true_rpy = numpy.zeros((STEPS,3))

        times = numpy.zeros((STEPS,1))

        for k in range(STEPS):

            model.rock_and_turn(dT=dT)

            true_pos[k,:] = model.get_pos()
            true_vel[k,:] = model.get_vel()
            true_rpy[k,:] = model.get_rpy()

            ng = numpy.random.randn(3,) * 1e-3
            na = numpy.random.randn(3,) * 1e-3
            np = numpy.random.randn(3,) * 1e-3
            nv = numpy.random.randn(3,) * 1e-3
            nm = numpy.random.randn(3,) * 10.0

            # convert from rad/s to deg/s
            gyro = model.get_gyro() / 180.0 * math.pi
            accel = model.get_accel()

            sim.predict(gyro+ng, accel+na, dT=dT)

            if True and k % 60 == 59:
                sim.correction(pos=true_pos[k,:]+np)

            if True and k % 60 == 59:
                sim.correction(vel=true_vel[k,:]+nv)

            if k % 20 == 8:
                sim.correction(baro=-true_pos[k,2]+np[2])

            if True and k % 20 == 15:
                sim.correction(mag=model.get_mag()+nm)

            history[k,:] = sim.state
            history_rpy[k,:] = quat_rpy(sim.state[6:10])
            times[k] = k * dT

            if k > 100:
                numpy.testing.assert_almost_equal(sim.state[0:3], true_pos[k,:], decimal=0)
                numpy.testing.assert_almost_equal(sim.state[3:6], true_vel[k,:], decimal=0)
                # only test roll and pitch because of wraparound issues
                numpy.testing.assert_almost_equal(history_rpy[k,0:2], true_rpy[k,0:2], decimal=1)

        if VISUALIZE:
            self.plot(times, history, history_rpy, true_pos, true_vel, true_rpy)

        return sim.state, history, times

if __name__ == '__main__':
    selected_test = None

    if selected_test is not None:
        VISUALIZE = True
        suite = unittest.TestSuite()
        suite.addTest(StaticTestFunctions(selected_test))
        unittest.TextTestRunner().run(suite)
    else:
        unittest.main()