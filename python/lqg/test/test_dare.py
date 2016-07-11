import unittest
import numpy
import numpy.testing
import math

import dare


class StaticTestFunctions(unittest.TestCase):

    def setUp(self):
        """ Nothing to do for setup """

    def test_dare_1(self):
        """ Verify solution to discrete riccati equation  """

        A = numpy.asarray([[0.9990,    0,   0],
                           [2.6285,    0.9205,   0],
                           [-0.1103,         -0.0795,    1.0000]], dtype=numpy.float64)
        B = numpy.asarray([1,2.6312,-0.1104], dtype=numpy.float64)
        Q = numpy.asarray([[0.999, 0, 0], [0, 0.001, 0], [0, 0, 1e-5]], dtype=numpy.float64)
        R = 1001

        P = numpy.asarray([[45.9599, 0.2160, -0.0977],
                           [0.2160,  0.0079, -0.0017],
                           [-0.0977,   -0.0017, 0.0020]], dtype=numpy.float64)

        X = dare.dare(A.transpose(),B,Q,R)

        numpy.testing.assert_allclose(X,P, rtol=1e-01)

        return None

    def test_dare_2(self):
        """ Verify solution to discrete riccati equation  """

        A = numpy.asarray([[0.9990,       0,   0],
                           [32.0221, 0.9205,   0],
                           [1.3438, -0.0795,   1.0000]], dtype=numpy.float64)
        B = numpy.asarray([1,32.0542,-1.3452], dtype=numpy.float64)
        Q = numpy.asarray([[0.999, 0, 0], [0, 1e-5, 0], [0, 0, 1e-7]], dtype=numpy.float64)
        R = 1001

        P = numpy.asarray([[49.8590, 0.0235, -0.00974],
                           [0.0235,  7.71e-5, -1.52e-5],
                           [-0.00974,-1.52e-5, 1.82e-5]], dtype=numpy.float64)

        X = dare.dare(A.transpose(),B,Q,R)

        numpy.testing.assert_allclose(X,P, rtol=1e-01)

        return None

    def test_kalman_1(self):
        """ Verify solution to steady state Kalman gains """

        A = numpy.asarray([[1,    2.6312,   -0.1104],
                           [0,    0.9205,   -0.0795],
                           [0,         0,    1.0000]], dtype=numpy.float64)
        B = numpy.asarray([0.1104, 0.0795, 0], dtype=numpy.float64)
        Q = numpy.asarray([[1, 0, 0], [0, 1e-3, 0], [0, 0, 1e-5]], dtype=numpy.float64)
        R = 1000

        L = numpy.asarray([0.0439, 2.0647e-4, -9.3383e-5], dtype=numpy.float64)

        X = dare.kalman(A.transpose(),B,Q,R)

        numpy.testing.assert_allclose(X,L, rtol=1e-01)

        return None

    def test_lqr_1(self):
        """ Verify solution to LQR gains """

        A = numpy.asarray([[   1.000000000000000,   0.002500000000000,   0.003334332394748],
                   [0,   1.000000000000000,   2.631164971838571],
                   [0,                   0,   0.920545702498017]], dtype=numpy.float64)
        B = numpy.asarray([0.000092646225341, 0.110417924232576, 0.079454297501983], dtype=numpy.float64)
        Q = numpy.asarray([[1, 0, 0], [0, 1e-3, 0], [0, 0, 1e-5]], dtype=numpy.float64)
        R = 1000

        L = numpy.asarray([[0.031295094898266, 0.008505395454746, 0.250329168660926]], dtype=numpy.float64)

        X = dare.lqr(A.transpose(),B,Q,R)

        numpy.testing.assert_allclose(X,L, rtol=1e-01)

        return None
if __name__ == '__main__':
    unittest.main()
