
import unittest
import numpy
import math

import rtlqr


class StaticTestFunctions(unittest.TestCase):

    def setUp(self):
        rtlqr.init()

    def test_check_solution(self):
        """ simulate a static set of inputs and measurements
        """

        rtlqr.configure(tau=-3.5)
        rtlqr.configure(gains=numpy.array([7.0,7.0,3.0,7.0], dtype=numpy.float64))
        gains = rtlqr.solve()

        print `gains`

        return None

if __name__ == '__main__':
    unittest.main()

    rtlqr.configure(tau=-3.5)
    rtlqr.configure(gains=numpy.array([7.0,7.0,3.0,7.0], dtype=numpy.float64))
    gains = rtlqr.solve()
    print `gains`