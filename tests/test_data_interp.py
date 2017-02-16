import unittest
from pym import func as pym
import numpy as np

class dataInterpTestCase(unittest.TestCase):
    def setUp(self):
        # set up a couple curves:
        self.A = pym.curve([0., 1., 2., 3., 4.],
                           [0., 1., 2., 3., 4.],
                           name="y=x @ dx = 1")
        self.B = pym.curve([0., 1., 2., 3., 4.],
                           [4., 3., 2., 1., 0.],
                           name="y=4-x @ dx = 1")
        self.C = pym.curve([0., 2., 4.],
                           [0., 2., 4.],
                           name="y=x @ dx = 2")
        self.D = pym.curve([0., 2., 4.],
                           [4., 2., 0.],
                           name="y=-x @ dx = 2")

    def test_data_inrange(self):
        A = self.A.copy()
        self.assertEqual(A.inrange(0.5), True,
                         'incorrect in range')

    def test_data_out_of_range(self):
        A = self.A.copy()
        self.assertEqual(A.inrange(-5.), False,
                         'incorrect out of range')

    def test_interpolation_first_point(self):
        # the value between the first point and second point should be
        # interpolated correctly
        self.assertEqual(self.A.at(0.5), 0.5,
                         'incorrect first point interpolation')

    def test_interpolation_last_point(self):
        # the value between the last point and penultimate point should be
        # interpolated correctly
        self.assertEqual(self.A.at(3.5), 3.5,
                         'incorrect last point interpolation')

    def test_interpolation_central(self):
        # any value in the interior should be interpolated correctly
        self.assertEqual(self.A.at(1.75), 1.75,
                         'incorrect central point interpolation')

    def test_uncertainty_interpolation(self):
        # any value should have correctly calculated uncertainty even when
        # interpolated
        A = pym.curve(x=[1., 3., 4.], y=[0., 1., 2.], u_x=[0.1, 0.1, 0.1],
                      u_y=[0.1, 0.2, 0.3])
        self.assertEqual(A.u_y_at(2.0, 2.0), 5.0,
                         'incorrect abscissa error')

    def test_find(self):
        # any value should be able to find its abscissa point
        A = self.A.copy()
        self.assertEqual(A.find(1.5), 1.5,
                         'incorrect find point')

    def tearDown(self):
        del self.A
        del self.B
        del self.C
        del self.D

if __name__ == '__main__':
    interp = unittest.TestLoader().loadTestsFromTestCase(dataInterpTestCase)
    unittest.TextTestRunner(verbosity=2).run(interp)
