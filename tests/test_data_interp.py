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
        A = pym.curve(x=[0., 1., 2.], y=[0., 1., 2.], u_x=[0.1, 0.1, 0.1],
                      u_y=[0.1, 0.2, 0.3])
        m = 1.0
        xi = 0.5
        dxi = 0.5
        ux1 = 0.1
        ux2 = 0.1
        uy1 = 0.1
        uy2 = 0.2
        dx = 0.1
        uy = m * np.sqrt(dx**2. + uy1**2. + uy2**2. +
                         dxi**2. * (ux1**2. + ux2**2.))
        self.assertEqual(A.u_y_at(xi, dx), uy,
                         'incorrect abscissa error')

    def test_find(self):
        # any value should be able to find its abscissa point
        A = self.A.copy()
        self.assertEqual(A.find(1.5), 1.5,
                         'incorrect find point')

    def test_extrapolation_right(self):
        # values to the right of the curve should be extrapolated sort of well
        A = self.A.copy()
        self.assertEqual(A.extrapolate(5.0), 5.0,
                         'incorrect extrapolated value to the right')

    def test_extrapolation_left(self):
        # values to the right of the curve should be extrapolated sort of well
        A = self.A.copy()
        self.assertEqual(A.extrapolate(-1.0), -1.0,
                         'incorrect extrapolated value to the left')

    def test_find_nearest_up_left(self):
        # values should be able to find nearest up and down
        A = self.A.copy()
        x, y = A.find_nearest_up(0.5)
        self.assertEqual(x, 1.0, 'incorrect nearest up left')

    def test_find_nearest_down_left(self):
        # values should be able to find nearest up and down
        A = self.A.copy()
        x, y = A.find_nearest_down(0.5)
        self.assertEqual(x, 0.0, 'incorrect nearest down left')

    def test_find_nearest_up_central(self):
        # values should be able to find nearest up and down
        A = self.A.copy()
        x, y = A.find_nearest_up(1.5)
        self.assertEqual(x, 2.0, 'incorrect nearest up central')

    def test_find_nearest_down_central(self):
        # values should be able to find nearest up and down
        A = self.A.copy()
        x, y = A.find_nearest_down(1.5)
        self.assertEqual(x, 1.0, 'incorrect nearest down central')

    def test_find_nearest_up_right(self):
        # values should be able to find nearest up and down
        A = self.A.copy()
        x, y = A.find_nearest_up(3.5)
        self.assertEqual(x, 4.0, 'incorrect nearest up right')

    def test_find_nearest_down_right(self):
        # values should be able to find nearest up and down
        A = self.A.copy()
        x, y = A.find_nearest_down(3.5)
        self.assertEqual(x, 3.0, 'incorrect nearest down right')

    def test_average_full_range(self):
        # values should be easily averageable ccross the whole range
        A = self.A.copy()
        self.assertEqual(A.average(), 2.0, 'incorrect full range average')

    def test_average_small_range(self):
        # values should be averageable across small ranges
        A = self.A.copy()
        self.assertEqual(A.average(1., 4.), 2.5,
                         'incorrect small range average')

    def test_round_to_amt(self):
        # a static method should round to a certain amount
        A = self.A.copy()
        x = A.round_to_amt(1.20, 0.25)
        self.assertEqual(x, 1.00,
                         'incorrect round to amount')

    def test_floating_avg(self):
        # values should rolling average well
        A = self.A.copy()
        B = A.rolling_avg(2.0)
        self.assertEqual(np.array_equal(B.y, [0.5, 2.5, 4.0]), True,
                         'incorrect rolling average')

    def test_floating_avg_uncertainty(self):
        # values when floating averaged should give proper uncertainty
        A = self.A.copy()
        B = A.rolling_avg(2.0)
        stdev_arr = [np.std([0., 1.]), np.std([2., 3.]), np.std([4.])]
        self.assertEqual(np.array_equal(B.u_y, stdev_arr), True,
                         'incorrect rolling average uncertainty')

    def tearDown(self):
        del self.A
        del self.B
        del self.C
        del self.D

if __name__ == '__main__':
    interp = unittest.TestLoader().loadTestsFromTestCase(dataInterpTestCase)
    unittest.TextTestRunner(verbosity=2).run(interp)
