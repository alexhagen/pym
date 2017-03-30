import unittest
from pym import func as pym
import numpy as np


class dataIntegTestCase(unittest.TestCase):
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

    def test_integrate_full_range(self):
        A = self.A.copy()
        self.assertEqual(A.integrate(), 16. / 2.,
                         'incorrect full range integral')

    def test_integrate_small_range(self):
        A = self.A.copy()
        self.assertEqual(A.integrate(x_min=1.0, x_max=3.0), (9. - 1.) / 2.,
                         'incorrect small range integral')

    def test_integral_binned(self):
        A = pym.curve([0., 1., 2.], [5., 4., 5.], data='binned')
        print A.integrate()
        self.assertEqual(A.integrate(), 14., 'incorrect binned integral')

    def test_integral_sparse(self):
        A = pym.curve([0., 5., 10., 15., 20.], [0., 5., 10., 15., 20.],
                      data='binned')
        A.rebin([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        print A.x, A.y
        self.assertEqual(A.y[3], 0., 'incorrect rebin of sparse data')

    def test_integrate_bin_nans(self):
        A = pym.curve([0., 5., 10., 15., 20.], [np.nan, 5., 10., 15., 20.],
                      data='binned')
        print A.integrate(2., 7.)
        self.assertEqual(A.integrate(2., 7.), 10.,
                         'incorrect integration of data with nan')

    def test_rebin_nans(self):
        A = pym.curve([0., 5., 10., 15., 20.], [np.nan, 5., 10., 15., 20.],
                      data='binned')
        A.rebin([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        print A.x, A.y
        self.assertEqual(A.y[3], 0., 'incorrect rebin of sparse data with nan')

    def test_integral_binned_between(self):
        A = pym.curve([0., 1., 2.], [5., 4., 5.], data='binned')
        print A.integrate(x_min=0.5, x_max=1.5)
        self.assertEqual(A.integrate(x_min=0.5, x_max=1.5), 4.5,
                         'incorrect binned integral between')

    def test_integral_binned_top_range(self):
        A = pym.curve([0., 1., 2.], [5., 4., 5.], data='binned')
        print A.integrate(x_min=1.5, x_max=2.5)
        self.assertEqual(A.integrate(x_min=1.5, x_max=2.5), 4.5,
                         'incorrect binned integral between')

    def test_derivative_at_point(self):
        A = pym.curve([0., 1., 2.], [5., 4., 5.])
        self.assertEqual(A.derivative(1.), 0., 'incorrect derivative at point')

    def test_derivative_between_points(self):
        A = pym.curve([0., 1., 2.], [5., 4., 5.])
        epsilon = (A.derivative(0.5) - (-1.) < 1.0E-5)
        # note that A.derivative(0.5) is off by ~ 2.0E-14....could use some
        # better derivation, I guess.  Lagrange?
        self.assertEqual(epsilon, True,
                         'incorrect derivative in between points')

    def test_normalize_int(self):
        A = self.A.copy()
        A.normalize()
        self.assertEqual(A.y[1], 1. / 8., 'incorrect normalized by integral')

    def test_normalize_max(self):
        A = self.A.copy()
        A.normalize(norm='max')
        self.assertEqual(A.y[1], 1. / 4., 'incorrect normalized by max')

    def tearDown(self):
        del self.A
        del self.B
        del self.C
        del self.D

if __name__ == '__main__':
    integ = unittest.TestLoader().loadTestsFromTestCase(dataIntegTestCase)
    unittest.TextTestRunner(verbosity=2).run(integ)
