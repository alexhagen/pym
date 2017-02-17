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

    def tearDown(self):
        del self.A
        del self.B
        del self.C
        del self.D

if __name__ == '__main__':
    integ = unittest.TestLoader().loadTestsFromTestCase(dataIntegTestCase)
    unittest.TextTestRunner(verbosity=2).run(integ)
