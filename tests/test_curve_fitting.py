import unittest
from pym import func as pym
import numpy as np


class dataFitTestCase(unittest.TestCase):
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

    def test_fit_exp(self):
        A = pym.curve(np.arange(5.), np.exp(2. * np.arange(5.)))
        A.fit_exp()
        fit = np.all((A.coeffs - [2., 0.]) < 1.0E-6)
        self.assertEqual(fit, True, 'incorrect log fitting')

    def test_fit_lin(self):
        A = pym.curve(np.arange(5.), 2. * np.arange(5.)) + 4.0
        A.fit_lin()
        fit = np.all((A.coeffs - [2., 4.]) < 1.0E-6)
        self.assertEqual(fit, True, 'incorrect lin fitting')

    def test_fit_gen(self):
        A = pym.curve(np.arange(5.), np.exp(2. * np.arange(5.)) + 4.0)
        def func(x, alpha, beta):
            return np.exp(alpha * x) + beta
        A.fit_gen(func)
        fit = np.all((A.coeffs - [2., 4.]) < 1.0E-6)
        self.assertEqual(fit, True, 'incorrect lin fitting')

    def test_fit_gauss(self):
        x = np.linspace(0., 100.)
        y = 100.0 * np.exp(-np.power(x - 50., 2.) / (2. * np.power(15., 2.)))
        A = pym.curve(x, y)
        A.fit_gauss()
        fit = np.all((A.coeffs - [100., 50., 15.]) < 1.0E-6)
        self.assertEqual(fit, True, 'incorrect gauss fitting')

    def test_fit_square(self):
        x = np.linspace(0., 100.)
        y = 100.0 * np.power(x, 2.) + 5. * x + 6.0
        A = pym.curve(x, y)
        A.fit_square()
        fit = np.all((A.coeffs - [100., 5., 6.]) < 1.0E-6)
        self.assertEqual(fit, True, 'incorrect square fitting')

    def test_fit_cube(self):
        x = np.linspace(0., 100.)
        y = -5.0 * np.power(x, 3.) + 100.0 * np.power(x, 2.) + 5. * x + 6.0
        A = pym.curve(x, y)
        A.fit_cube()
        fit = np.all(np.abs(A.coeffs - [-5., 100., 5., 6.]) < 1.0E-6)
        self.assertEqual(fit, True, 'incorrect cube fitting')

    def test_fit_at(self):
        x = np.linspace(0., 100.)
        y = -5.0 * np.power(x, 3.) + 100.0 * np.power(x, 2.) + 5. * x + 6.0
        A = pym.curve(x, y)
        A.fit_cube()
        eta = A.fit_at(0.5) -\
            (-5. * 0.5**3. + 100.0 * 0.5**2. + 5. * 0.5 + 6)
        self.assertEqual(np.abs(eta) < 1.0E-6, True, 'incorrect fit at')

    def tearDown(self):
        del self.A
        del self.B
        del self.C
        del self.D

if __name__ == '__main__':
    fit = unittest.TestLoader().loadTestsFromTestCase(dataFitTestCase)
    unittest.TextTestRunner(verbosity=2).run(fit)
