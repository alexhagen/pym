import unittest
from pym import func as pym

class CurveTestCase(unittest.TestCase):
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

    def test_curve_subtraction(self):
        # the curves should subtract correctly
        newcurve = self.A - self.B
        self.assertEqual(newcurve.at(0.5), -3.,
                         'incorrect subtraction')

    def test_curve_division(self):
        # the division of curves should be correct
        newcurve = self.A / self.B
        self.assertEqual(newcurve.at(0.5), 1./7.,
                         'incorrect division with matching grid')

    def tearDown(self):
        del self.A
        del self.B
        del self.C
        del self.D

if __name__ == '__main__':
    interpolation = unittest.TestLoader().loadTestsFromTestCase(CurveTestCase)
    unittest.TextTestRunner(verbosity=2).run(interpolation)
