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


def suite():
    suite = uinttest.TestSuite()
    suite.addTest(CurveTestCase('test_interpolation_first_point'))
    suite.addTest(CurveTestCase('test_interpolation_last_point'))
    return suite

if __name__ == '__main__':
    interpolation = unittest.TestLoader().loadTestsFromTestCase(CurveTestCase)
    unittest.TextTestRunner(verbosity=2).run(interpolation)
