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

    def test_curve_addition_float(self):
        # the curves should add correctly, and in place
        A = self.A.copy()
        A.add(5.)
        self.assertEqual(A.y[0], 5., 'incorrect curve float addition')

    def test_curve_addition_curve(self):
        # two curves should add correctly and in place
        A = self.A.copy()
        A = A.add(A.copy())
        self.assertEqual(A.y[1], 2., 'incorrect curve curve addition')

    def test_curve_operator_addition_float(self):
        # curve should add with float correctly out of place with the operator
        A = self.A.copy()
        B = A + 5.
        self.assertEqual(A.y[1], 1., 'curve operator addition is in place')
        self.assertEqual(B.y[0], 5., 'curve operator addition is incorrect')

    def test_curve_operator_addition_curve(self):
        # curves should add correctly out of place with the operator
        A = self.A.copy()
        B = A + A
        self.assertEqual(A.y[1], 1., 'curve operator addition is in place')
        self.assertEqual(B.y[1], 2., 'curve operator addition is incorrect')

    def test_curve_operator_subtraction_curve(self):
        # the curves should subtract correctly
        A = self.A.copy()
        B = self.B.copy()
        A = A - B
        self.assertEqual(A.y[1], -2.,
                         'incorrect subtraction')

    def test_curve_operator_rsubtraction_float(self):
        # the curves should subtract correctly
        A = self.A.copy()
        A = 5. - A
        self.assertEqual(A.y[1], 4.,
                         'incorrect subtraction')

    def test_curve_operator_subtraction_float(self):
        # the curves should subtract correctly
        A = self.A.copy()
        A = A - 5.
        self.assertEqual(A.y[1], -4.,
                         'incorrect subtraction')

    def test_curve_multiply_float(self):
        A = self.A.copy()
        A.multiply(5.)
        self.assertEqual(A.y[1], 5., 'incorrect float multiplication')

    def test_curve_multiply_curve_same_x(self):
        A = self.A.copy()
        B = self.A.copy()
        A.multiply(B)
        self.assertEqual(A.y[2], 4., 'incorrect curve multiplication')

    def test_curve_multiply_curve_offset_x(self):
        A = pym.curve([0., 2., 4.], [0., 2., 4.])
        B = pym.curve([1., 3., 5.], [1., 3., 5.])
        A.multiply(B)
        self.assertEqual(A.y[2], 9., 'incorrect curve multiplication')

    def test_curve_operator_multiply_float(self):
        A = self.A.copy()
        B = A * 5.
        self.assertEqual(A.y[1], 1., 'multiply occured in place')
        self.assertEqual(B.y[1], 5., 'incorrect float multiplication')

    def test_curve_operator_multiply_curve_same_x(self):
        A = self.A.copy()
        B = A * A
        self.assertEqual(A.y[1], 1., 'multiply occured in place')
        self.assertEqual(B.y[2], 4., 'incorrect curve multiplication')

    def test_curve_operator_multiply_curve_offset_x(self):
        A = pym.curve([0., 2., 4.], [0., 2., 4.])
        B = pym.curve([1., 3., 5.], [1., 3., 5.])
        C = A * B
        self.assertEqual(A.y[1], 2., 'multiply occured in place')
        self.assertEqual(C.y[2], 9., 'incorrect curve multiplication')

    def test_curve_divide_float(self):
        A = self.A.copy()
        A.divide(5.)
        self.assertEqual(A.y[1], 0.2, 'incorrect float division')

    def test_curve_divide_curve_same_x(self):
        A = self.A.copy()
        B = self.A.copy()
        A.divide(B)
        self.assertEqual(A.y[2], 1., 'incorrect curve division')

    def test_curve_divide_curve_offset_x(self):
        A = pym.curve([0., 2., 4.], [0., 2., 4.])
        B = pym.curve([1., 3., 5.], [1., 3., 5.])
        A.divide(B)
        self.assertEqual(A.y[2], 1., 'incorrect curve division')

    def test_curve_operator_divide_float(self):
        A = self.A.copy()
        B = A / 5.
        self.assertEqual(A.y[1], 1., 'division occured in place')
        self.assertEqual(B.y[1], 0.2, 'incorrect float division')

    def test_curve_operator_divide_curve_same_x(self):
        A = self.A.copy()
        B = A / A
        self.assertEqual(A.y[1], 1., 'division occured in place')
        self.assertEqual(B.y[2], 1., 'incorrect curve division')

    def test_curve_operator_divide_curve_offset_x(self):
        A = pym.curve([0., 2., 4.], [0., 2., 4.])
        B = pym.curve([1., 3., 5.], [1., 3., 5.])
        C = A / B
        self.assertEqual(A.y[1], 2., 'division occured in place')
        self.assertEqual(C.y[2], 1., 'incorrect curve division')

    def test_curve_or(self):
        A = pym.curve([0., 2., 4.], [0., 2., 4.])
        B = pym.curve([1., 3., 5.], [1., 3., 5.])
        C = A | B
        self.assertEqual(A.y[1], 2., 'addition occurred in place')
        self.assertEqual(C.y[1], 1., 'data addition didnt occur')

    def tearDown(self):
        del self.A
        del self.B
        del self.C
        del self.D

if __name__ == '__main__':
    interpolation = unittest.TestLoader().loadTestsFromTestCase(CurveTestCase)
    unittest.TextTestRunner(verbosity=2).run(interpolation)
