import unittest
from pym import func as pym
import numpy as np

class dataInputTestCase(unittest.TestCase):
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

    def test_data_sort(self):
        # sorting should work correctly
        A = pym.curve([0., 2., 4., 1., 3.], [0., 2., 4., 1., 3.])
        self.assertEqual(np.array_equal(A.y, [0., 1., 2., 3., 4.]), True,
                         'incorrect sorted typical')

    def test_data_sort_repeated_x(self):
        # sorting with a repeated x value should
        A = pym.curve([0., 1., 0.], [1., 2., 0.])
        self.assertEqual(np.array_equal(A.y, [1., 0., 2.]), True,
                         'incorrect repeated x sort')

    def test_data_unbalanced_uncertainty_tuple(self):
        A = pym.curve([0., 2., 1.], [0., 1., 2.],
                      u_x=[(0.1, 0.2), (0.2, 0.4), (0.1, 0.2)],
                      u_y=[(0.2, 0.1), (0.4, 0.2), (0.2, 0.1)])
        self.assertEqual(A.u_y[1, 1], 0.1,
                         'incorrect unbalanced tuple uncertainty')

    def tests_data_unbalanced_uncertainty(self):
        A = pym.curve([0., 2., 1.], [0., 1., 2.],
                      u_x=[[0.1, 0.4, 0.1], [0.2, 0.4, 0.2]],
                      u_y=[[0.1, 0.4, 0.1], [0.2, 0.4, 0.2]])
        self.assertEqual(A.u_y[1, 1], 0.2, 'incorrect unbalanced uncertainty')

    def test_data_added(self):
        A = pym.curve([0., 2., 4.], [0., 2., 4.])
        A.add_data([1., 3.], [1., 3.])
        self.assertEqual(np.array_equal(A.y, [0., 1., 2., 3., 4.]), True,
                         'incorrect sorted add data')

    def test_data_added_u_y(self):
        A = pym.curve([0., 2., 4.], [0., 2., 4.], u_y=[0., 0.2, 0.4])
        A.add_data([1., 3.], [1., 3.], u_y=[0.1, 0.3])
        self.assertEqual(np.array_equal(A.u_y, [0., 0.1, 0.2, 0.3, 0.4]), True,
                         'incorrect sorted add data u_y')

    def test_copy(self):
        A = pym.curve([0., 2., 4.], [0., 2., 4.], u_y=[0., 0.2, 0.4])
        B = A.copy()
        self.assertEqual(np.array_equal(B.y, [0., 2., 4.]), True,
                         'incorrect curve copy')

    def test_copy_modify_in_place(self):
        A = pym.curve([0., 2., 4.], [0., 2., 4.], u_y=[0., 0.2, 0.4])
        B = A.copy()
        A.y = [1., 2., 3.]
        self.assertEqual(np.array_equal(B.y, [0., 2., 4.]), True,
                         'incorrect curve copy with inplace editing')

    def test_crop_x_replace(self):
        A = self.A.copy()
        A.crop(x_min=1., x_max=3., replace='remove')
        self.assertEqual(np.array_equal(A.y, [1., 2., 3.]), True,
                         'incorrect curve x cropping')

    def test_crop_y_replace(self):
        A = self.A.copy()
        A.crop(y_min=1., y_max=3., replace='remove')
        self.assertEqual(np.array_equal(A.x, [1., 2., 3.]), True,
                         'incorrect curve y cropping')

    def test_decimate(self):
        A = self.A.copy()
        A.decimate(2)
        self.assertEqual(np.array_equal(A.x, [0., 2., 4.]), True,
                         'incorrect decimation')

    def test_decimate_length(self):
        A = self.A.copy()
        A.decimate(length=2)
        self.assertEqual(np.array_equal(A.x, [0., 3.]), True,
                         'incorrect decimation by length')

    def tearDown(self):
        del self.A
        del self.B
        del self.C
        del self.D

if __name__ == '__main__':
    input = unittest.TestLoader().loadTestsFromTestCase(dataInputTestCase)
    unittest.TextTestRunner(verbosity=2).run(input)
