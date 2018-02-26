"""A symbolically based two-d data toolset."""
import numpy as np


def symfunc(object):
    """A symbolically defined function."""
    def __init__(self, func, dfunc=None, bounds=[-np.inf, np.inf], name=None):
        r"""Define the symbolically defined function.

        :param function func: the function that takes a single input and gives
            a single output.  Should be continuous.
        :param function dfunc: the derivative of the function that takes a
            single input and gives a single output.  If unknown, we will use
            methods that don't require the derivative.
        :param list bounds: the bounds of input that the function is defined
            over, default :math:`-\infty` to :math:`\infty`
        :param str name: the name of the function, preferably in the format
            :math:`f\left(x\right)`
        """
        self.func = func
        self.name = name
        self.dfunc = dfunc

    def find(self, y, p0=0.0):
        r"""Find the ordinate :math:`x` where :math:`func = y`.

        :param float y: the value :math:`y` to search for
        :param float p0: the guess for :math:`x_{0}`
        :returns float x0: the ordinate where :math:`f\left( x_{0} \right) = y`
        """
        if self.dfunc is None:
            return self.secant(y, p0)
        else:
            return self.newton(y, p0)

    def secant(self, y, p0=0.0):
        r"""Find where :math:`x_{0} = y` using secant method.

        :param float y: the value :math:`y` to search for
        :param float p0: the guess for :math:`x_{0}`
        :returns float x0: the ordinate where :math:`f\left( x_{0} \right) = y`
        """
        pass

    def newton(self, y, p0=0.0):
        r"""Find where :math:`x_{0} = y` using Newton's method.

        :param float y: the value :math:`y` to search for
        :param float p0: the guess for :math:`x_{0}`
        :returns float x0: the ordinate where :math:`f\left( x_{0} \right) = y`
        """
        pass
