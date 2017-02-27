import math
import numpy as np
import sys
import os
sys.path.append(os.environ['HOME'] + '/code/')
from pyg import twod as ahp
from scipy import nanmean
from scipy.optimize import curve_fit
from scipy.odr import *


class curve(object):
    r""" An object to expose some numerical methods and plotting tools.

    A ``curve`` object takes any two dimensional dataset and its uncertainty
    (both in the :math:`x` and :math:`y` direction).  Each data set includes
    :math:`x` and :math:`y` data and uncertainty associated with that, as well
    as a name and a data shape designation (whether this is smooth data or
    binned).

    There exist three ways to add uncertainty to the measurements.  The first is
    to define an array or list of values that define the absolute uncertainty at
    each ``x``.  The second is to define a list of tuples that define the lower
    and upper absolute uncertainty at each ``x``, respectively. The final way is
    to define a two dimensional array, where the first row is the lower absolute
    uncertainty at each ``x``, and the second row is the upper absolute
    uncertainty at each ``x``.

    :param list-like x: The ordinate data of the curve
    :param list-like u_x: The uncertainty in the ordinate data of the curve
    :param list-like y: The abscissa data of the curve
    :param list-like u_y: The uncertainty in the abscissa data of the curve
    :param str name: The name of the data set, used for plotting, etc.
    :param str data: The type of data, whether 'smooth' or 'binned'. This
        parameter affects the interpolation (and in turn, many other functions)
        by determining what the value is between data points.  For smooth data,
        linear interpolation is enacted to find values between points, for
        binned data, constant interpolation is used.
    :return: the ``curve`` object.
    :rtype: curve
    """

    ###########################################################################
    # Data Input - tests in tests/test_data_input.py
    ###########################################################################
    def __init__(self, x, y, name='', u_x=None, u_y=None, data='smooth'):
        self.name = name
        self.data = data
        self.epsilon = 0.05
        # assert that x and y are 1d lists of same size
        if isinstance(x, list):
            self.x = np.array(x)
        else:
            self.x = x
        if isinstance(y, list):
            self.y = np.array(y)
        else:
            self.y = y
        if isinstance(u_x, list):
            self.u_x = np.array(u_x)
        else:
            self.u_x = u_x
        if isinstance(u_y, list):
            self.u_y = np.array(u_y)
        else:
            self.u_y = u_y
        self.sort()

    def sort(self):
        r""" ``sort()`` sorts the list depending on the :math:`x` coordinate.

        ``sort()`` sorts all of the data input to the curve so that it is
        ordered from decreasing :math:`x` to increasing :math:`x`.

        :return: the ``curve`` object, but it has been sorted in-place.
        :rtype: curve
        """
        idx = self.x.argsort()
        self.x = self.x[idx]
        self.y = self.y[idx]
        if self.u_x is not None:
            if len(self.u_x.shape) > 1:
                if self.u_x.shape[1] == len(self.x):
                    self.u_x = self.u_x[:, idx]
                else:
                    self.u_x = self.u_x[idx, :]
            else:
                self.u_x = self.u_x[idx]
        if self.u_y is not None:
            if len(self.u_y.shape) > 1:
                if self.u_y.shape[1] == len(self.y):
                    self.u_y = self.u_y[:, idx]
                else:
                    self.u_y = self.u_y[idx, :]
            else:
                self.u_y = self.u_y[idx]

    def add_data(self, x, y, u_x=None, u_y=None):
        """ ``add_data(x,y)`` adds data to the already populated x and y.

        :param list-like x: The ordinate data to add to the already populated
            curve object.
        :param list-like y: The abscissa data to add to the already populated
            curve object.
        :param list-like u_x: The uncertainty in the ordinate data to be added.
        :param list-like u_y: The uncertainty in the abscissa data to be added.
        :return: A curve object with the added data, fully sorted.
        :rtype: curve
        """
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        if self.u_x is not None:
            self.u_x = np.append(self.u_x, u_x)
        if self.u_y is not None:
            self.u_y = np.append(self.u_y, u_y)
        self.sort()

    def copy(self):
        r""" ``copy()`` performs a deep copy of the curve and passes it out to
        another ``curve`` object so that it can be manipulated out-of-place.

        :return: a copy of the ``curve`` object calling the function
        :rtype: curve
        """
        newx = self.x.copy()
        newy = self.y.copy()
        newuy = None
        newux = None
        if self.u_y is not None:
            newuy = self.u_y.copy()
        if self.u_x is not None:
            newux = self.u_x.copy()
        newname = self.name
        return curve(newx, newy, u_y=newuy, u_x=newux, name=newname)

    def crop(self, y_min=None, y_max=None, x_min=None, x_max=None,
             replace=None):
        r""" ``crop(y_min, y_max, x_min, x_max, replace)`` will find any data
        points that fall outside of the rectangle with corners at
        ``(x_min, y_min)`` to ``(x_max, y_max)`` and replace it with the value
        specified as ``return``.


        :param float x_min: A value for which any values with :math:`x<x_{min}`
            will be replaced with the value ``replace``.
        :param float x_max: A value for which any values with :math:`x>x_{max}`
            will be replaced with the value ``replace``.
        :param float y_min: A value for which any values with :math:`y<y_{min}`
            will be replaced with the value ``replace``.
        :param float y_max: A value for which any values with :math:`y>y_{max}`
            will be replaced with the value ``replace``.
        :param float replace: The value to replace any value outside of the
            rectangle with.  Default ``None``.
        :return: the cropped ``curve`` object
        """
        remove = [False for i in range(len(self.x))]
        if y_min is not None:
            for i in range(len(self.x)):
                if self.y[i] < y_min:
                    if replace is None:
                        self.y[i] = y_min
                    elif replace is "remove":
                        remove[i] = True
                    elif isinstance(replace, float):
                        self.y[i] = replace
                if self.u_y is not None:
                    if self.y[i] - self.u_y[i] < y_min:
                        self.u_y[i] = self.y[i] - y_min

        if y_max is not None:
            for i in range(len(self.x)):
                if self.y[i] > y_max:
                    if replace is None:
                        self.y[i] = y_max
                    elif replace is "remove":
                        remove[i] = True
                    elif isinstance(replace, float):
                        self.y[i] = replace
                if self.u_y is not None:
                    if self.y[i] + self.u_y[i] > y_max:
                        self.u_y[i] = y_max - self.y[i]

        if x_min is not None:
            for i in range(len(self.x)):
                if self.x[i] < x_min:
                    if replace is None:
                        self.x[i] = x_min
                    elif replace is "remove":
                        remove[i] = True
                    elif isinstance(replace, float):
                        self.x[i] = replace

        if x_max is not None:
            for i in range(len(self.x)):
                if self.x[i] > x_max:
                    if replace is None:
                        self.x[i] = x_max
                    elif replace is "remove":
                        remove[i] = True
                    elif isinstance(replace, float):
                        self.x[i] = replace

        if replace == "remove":
            self.x = np.delete(self.x, np.where(remove))
            if self.u_x is not None:
                self.u_x = np.delete(self.u_x, np.where(remove))
            self.y = np.delete(self.y, np.where(remove))
            if self.u_y is not None:
                self.u_y = np.delete(self.u_y, np.where(remove))
        return self

    def find_first_above(self, y_min):
        r""" ``find_first_above(y)`` finds the first ``(x, y)`` tuple with y
            value above the given value y

        :param float y_min: the comparitor value
        :returns: the tuple (x, y) which is the first in ``x`` space where
            ``y`` is above the given y_min
        """
        i = 0
        while i < len(self.x):
            if self.y[i] > y_min:
                return self.x[i], self.y[i]
            i += 1
        return (np.nan, np.nan)

    def rebin(self, x=None):
        r""" ``rebin`` redistributes the curve along a new set of x values

        ``rebin(x)`` takes a list-like input of new points on the ordinate and
        redistributes the abscissa so that the x values are only on those
        points.  For continuous/smooth data, this simply interpolates the
        previous curve to the new points.  For binned data, this integrates
        between left bin points and redistributes the fraction of data between
        those points.

        :param list x: the new x values to redistribute the curve. If binned,
            this indicates the left edge
        :returns: the curve object with redistributed values
        """
        if self.data == 'smooth':
            newy = [self.at(_x) for _x in x]
        elif self.data == 'binned':
            bin_widths = [x2 - x1 for x1, x2 in zip(self.x[:-1], self.x[1:])]
            # assume the last bin has the same width
            bin_widths = bin_widths + [bin_widths[-1]]
            newy = [self.integrate(x_min=_x, x_max=_x + bw)
                    for _x, bw in zip(x, bin_widths)]
        self.x = np.array(x)
        self.y = np.array(newy)
        self.sort()
        return self

    def decimate(self, R=None, length=None):
        r""" ``decimate(R)`` will remove all but every ``R`` th point in the
        curve.


        :param int R: An integer value telling how often to save a point.
        :param int length: *Alternate*, an integer telling how big you
            want the final array.
        :return: the decimated ``curve`` object
        """
        if length is not None:
            R = (len(self.x) / length) + 1
        self.y = self.y[::R]
        self.x = self.x[::R]
        if self.u_x is not None:
            self.u_x = self.u_x[::R]
        if self.u_y is not None:
            self.u_y = self.u_y[::R]
        return self

    ###########################################################################
    # Data Retrieving and Interpolation - tests in tests/test_data_interp.py
    ###########################################################################
    def inrange(self, x):
        """ ``inrange(x)`` checks if a point is within the range of data.

        :param float x: The data point to check if it is in the range of the
            existing curve data.
        :return: Whether or not the data is in the range of the curve data.
        :rtype: bool
        """
        if x >= self.x.min() and x <= self.x.max():
            return True
        else:
            return False

    def at(self, x):
        """ ``at(x)`` finds a value at x.

        ``at(x)`` uses interpolation or extrapolation to determine the value
        of the curve at a given point, :math:`x`.  The function first checks
        if :math:`x` is in the range of the curve.  If it is in the range, the
        function calls :py:func:`interpolate` to determine the value.  If it is
        not in the range, the function calls :py:func:`extrapolate` to
        determine the value.

        :param float x: The coordinate of which the value is desired.
        :returns: the value of the curve at point :math:`x`
        :rtype: float
        """
        if isinstance(x, float):
            x = [x]
        y = np.ones_like(x)
        for index, xi in zip(range(len(x)), x):
            if xi in self.x:
                y[index] = self.y[list(self.x).index(xi)]
            else:
                if xi > self.x.min() and xi < self.x.max():
                    if self.data == 'binned':
                        _, y[index] = self.find_nearest_down(xi)
                    else:
                        # if it is in the data range, interpolate
                        y[index] = self.interpolate(xi)
                else:
                    # if it is not in the data range, extrapolate
                    y[index] = self.extrapolate(xi)
        if len(y) == 1:
            y = y[0]
        return y

    def u_y_at(self, x, dx=0.0):
        r""" ``u_y_at(x)`` finds a the uncertainty of a value at x.

        ``u_y_at(x)`` uses interpolation or extrapolation to determine the
        uncertainty of the value of the curve at a given point, :math:`x`.  The
        function first checks if :math:`x` is in the range of the curve.  If it
        is in the range, the function calls :py:func:`interpolate` and
        :py:func:`propogate_error` to find the uncertainty of the point.  If it
        is not in the range, the function calls :py:func:`extrapolate` and
        :py:func:`propogate_error` to determine the value.

        We use the following equation to perform the interpolation:

        .. math::

            y\left(x\right) = \left(x-x_{\downarrow}\right)
                \frac{\left(y_{\uparrow}-y_{\downarrow}\right)}
                     {\left(x_{\uparrow}-x_{\downarrow}\right)}

        And using the *error propagation formula* from (Knoll, 1999), which is

        .. math::

            \sigma_{\zeta}^{2} =
                \left(\frac{\partial\zeta}{\partial x}\right)^{2}\sigma_{x}^{2}
                +
                \left(\frac{\partial\zeta}{\partial y}\right)^{2}\sigma_{y}^{2}

        for a derived value :math:`\zeta`, we can apply this to interpolation
        and get:

        .. math::

            \sigma_{y}^{2} =
                \left(\frac{\partial y}{\partial x}\right)^{2}\sigma_{x}^{2}
                +
                \left(\frac{\partial y}{\partial x_{\downarrow}}\right)^{2}
                \sigma_{x_{\downarrow}}^{2}
                +
                \left(\frac{\partial y}{\partial x_{\uparrow}}\right)^{2}
                \sigma_{x_{\uparrow}}^{2}
                +
                \left(\frac{\partial y}{\partial y_{\downarrow}}\right)^{2}
                \sigma_{y_{\downarrow}}^{2}
                +
                \left(\frac{\partial y}{\partial y_{\uparrow}}\right)^{2}
                \sigma_{y_{\uparrow}}^{2}

        and, performing the derivatives, we can get:

        .. math::

            \sigma_{y}^{2}=\left(\frac{\left(y_{\uparrow}-y_{\downarrow}\right)}
            {\left(x_{\uparrow}-x_{\downarrow}\right)}\right)^{2}
            \sigma_{x}^{2}+\left(-\left(x-x_{\uparrow}\right)
            \frac{\left(y_{\uparrow}-y_{\downarrow}\right)}
            {\left(x_{\uparrow}-x_{\downarrow}\right)^{2}}\right)^{2}
            \sigma_{x_{\downarrow}}^{2}+\left(\left(x-x_{\downarrow}\right)
            \frac{\left(y_{\uparrow}-y_{\downarrow}\right)}{
            \left(x_{\uparrow}-x_{\downarrow}\right)^{2}}\right)^{2}
            \sigma_{x_{\uparrow}}^{2}\\+\left(-\frac{\left(x-x_{\downarrow}
            \right)}{\left(x_{\uparrow}-x_{\downarrow}\right)}\right)^{2}
            \sigma_{y_{\downarrow}}^{2}+\left(\frac{
            \left(x-x_{\downarrow}\right)}{\left(x_{\uparrow}-x_{\downarrow}
            \right)}\right)^{2}\sigma_{y_{\uparrow}}^{2}

        Finally, if we take :math:`m=\frac{\left(y_{\uparrow}-y_{\downarrow}
        \right)}{\left(x_{\uparrow}-x_{\downarrow}\right)}`, and
        :math:`\Delta\xi=\frac{\left(x-x_{\downarrow}\right)}{\left(x_{
        \uparrow}-x_{\downarrow}\right)}`, we can get:

        .. math::

            \sigma_{y}^{2}=m^{2}\left[\sigma_{x}^{2}+
            \sigma_{y_{\downarrow}}^{2}+\sigma_{y_{\uparrow}}^{2}+
            \Delta\xi^{2}\left(\sigma_{x_{\downarrow}}^{2}+
            \sigma_{x_{\uparrow}}^{2}\right)\right]

        and the square root of that is the uncertainty.

        .. math::

            \sigma_{y}=m\sqrt{\sigma_{x}^{2}+\sigma_{y_{\downarrow}}^{2}+
            \sigma_{y_{\uparrow}}^{2}+\Delta\xi^{2}\left(
            \sigma_{x_{\downarrow}}^{2}+\sigma_{x_{\uparrow}}^{2}\right)}

        Note that if an uncertainty in x is not supplied, that the first term
        will go to zero, giving

        .. math::

            \require{cancel}
            \sigma_{y}=m\sqrt{\cancel{\sigma_{x}^{2}}
            +\sigma_{y_{\downarrow}}^{2}+
            \sigma_{y_{\uparrow}}^{2}+\Delta\xi^{2}\left(
            \sigma_{x_{\downarrow}}^{2}+\sigma_{x_{\uparrow}}^{2}\right)}

        :param float x: The coordinate of which the value is desired.
        :param float dx: *Optional* The uncertainty in the x coordinate
            requested, given in the above equations as :math:`\sigma_{x}`.
        :returns: :math:`\sigma_{y}`, the uncertainty of the value of the curve
            at point :math:`x`
        :rtype: float
        """
        if isinstance(x, float):
            x = [x]
        u_y = np.ones_like(x)
        for index, xi in zip(range(len(x)), x):
            if xi in self.x:
                u_y[index] = self.u_y[list(self.x).index(xi)]
            else:
                if xi > self.x.min() and xi < self.x.max():
                    # if it is in the data range, interpolate
                    xi1, y1, uxi1, uy1 = self.find_nearest_down(xi, error=True)
                    xi2, y2, uxi2, uy2 = self.find_nearest_up(xi, error=True)
                    m = (y2 - y1) / (xi2 - xi1)
                    dxi = (xi - xi1) / (xi2 - xi1)
                    u_y[index] = m * np.sqrt(dx**2. + uy1**2. + uy2**2. +
                                             dxi**2. * (uxi1**2. + uxi2**2.))
                else:
                    # if it is not in the data range, extrapolate
                    u_y[index] = self.extrapolate(xi)
                    # find the uncertainty extrapolated
        if len(u_y) == 1:
            u_y = u_y[0]
        return u_y

    def find(self, y):
        r""" ``find(y)`` finds values of :math:`x` that have value :math:`y`

        This function takes a parameter :math:`y` and finds all of the ordinate
        coordinates that have that value.  Basically, this is a root-finding
        problem, but since we have a linear interpolation, the actual
        root-finding is trivial.  The function first finds all intervals in
        the dataset that include the value :math:`y`, and then solves the
        interpolation to find those :math:`x` values according to

        .. math::

            x=\left(y-y_{\downarrow}\right)\frac{\left(x_{\uparrow}
            -x_{\downarrow}\right)}{\left(y_{\uparrow}-y_{\downarrow}\right)}
            +x_{\downarrow}

        :param float y: the value which ordinate values are desired
        :return: a list of :math:`x` that have value :math:`y`
        :rtype: list
        """
        # take the entire list of y's and subtract the value.  those intervals
        # where the sign changes are where the function crosses the value
        y_p = y - self.y
        # find where the sign change is
        (interval, ) = np.where(np.multiply(y_p[:-1], y_p[1:]) < 0.)
        # using those intervals, create y_0s and y_1s
        y_left = self.y[interval]
        y_right = self.y[interval + 1]
        x_left = self.x[interval]
        x_right = self.x[interval + 1]
        # generate an array by solving the point slope form equation
        x_where = np.zeros_like(y_left)
        for i in range(len(y_left)):
            x_where[i] = ((x_right[i] - x_left[i]) / (y_right[i] - y_left[i]))\
                * (y - y_left[i]) + x_left[i]
        # return all of those intervals
        return x_where

    def interpolate(self, x):
        r""" ``interpolate(x)`` finds the value of a point in the curve range.

        The function uses linear interpolation to find the value of a point in
        the range of the curve data.  First, it uses
        :py:func:`find_nearest_down` and :py:func:`find_nearest_up` to find the
        two points comprising the interval which :math:`x` exists in.  Then, it
        casts the linear interpolation as a line in point slope form and solves

        .. math::

            y=\frac{\left(y_{1}-y_{0}\right)}{\left(x_{1}-x_{0}\right)}
            \left(x-x_{0}\right)+y_{0}

        :param float x: The coordinate of the desired value.
        :return: the value of the curve at :math:`x`
        :rtype: float
        """
        # if not, we have to do linear interpolation
        # find closest value below
        x_down, y_down = self.find_nearest_down(x)
        # find the closest value above
        x_up, y_up = self.find_nearest_up(x)
        # find the percentage of x distance between
        x_dist = (x - x_down)
        # find the slope
        m = (y_up - y_down) / (x_up - x_down)
        # find the y value
        y = y_down + x_dist * m
        return y

    def extrapolate(self, x):
        r""" ``extrapolate(x)`` finds value of a point out of the curve range.

        The function uses linear extrapolation to find the value of a point
        without the range of the already existing curve.  First, it determines
        whether the requested point is above or below the existing data. Then,
        it uses :py:func:`find_nearest_down` or :py:func:`find_nearest_up` to
        find the nearest point.  Then it uses :py:func:`find_nearest_down` or
        :py:func:`find_nearest_up` to find the second nearest point.  Finally,
        it solves the following equation to determine the value

        .. math::

            y=\frac{\left(y_{\downarrow}-y_{\downarrow \downarrow}
            \right)}{\left(x_{\downarrow}-x_{\downarrow \downarrow}\right)}
            \left(x-x_{\downarrow}\right)+y_{\downarrow}

        :param float x: the ordinate of the value requested
        :returns: the value of the curve at point :math:`x`
        :rtype: float
        """
        # find whether the point is above or below
        if x <= self.x.min():
            x1 = self.x[0]
            x2 = self.x[1]
        elif x >= self.x.max():
            x1 = self.x[-1]
            x2 = self.x[-2]
        # now find the slope
        m = (self.at(x1) - self.at(x2)) / (x1 - x2)
        # find the y change between closest point and new point
        dy = m * (x - x1)
        # find the new point
        return self.at(x1) + dy

    def find_nearest_down(self, x, error=False):
        r""" ``find_nearest_down(x)`` will find the actual data point that is
        closest in negative ``x``-distance to the data point ``x`` passed to
        the function.

        :param float x: The data point ``x`` which to find the closest value
            below.
        :param bool error: If true, the u_x and u_y will be returned at that
            point, even if they are ``None``.
        :return: a tuple containing the ``x`` and ``y`` value of the data point
            immediately below in ``x`` value to the value passed to the
            function, optionally containing the ``u_x`` and ``u_y`` value.
        """
        dx = x - self.x
        dx[dx < 0.] = np.inf
        idx = np.abs(dx).argmin()
        if error:
            ux = None
            uy = None
            if self.u_x is not None:
                ux = self.u_x[idx]
            if self.u_y is not None:
                uy = self.u_y[idx]
            return (self.x[idx], self.y[idx], ux, uy)
        else:
            return (self.x[idx], self.y[idx])

    def find_nearest_up(self, x, error=False):
        r""" ``find_nearest_up(x, error=False)`` will find the actual data
        point that is closest in positive ``x``-distance to the data point
        ``x`` passed to the function.

        :param float x: The data point ``x`` which to find the closest value
            above.
        :param bool error: If true, the u_x and u_y will be returned at that
            point, even if they are ``None``.
        :return: a tuple containing the ``x`` and ``y`` value of the data point
            immediately above in ``x`` value to the value passed to the
            function, optionally containing the ``u_x`` and ``u_y`` value.
        :rtype: tuple
        """
        dx = x - self.x
        dx[dx > 0.] = np.inf
        idx = np.abs(dx).argmin()
        if error:
            ux = None
            uy = None
            if self.u_x is not None:
                ux = self.u_x[idx]
            if self.u_y is not None:
                uy = self.u_y[idx]
            return (self.x[idx], self.y[idx], ux, uy)
        else:
            return (self.x[idx], self.y[idx])

    def average(self, xmin=None, xmax=None):
        r""" ``average()`` will find the average ``y``-value across the entire
        range.

        :param float xmin: The lower bound of ``x``-value to include in the
            average.  Default:  ``x.min()``
        :param float xmax: The upper bound of ``x``-value to include in the
            average.  Default: ``x.max()``
        :return: A float value equal to

        .. math::

            \bar{y} = \frac{\int_{x_{min}}^{x_{max}} y dx}
            {\int_{x_{min}}^{x_{max}} dx}


        :rtype: float
        """
        if xmin is None:
            xmin = self.x.min()
        if xmax is None:
            xmax = self.x.max()
        mean = self.integrate(xmin, xmax) \
            / (xmax - xmin)
        return mean

    @staticmethod
    def round_to_amt(num, amt):
        r""" ``round_to_amt`` is a static method that round a number to an
            arbitrary interval

        Given a number ``num`` such as :math:`1.2` and an amount ``amt`` such
        as :math:`0.25`, ``round_to_amt`` would return :math:`1.20` because
        that is the closest value downward on a :math:`0.25` wide grid.

        :param float num: the number to be rounded.
        :param float amt: the amount to round the number to.
        :returns: the number after it has been rounded.
        """
        return float(np.floor(num / amt)) * amt

    def rolling_avg(self, bin_width=0.1):
        r""" ``rolling_avg(bin_width)`` redistributes the data on a certain bin
        width, propogating the error needed.

        If we have data in an array such as

        .. math::

            \left[\begin{array}{c}
                \vec{x}\\
                \vec{y}
                \end{array}\right]=\left[\begin{array}{cccc}
                0.1 & 0.75 & 1.75 & 1.9\\
                1.0 & 2.0 & 3.0 & 4.0
                \end{array}\right]

        and we want to see the data only on integer bins, we will return

        .. math::

            \left[\begin{array}{c}
                \vec{x}\\
                \vec{y}
                \end{array}\right]=\left[\begin{array}{cc}
                0.0 & 2.0\\
                1.5 & 3.5
                \end{array}\right]

        This function will also return the uncertainty in each bin, taking into
        account both the uncertainty of each value in the bin, as well as the
        uncertainty caused by standard deviation within the bin itself.  This
        can be expressed by

        .. math::

            \left[\begin{array}{c}
                \vec{x}\\
                \vec{y}\\
                \vec{u}_{x}\\
                \vec{u}_{y}
                \end{array}\right]=\left[\begin{array}{c}
                \frac{\sum_{x\text{ in bin}}x}{N_{x}}\\
                \frac{\sum_{x\text{ in bin}}y}{N_{y}}\\
                \frac{\sum_{x\text{ in bin}}\sqrt{
                    \left(\frac{\text{bin width}}{2}\right)^{2}
                    +\text{mean}\left(\sigma_{x}\right)^{2}}}{N_{x}}\\
                \frac{\sum_{x\text{ in bin}}\sqrt{\sigma_{y}^{2}
                    +stdev_{y}^{2}}}{N_{x}}
                \end{array}\right]

        :param float bin_width: The width in which the redistribution will
            happen.
        :rtype: The redistributed curve.
        """
        new = self.copy()
        new_x = []
        new_y = []
        new_u_x = []
        new_u_y = []
        # find the start bin (round the minimum value to the next lowest bin)
        bin_start = self.round_to_amt(np.min(self.x), bin_width)
        # then, for everything in a certain bin:
        for left in np.arange(bin_start, np.max(self.x) + bin_width,
                              bin_width):
            # average to find the mean
            sample = [y for x, y in zip(self.x, self.y)
                      if x >= left and x < left + bin_width]
            if self.u_y is not None:
                u_sample = [u_y for x, u_y in zip(self.x, self.u_y)
                            if x >= left and x < left + bin_width]
            if self.u_x is not None:
                u_left = [u_x for x, u_x in zip(self.x, self.u_x)
                          if x >= left and x < left + bin_width]
            if len(sample) > 0:
                # determine the standard deviation
                std = np.std(sample)
                # propagate the uncertainty and add the standard deviation
                if self.u_y is not None:
                    u_y_sample = np.sqrt(np.mean(u_sample)**2 + std**2)
                else:
                    u_y_sample = std
                if self.u_x is not None:
                    u_x_sample = np.sqrt((bin_width / 2.)**2 +
                                         (np.mean(u_left))**2)
                else:
                    u_x_sample = bin_width / 2.
                # add to new distribution
                new_x.extend([left + bin_width / 2.])
                new_y.extend([np.mean(sample)])
                new_u_y.extend([u_y_sample])
                new_u_x.extend([u_x_sample])
        new.x = np.array(new_x)
        new.y = np.array(new_y)
        new.u_y = np.array(new_u_y)
        new.u_x = np.array(new_u_x)
        new.sort()
        return new

    ###########################################################################
    # Data Integration and Normalization - tests in tests/test_data_integ.py
    ###########################################################################
    def integrate(self, x_min=None, x_max=None, quad='lin'):
        r""" ``integrate`` integrates under the curve.

        ``integrate`` will integrate under the given curve, providing the
        result to :math:`\int_{x_{min}}^{x_{max}}`.  ``x_min`` and ``x_max``
        can be provided to change the range of integration.  ``quad`` can also
        be provided to change the quadrature, but the only quadrature currently
        supported is ``'lin'`` which uses trapezoidal rule to integrate the
        curve.

        :param float x_min: *Optional* the bottom of the range to be integrated.
        :param float x_max: *Optional* the top of the range to be integrated.
        :param str quad: *Optional* the "quadrature" to be used for numerical
            integration.
        :returns: the result of the integration.
        """
        if self.data != 'binned':
            if x_min is None:
                x_min = np.min(self.x)
            if x_max is None:
                x_max = np.max(self.x)
            return self.trapezoidal(x_min=x_min, x_max=x_max, quad=quad)
        else:
            return self.bin_int(x_min, x_max)

    def bin_int(self, x_min=None, x_max=None):
        r""" ``bin_int`` integrates a bar chart.

        ``bin_int`` is a convenience function used through the class when
        calling ``integrate``.  It integrates for curves that have the
        ``.data`` property set to ``'binned'``.  It does this simply by summing
        the bin width and bin heights, such that

        .. math::

            \int_{x_{min}}^{x_{max}} \approx \sum_{i=1,\dots}^{N} \Delta x
            \cdot y

        Note that this function assumes that the last bin has the same bin
        width as the penultimate bin width.  This could be remedied in certain
        ways, but I'm not sure which to choose yet.

        :param float x_min: *Optional* the bottom of the range to be integrated.
        :param float x_max: *Optional* the top of the range to be integrated.
        :returns: the result of the integration.
        """
        bin_widths = [x2 - x1 for x1, x2 in zip(self.x[:-1], self.x[1:])]
        # assume the last bin has the same width
        bin_widths = bin_widths + [bin_widths[-1]]
        print bin_widths
        bin_heights = self.y
        if x_min is None:
            x_min = np.min(self.x)
        if x_max is None:
            x_max = np.max(self.x) + bin_widths[-1]
        integral = 0.0
        # for each bin, find what fraction is within the range
        for _x, bw, bh in zip(self.x, bin_widths, bin_heights):
            fractional_bin_width = (np.min([_x + bw, x_max])
                                    - np.max([_x, x_min])) / bw
            if fractional_bin_width < 0:
                fractional_bin_width = 0.0
            integral += fractional_bin_width * bh
        return integral

    def derivative(self, x, epsilon=None):
        r""" ``derivative(x)`` takes the derivative at point :math:`x`.

        ``derivative(x)`` takes the derivative at point provided ``x``, using a
        surrounding increment of :math:`\varepsilon`, provided by ``epsilon``.
        ``epsilon`` has a default value of :math:`\min \frac{\Delta x}{100}`,
        but you can specify this smaller if your points are closer.  Because
        we're currently only using linear integration, this won't change a thing
        as long as its smaller than the change in your ordinate variable.

        :param float x: The ordinate to take the derivative at.
        :param float epsilon: The :math:`\Delta x` around the point at
            :math:`x` used to calculate the derivative.
        :returns: the derivative at point ``x``
        """
        if epsilon is None:
            xs = self.x[1:] - self.x[:-1]
            epsilon = np.min(np.abs(xs)) / 100.
        return (self.at(x + epsilon) - self.at(x - epsilon)) / (2. * epsilon)

    def trapezoidal(self, x_min, x_max, quad='lin'):
        r""" ``trapezoidal()`` uses the trapezoidal rule to integrate the curve.

        ``trapezoidal(x_min, x_max)`` integrates the curve using the
        trapezoidal rule, i.e.

        .. math::

            \int_{x_{min}}^{x_{max}}y dx \approx
            \sum_{i=1,\dots}^{N} \left(x_{\uparrow} - x_{\downarrow}\right)
            \cdot \left( \frac{y_{\downarrow} + y_{uparrow}}{2}\right)

        Right now, it uses :math:`10 \times N_{x}` points to integrate between
        values, but that is completely arbitrary and I'll be looking into
        changing this. There is also the ability to pass ``quad`` to the
        function as ``'log'`` **CURRENTLY FAILING** and it will calculate the
        trapezoids in logarithmic space, giving exact integrals for exponential
        functions.

        :param float x_min: the left bound of integration.
        :param float x_max: the right bound of integration.
        :param str quad: the type of quadrature to use, currently only ``'lin'``
            or ``'log'``
        :returns: the integral of the curve from trapezoidal rule.
        """
        numpoints = len(self.x) * 10
        if quad is 'lin':
            x_sub = np.linspace(x_min, x_max, numpoints)
            # then, between each x, we find the value there
            y_sub = [self.at(x_i) for x_i in x_sub]
            integral = np.sum([((x_sub[i+1] - x_sub[i]) * y_sub[i]) +
                               ((x_sub[i+1] - x_sub[i]) * (y_sub[i+1] - y_sub[i])) / 2.
                               for i in np.arange(0, len(x_sub) - 1)])
        # then, we do the trapezoidal rule
        return integral

    def normalize(self, xmin=None, xmax=None, norm='int'):
        r""" ``normalize()`` normalizes the entire curve to be normalized.

        **Caution! This will change all of the y values in the entire curve!**

        Normalize will take the data of the curve (optionally just the data
        between ``xmin`` and ``xmax``) and normalize it based on the option
        given by ``norm``.  The options for norm are ``max`` and ``int``. For a
        ``max`` normalization, first the function finds the maximum value of
        the curve in the range of the :math:`x` data and adjusts all :math:`y`
        values according to

        .. math::

            y = \frac{y}{y_{max}}

        For an ``int`` normalization, the function adjusts all :math:`y` values
        according to

        .. math::

            y=\frac{y}{\int_{x_{min}}^{x_{max}}y \left( x \right) dx}

        :param float xmin: optional argument giving the lower bound of the
            integral in an integral normalization or the lower bound in which
            to find the max in a max normalization
        :param float xmax: optional argument giving the upper bound of the
            integral in an integral normalization or the upper bound in which
            to find the max in a max normalization
        :param str norm: a string of 'max' or 'int' (default 'max') which
            defines which of the two types of normalization to perform
        :return: None
        """
        if norm is 'max':
            self.y = self.y / self.y.max()
        elif norm is 'int':
            self.y = self.y / self.integrate()
        return self

    ###########################################################################
    # Curve Arithmetic - tests in tests/test_curve_arithmetic.py
    ###########################################################################
    def add(self, right, name=None):
        r""" ``add(value)`` adds a value to the curve.

        The ``add`` function will add the provided value to the curve in place.

        :param number right: the number or curve to be added to the curve
        :returns: ``curve`` with added :math:`y` values
        """
        _right = right
        if isinstance(_right, curve):
            # first trim the curves to the same range (smallest)
            # and resample these to the most points we can get
            _right = right.copy()
            self.y += _right.y
        elif isinstance(_right, float):
            self.y += _right
        elif isinstance(_right, int):
            self.y += float(_right)
        if name is not None:
            self.name = name
        return self

    def __add__(self, right):
        _left = self.copy()
        if isinstance(right, curve):
            _right = right.copy()
        else:
            _right = right
        _left = _left.add(_right)
        return _left

    def __sub__(self, right):
        _left = self.copy()
        if isinstance(right, curve):
            _right = right.copy()
            _right.y = -_right.y
        else:
            _right = -right
        _left = _left.add(_right)
        return _left

    def __rsub__(self, left):
        _left = left
        _right = self.copy()
        _right.y = -_right.y
        _right = _right.add(_left)
        return _right

    def multiply(self, mult):
        r""" ``multiply(mult)`` multiplies the curve by a value.

        The ``multiply`` function will multiply the curve by the value passed
        to it in ``mult``.  This value can be an array with the same size or a
        scalar of type integer or float.  Note that this will only change the
        value (``y``) of the function, not the abscissa (``x``).

        :param number mult: the number to multiply the curve by
        :returns: the curve after multiplication
        """
        if isinstance(mult, int) or isinstance(mult, float):
            for i in range(len(self.y)):
                self.y[i] = mult * self.y[i]
                if self.u_y is not None:
                    self.u_y[i] = mult * self.u_y[i]
        if isinstance(mult, curve):
            self.curve_mult(mult)
        return self

    def curve_mult(self, mult):
        r""" ``curve_mult(curve)`` multiplies two curves together.

        This is a helper class, usually only called through ``curve.multiply``,
        or using the ``*`` operator. The class first takes a unique set of
        ``x`` points that are within the range of both curves. Then, it
        multiplies those two together.

        :param number mult: the curve to multiply by
        :returns: the left ``curve`` object, with the values multipled in
            place.
        """
        x1min = np.min(self.x)
        x2min = np.min(mult.x)
        x1max = np.max(self.x)
        x2max = np.max(mult.x)
        xmin = np.max([x1min, x2min])
        xmax = np.min([x1max, x2max])
        allxs = np.append(self.x, mult.x)
        allxs = allxs[allxs >= xmin]
        allxs = allxs[allxs <= xmax]
        xs = np.unique(allxs)
        ys = [self.at(x) for x in xs]
        zs = [mult.at(x) for x in xs]
        product = [y * z for y, z in zip(ys, zs)]
        self.x = np.array(xs)
        self.y = np.array(product)
        self.u_y = None
        self.u_x = None
        self.sort()
        return self

    def __rmul__(self, mult):
        _left = mult
        _right = self.copy()
        _right.multiply(_left)
        return _right

    def __mul__(self, mult):
        _left = self.copy()
        if isinstance(mult, curve):
            _right = mult.copy()
        else:
            _right = mult
        _left.multiply(_right)
        return _left

    def divide(self, denominator):
        r""" ``divide(denominator)`` divides a curve by a value.

        The ``divide`` function will divide the curve by the value provided in
        ``numerator``.  Note that this will only change the value (``y``) of
        the function, not the abscissa (``x``).

        :param number denominator: the number to divide the curve by.
        :returns: none
        """
        oldy = np.copy(self.y)
        if isinstance(denominator, int) or isinstance(denominator, float):
            denominator = float(denominator)
            for i in range(len(self.y)):
                self.y[i] = self.y[i] / denominator
                if self.u_y is not None:
                    self.u_y[i] = self.y[i] * self.u_y[i] / oldy[i]
        if isinstance(denominator, curve):
            self.curve_div(denominator)
        return self

    def divide_by(self, numerator):
        r""" ``divide_by(numerator)`` divides a value by the curve.

        The ``divide`` function will divide the value provided in ``numerator``
        by the values in the curve.  Note that this will only change the value
        (``y``) of the function, not the abscissa (``x``).

        :param number numerator: the number to be divided by the curve.
        :returns: none
        """
        oldy = np.copy(self.y)
        if isinstance(numerator, int) or isinstance(numerator, float):
            numerator = float(numerator)
            for i in range(len(self.y)):
                self.y[i] = self.y[i] / numerator
                if self.u_y is not None:
                    self.u_y[i] = self.y[i] * self.u_y[i] / oldy[i]
        if isinstance(numerator, curve):
            numerator.curve_div(self)
            self = numerator.copy()
        return self

    def curve_div(self, right):
        r""" ``curve_div(curve)`` divides one curve by another.

        This is a helper class, usually only called through ``curve.divide``,
        or using the ``/`` operator. The class first takes a unique set of
        ``x`` points that are within the range of both curves. Then, it
        divides the ``y`` values by the other.

        :param number right: the curve to divide by.
        :returns: the left ``curve`` object, with the values divided in
            place.
        """
        x1min = np.min(self.x)
        x2min = np.min(right.x)
        x1max = np.max(self.x)
        x2max = np.max(right.x)
        xmin = np.max([x1min, x2min])
        xmax = np.min([x1max, x2max])
        allxs = np.append(self.x, right.x)
        allxs = allxs[allxs >= xmin]
        allxs = allxs[allxs <= xmax]
        xs = np.unique(allxs)
        ys = [self.at(x) for x in xs]
        zs = [right.at(x) for x in xs]
        with np.errstate(divide='ignore', invalid='ignore'):
            quotient = np.divide(ys, zs)
        self.x = np.array(xs)
        self.y = np.array(quotient)
        self.u_y = None
        self.u_x = None
        self.sort()
        return self

    def __rdiv__(self, left):
        _right = self.copy()
        if isinstance(left, curve):
            _left = left.copy()
        else:
            _left = left
        _right.divide_by(_left)
        return _right

    def __div__(self, right):
        _left = self.copy()
        if isinstance(right, curve):
            _right = right.copy()
        else:
            _right = right
        _left.divide(_right)
        return _left

    def __or__(self, other):
        """ a convienience class to add data to the already populated x and y.

        :param list-like x: The ordinate data to add to the already populated
            curve object.
        :param list-like y: The abscissa data to add to the already populated
            curve object.
        :return: A curve object with the added data, fully sorted.
        :rtype: curve
        """
        left = self.copy()
        left.add_data(other.x, other.y)
        return left

    ###########################################################################
    # Curve Fitting - tests in tests/test_curve_fitting.py
    ###########################################################################
    def fit_exp(self):
        r""" ``fit_exp`` fits an exponential to the function.

        ``fit_exp`` fits an exponential of form :math:`y=B\cdot \exp \left(
        \alpha\cdot x\right)` to the curve, returning the parameters
        :math:`\left(\alpha, B\right)` as a tuple.

        :returns: the tuple :math:`\left(\alpha, B\right)`
        """
        def exp_func(coeffs=None, x=None):
            return np.exp(np.polyval(coeffs, x))
        polyx = np.array([x1 for x1 in self.x], dtype=float)
        logy = np.array([np.log(y1) for y1 in self.y], dtype=float)
        coeffs = np.polyfit(polyx, logy, 1.0)
        self.fun = exp_func
        self.coeffs = coeffs
        self.fit_exp_bool = True
        return self

    def fit_lin(self):
        r""" ``fit_lin`` fits a linear function to the curve.

        ``fit_lin`` fits a linear function of form :math:`y=m\cdot x + b` to the
        curve, returning the parameters :math:`\left(m, b\right)` as a tuple.

        :returns: the tuple :math:`\left(m, b\right)`
        """
        def lin_func(coeffs=None, x=None):
            return np.polyval(coeffs, x)
        coeffs = np.polyfit(self.x, self.y, 1)
        self.fun = lin_func;
        self.coeffs = coeffs;
        self.fit_exp_bool = True
        return self

    def fit_gen(self, fun, guess=None, u_y=None):
        r""" ``fit_gen`` fits a general function to the curve.

        ``fit_gen`` fits a general function to the curve.  The general function
        is a python function that takes a parameters and an ordinate variable,
        ``x`` and returns the value of the function at that point, ``y``.  The
        function must have the prototype ``def func(x, alpha, beta, ...):``.
        Then, the coefficients are returned as a tuple.

        :returns: the coefficients to the general function
        """
        self.fun = fun
        fit = curve_fit(fun, self.x, self.y, p0=guess,
                        sigma=u_y, absolute_sigma=True)
        self.coeffs = fit[0]
        self.fit_exp_bool = False
        return self

    def fit_gauss(self, guess=None):
        r""" ``fit_gauss`` fits a gaussian function to the curve.

        ``fit_gauss`` fits a gaussian function of form :math:`y=\alpha \exp
        \left[ -\frac{\left(x - \mu\right)^{2}}{2 \sigma^{2}}\right]` to the
        curve, returning the parameters :math:`\left(\alpha, \mu, \sigma\right)`
        as a tuple.

        :returns: the tuple :math:`\left(\alpha, \mu, \sigma\right)`
        """
        def gauss_fun(x, a, mu, sig):
            return a * np.exp(-np.power(x - mu, 2.) / (2. * np.power(sig, 2.)))
        self.fit_gen(gauss_fun, guess=guess)
        return self

    def fit_at(self,x):
        r""" ``fit_at`` returns the point at coordinate :math:`x` from a previously fitted curve.

        :param float x: the ordinate variable for which the fit value is needed.
        """
        if self.fit_exp_bool:
            return self.fun(self.coeffs,x)
        else:
            return self.fun(x,*self.coeffs)

    def fit_square(self):
        r""" ``fit_square`` fits a function of order 2 to the curve.

        ``fit_square`` fits a quadratic function of form :math:`y=a x^{2} + b x
        + c` to the curve, returning the parameters :math:`\left(a, b,
        c\right)` as a tuple.

        :returns: the tuple :math:`\left(a, b, c\right)`
        """
        def square_func(coeffs,x):
            return np.polyval(coeffs,x)
        coeffs = np.polyfit(self.x,self.y,2)
        self.fun = square_func
        self.coeffs = coeffs
        self.fit_exp_bool = True
        return self

    def fit_cube(self):
        r""" ``fit_cube`` fits a function of order 3 to the curve.

        ``fit_cube`` fits a cubic function of form :math:`y=a x^{3} + b x^{2} +
        c x + d` to the curve, returning the parameters :math:`\left(a, b,
        c, d\right)` as a tuple.

        :returns: the tuple :math:`\left(a, b, c, d\right)`
        """
        def cube_func(coeffs,x):
            return np.polyval(coeffs,x);
        coeffs = np.polyfit(self.x,self.y,3);
        self.fun = cube_func;
        self.coeffs = coeffs
        self.fit_exp_bool = True
        return self

    ###########################################################################
    # Curve Plotting - no tests currently
    ###########################################################################
    def plot(self, x=None, y=None, addto=None, # pragma: no cover
             linestyle=None, linecolor='black', # pragma: no cover
             yy=False, xerr=None, yerr=None, # pragma: no cover
             legend=True, env='plot', axes=None, # pragma: no cover
             polar=False): # pragma: no cover
        if addto is None:
            plot = ahp.pyg2d(env=env, polar=polar);
        else:
            plot = addto;
        if xerr is None:
            xerr = self.u_x
        if yerr is None:
            yerr = self.u_y
        if x is None and y is None:
            x = self.x;
            y = self.y;
        if self.data is 'binned':
            # plot the bins
            # setup a matix
            # preallocate this later ***********************************
            plot_x = np.array([]);
            plot_y = np.array([]);
            # plot the thick bars
            for i in np.arange(0,len(x)-1):
                plot_x = np.append(plot_x,x[i]);
                plot_y = np.append(plot_y,y[i]);
                plot_x = np.append(plot_x,x[i+1]);
                plot_y = np.append(plot_y,y[i]);
                plot_x = np.append(plot_x,np.nan);
                plot_y = np.append(plot_y,np.nan);
                self.binned_data_x = plot_x
                self.binned_data_y = plot_y
            plot.add_line(plot_x,plot_y,name=self.name,linewidth=4.0,linecolor=linecolor,
                linestyle='-', legend=legend);
            conn_x = np.array([]);
            conn_y = np.array([]);
            for i in np.arange(1,len(x)):
                conn_x = np.append(conn_x,x[i]);
                conn_y = np.append(conn_y,y[i-1]);
                conn_x = np.append(conn_x,x[i]);
                conn_y = np.append(conn_y,y[i]);
                conn_x = np.append(conn_x,np.nan);
                conn_y = np.append(conn_y,np.nan);
            plot.add_line(conn_x,conn_y,name=self.name+'connectors',linewidth=0.1,linestyle='-',linecolor=linecolor);
            plot.markers_off();
            plot.lines_on();
        elif self.data is 'smooth':
            if yy is False:
                plot.add_line(x,y,xerr=self.u_x,yerr=self.u_y,name=self.name,linestyle=linestyle,linecolor=linecolor, axes=axes);
            else:
                plot.add_line_yy(x,y,xerr=self.u_x,yerr=self.u_y,name=self.name,linestyle=linestyle,linecolor=linecolor, axes=axes);
        return plot;

    def plot_fit(self, xmin=None, xmax=None, addto=None, # pragma: no cover
                 linestyle=None,  linecolor=None, # pragma: no cover
                 name=None, axes=None): # pragma: no cover
        if addto is None:
            plot = ahp.pyg2d()
        else:
            plot = addto
        if xmin is None:
            xmin = self.x.min()
        if xmax is None:
            xmax = self.x.max()
        self.fitx = np.linspace(xmin, xmax, num=1000)
        self.fity = self.fit_at(self.fitx)
        if name is None:
            name = self.name + 'fit'
        plot.add_line(self.fitx, self.fity, name=name,
                      linestyle=linestyle, linecolor=linecolor, axes=axes)
        return plot
