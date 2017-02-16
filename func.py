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
                self.u_x = self.u_x[:, idx]
            else:
                self.u_x = self.u_x[idx]
        if self.u_y is not None:
            if len(self.u_y.shape) > 1:
                self.u_y = self.u_y[:, idx]
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

        if x_max is not None:
            for i in range(len(self.x)):
                if self.x[i] > x_max:
                    if replace is None:
                        self.x[i] = x_max
                    elif replace is "remove":
                        remove[i] = True

        if replace is "remove":
            self.x = np.delete(self.x, np.where(remove))
            if self.u_x is not None:
                self.u_x = np.delete(self.u_x, np.where(remove))
            self.y = np.delete(self.y, np.where(remove))
            if self.u_y is not None:
                self.u_y = np.delete(self.u_y, np.where(remove))
        return self

    def trim(self, trimcurv):
        print "trimming the curve"
        minx = np.min(self.x)
        maxx = np.max(self.x)
        if np.min(trimcurv.x) > minx:
            minx = np.min(trimcurv.x)
        if np.max(trimcurv.x) < maxx:
            maxx = np.max(trimcurv.x)
        xs = []
        ys = []
        for newx in np.linspace(minx, maxx, 50):
            xs.extend([newx])
            ys.extend([self.at(newx)])
        self.x = np.array(xs)
        self.y = np.array(ys)
        self.u_x = None
        self.u_y = None
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

            \sigma_{y}=m\sqrt{\cancel{\sigma_{x}^{2}}
            +\sigma_{y_{\downarrow}}^{2}+
            \sigma_{y_{\uparrow}}^{2}+\Delta\xi^{2}\left(
            \sigma_{x_{\downarrow}}^{2}+\sigma_{x_{\uparrow}}^{2}\right)}

        :param float x: The coordinate of which the value is desired.
        :param float dx: The uncertainty in the x coordinate requested, given in
            the above equations with :math:`sigma_{x}`.
        :returns: :math:`sigma_{y}`, the uncertainty of the value of the curve
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
                    u_y[index] = self.interpolate(xi)
                    # find the uncertainty interpolated
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

            x=\left(y-y_{0}\right)\frac{\left(x_{1}-x_{0}\right)}
            {\left(y_{1}-y_{0}\right)}+x_{0}

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
        :py:func:`find_nearest_up` to find the seconds nearest point.  Finally,
        it solves the following equation to determine the value

        .. math::

            y=\frac{\left(y_{1}-y_{0}\right)}{\left(x_{1}-x_{0}\right)}
            \left(x-x_{0}\right)+y_{0}

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

    def find_nearest_down(self, x):
        r""" ``find_nearest_down(x)`` will find the actual data point that is
        closest in negative ``x``-distance to the data point ``x`` passed to the
        function

        :param float x: The data point ``x`` which to find the closest value
            below.
        :return: a tuple containing the ``x`` and ``y`` value of the data point
            immediately below in ``x`` value to the value passed to the function
        :rtype: tuple
        """
        idx = (np.abs(x - self.x)).argmin()
        return (self.x[idx - 1], self.y[idx - 1])

    def find_nearest_up(self, x):
        r""" ``find_nearest_up(x)`` will find the actual data point that is
        closest in positive ``x``-distance to the data point ``x`` passed to the
        function

        :param float x: The data point ``x`` which to find the closest value
            above.
        :return: a tuple containing the ``x`` and ``y`` value of the data point
            immediately above in ``x`` value to the value passed to the function
        :rtype: tuple
        """
        idx = (np.abs(x - self.x)).argmin()
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

    def round_to_amt(self, num, amt):
        recip = 1.0 / amt
        return round(num * amt) / amt

    def rolling_avg(self, bin_width=0.1):
        r""" ``rolling_avg(bin_width)`` redistributes the data on a certain bin
        width, propogating the error needed.


        :param float bin_width: The width in which the redistribution will
            happen.
        :rtype: None
        """
        new_x = []
        new_y = []
        new_u_x = []
        new_u_y = []
        # find the start bin (round the minimum value to the next lowest bin)
        bin_start = self.round_to_amt(np.min(self.x), bin_width)
        # then, for everything in a certain bin:
        for left in np.arange(bin_start, np.max(self.x), bin_width):
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
                    u_y_sample = np.sqrt(np.std(u_sample)**2 + std**2)
                else:
                    u_y_sample = std
                if self.u_x is not None:
                    u_x_sample = np.sqrt((bin_width/2.)**2 + \
                        (np.mean(u_left))**2)
                else:
                    u_x_sample = bin_width/2.
                # add to new distribution
                new_x.extend([left + bin_width / 2.])
                new_y.extend([np.mean(sample)])
                new_u_y.extend([u_y_sample])
                new_u_x.extend([u_x_sample])
        self.x = np.array(new_x)
        self.y = np.array(new_y)
        self.u_y = np.array(new_u_y)
        self.u_x = np.array(new_u_x)
        self.sort()
        return self

    ###########################################################################
    # Data Integration and Normalization - tests in tests/test_data_integ.py
    ###########################################################################
    def integrate(self,x_min=None, x_max=None, quad='lin'):
        # for now, we'll just do simpsons rule until I write
        # more sophisticated
        if x_min is None:
            x_min = np.min(self.x)
        if x_max is None:
            x_max = np.max(self.x)
        if self.data != 'binned':
            return self.trapezoidal(x_min,x_max,quad)
        else:
            return self.bin_int(x_min, x_max)

    def bin_int(self,x_min=None, x_max=None):
        if x_min is None:
            x_min = np.min(self.x)
        if x_max is None:
            x_max = np.max(self.x)
        return np.sum([bin_height * bin_width
                       for bin_height, bin_width
                       in zip(self.y, np.array([0.] + self.x)
                       - np.array(self.x + self.x[-1]))])

    def derivative(self, _x, epsilon=None):
        if epsilon is None:
            epsilon = (self.x.max() - self.x.min())/1.E-5
        return (self.at(_x + epsilon) - self.at(_x - epsilon)) / (2. * epsilon)

    def trapezoidal(self,x_min,x_max,quad='lin'):
        # first we assert that all values are in the region
        # then, we find a bunch of x's between these values
        numpoints = 61;
        if quad is 'lin':
            x_sub = np.linspace(x_min,x_max,numpoints);
        elif quad is 'log':
            x_sub = np.logspace(np.log10(x_min),np.log10(x_max),num=numpoints);
        # then, between each x, we find the value there
        y_sub = [ self.at(x_i) for x_i in x_sub ];
        # then, we do the trapezoidal rule
        return np.sum([ ((x_sub[i+1]-x_sub[i])*y_sub[i]) + \
            ((x_sub[i+1]-x_sub[i])*(y_sub[i+1]-y_sub[i]))/2 \
            for i in np.arange(0,len(x_sub)-1) ]);

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
            self.y = self.y / \
                self.integrate()

    ###########################################################################
    # Curve Arithmetic - tests in tests/test_curve_arithmetic.py
    ###########################################################################
    def curve_mult(self, mult):
        # first, trim the curves to have only common data
        self.trim(mult)
        curve2 = mult
        xs = []
        ys = []
        u_ys = []
        u_xs = []
        for newx in np.linspace(self.x[0], self.x[-1], 50):
            r = self.at(newx) * curve2.at(newx)
            xs.extend([newx])
            ys.extend([r])
        self.x = np.array(xs)
        self.y = np.array(ys)
        self.u_y = None
        self.u_x = None
        self.sort()
        return self

    def multiply(self, mult):
        r""" ``multiply(mult)`` multiplies the curve by a value.

        The ``multiply`` function will multiply the curve by the value passed
        to it in ``mult``.  This value can be an array with the same size or a
        scalar of type integer or float.  Note that this will only change the
        value (``y``) of the function, not the abscissa (``x``).

        :param number mult: the number to multiply the curve by
        :returns: none
        """
        if isinstance(mult, int) or isinstance(mult, float):
            for i in range(len(self.y)):
                self.y[i] = mult * self.y[i]
                if self.u_y is not None:
                    self.u_y[i] = mult * self.u_y[i]
        if isinstance(mult, curve):
            self.curve_mult(mult)

    def __rmul__(self, mult):
        self.multiply(mult)
        return self

    def __mul__(self, mult):
        self.multiply(mult)
        return self

    def curve_div(self, right):
        # copy each so we're not making changes in place
        _left = self.copy()
        _right = right.copy()
        # we want to find an abscissa that has the most points with at least
        # one true data point and that doesn't require any extrapolation

        # first, trim the curves to have only common data
        self.trim(num)
        self.sort()
        curve2 = num
        xs = []
        ys = []
        u_ys = []
        u_xs = []
        for newx in np.linspace(self.x[0], self.x[-1], 50):
            r = self.at(newx) / curve2.at(newx)
            xs.extend([newx])
            ys.extend([r])
        self.x = xs
        self.y = ys
        self.u_y = None
        self.u_x = None
        self.sort()
        return self

    def divide(self, numerator):
        r""" ``divides(mult)`` divides a value by the curve.

        The ``divide`` function will divide the value provided in ``numerator``
        by the values in the curve.  This value can be an array with the same
        size or a scalar of type integer or float.  Note that this will only
        change the value (``y``) of the function, not the abscissa (``x``).

        :param number numerator: the number to be divided by the curve
        :returns: none
        """
        oldy = self.y.copy()
        if isinstance(numerator, int) or isinstance(numerator, float):
            numerator = float(numerator)
            for i in range(len(self.y)):
                self.y[i] = numerator / self.y[i]
                if self.u_y is not None:
                    self.u_y[i] = self.y[i] * self.u_y[i] / oldy[i]
        if isinstance(numerator, curve):
            self.curve_div(numerator)

    def __or__(self, other):
        """ a convienience class to add data to the already populated x and y.

        :param list-like x: The ordinate data to add to the already populated
            curve object.
        :param list-like y: The abscissa data to add to the already populated
            curve object.
        :return: A curve object with the added data, fully sorted.
        :rtype: curve
        """
        self.add_data(other.x, other.y)
        return self

    def __rdiv__(self, num):
        if not isinstance(num, curve):
            self.divide(num)
        else:
            self.curve_div(num)
        return self

    def __div__(self, denom):
        self.multiply(1.0 / denom)
        return self

    def __sub__(self, right):
        left = self.copy()
        _right = right.copy()
        _right.y = -_right.y
        left = left.add(_right)
        return left

    def add(self, right, name=None):
        r""" ``add(value)`` adds a value to the curve.

        The ``add`` function will add the provided value to the curve and
        return a copy of the curve with the added value

        :param number right: the number or curve to be added to the curve
        :returns: copy of self + right
        """
        left = self.copy()
        _right = right.copy()
        if isinstance(_right, curve):
            # first trim the curves to the same range (smallest)
            # and resample these to the most points we can get
            left.y += _right.y
        elif instance(_right, float):
            left.y += _right
        elif isinstance(_right, int):
            left.y += float(_right)
        if name is not None:
            left.name = name
        return left

    ###########################################################################
    # Curve Fitting - tests in tests/test_curve_fitting.py
    ###########################################################################
    def fit_exp(self):
        def exp_func(coeffs=None, x=None):
            return np.exp(np.polyval(coeffs, x))
        polyx = np.array([x1 for x1 in self.x], dtype=float)
        logy = np.array([np.log(y1) for y1 in self.y], dtype=float)
        coeffs = np.polyfit(polyx, logy, 1.0)
        self.fun = exp_func
        self.coeffs = coeffs
        self.fit_exp_bool = True

    def fit_lin(self):
        def lin_func(coeffs=None, x=None):
            return np.polyval(coeffs, x)
        coeffs = np.polyfit(self.x, self.y, 1)
        self.fun = lin_func;
        self.coeffs = coeffs;
        self.fit_exp_bool = True

    def fit_gen(self, fun, guess=None, u_y=None):
        self.fun = fun
        fit = curve_fit(fun, self.x, self.y, p0=guess,
                        sigma=u_y, absolute_sigma=True)
        self.coeffs = fit[0]
        self.fit_exp_bool = False

    def fit_gauss(self, guess=None):
        def gauss_fun(x, a, mu, sig):
            return a * np.exp(-np.power(x - mu, 2.) / (2. * np.power(sig, 2.)))
        self.fit_gen(gauss_fun, guess=guess)
        return self

    def fit_at(self,x):
        if self.fit_exp_bool:
            return self.fun(self.coeffs,x);
        else:
            return self.fun(x,*self.coeffs);

    def fit_square(self):
        def square_func(coeffs,x):
            return np.polyval(coeffs,x);
        coeffs = np.polyfit(self.x,self.y,2);
        self.fun = square_func;
        self.coeffs = coeffs
        self.fit_exp_bool = True

    def fit_cube(self):
        def cube_func(coeffs,x):
            return np.polyval(coeffs,x);
        coeffs = np.polyfit(self.x,self.y,3);
        self.fun = cube_func;
        self.coeffs = coeffs
        self.fit_exp_bool = True

    ###########################################################################
    # Curve Plotting - no tests currently
    ###########################################################################
    def plot(self, x=None, y=None, addto=None, linestyle=None,
             linecolor='black', yy=False, xerr=None, yerr=None, legend=True,
             env='plot', axes=None, polar=False):
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

    def plot_fit(self, xmin=None, xmax=None, addto=None, linestyle=None,
                 linecolor=None, name=None, axes=None):
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
