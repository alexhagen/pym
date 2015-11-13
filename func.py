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
    def __init__(self, x, y, name='', u_x=None, u_y=None, data='smooth'):
        self.name = name
        self.data = data
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
        """ ``sort()`` sorts the list depending on the :math:`x` coordinate."""
        idx = self.x.argsort()
        self.x = self.x[idx]
        self.y = self.y[idx]
        if self.u_x is not None:
            self.u_x = self.u_x[idx]
        if self.u_y is not None:
            self.u_y = self.u_y[idx]

    def add_data(self, x, y):
        """ ``add_data(x,y)`` adds data to the already populated x and y.

        :param list-like x: The ordinate data to add to the already populated
            curve object.
        :param list-like y: The abscissa data to add to the already populated
            curve object.
        :return: A curve object with the added data, fully sorted.
        :rtype: curve
        """
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.sort()

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
        y = np.ones_like(x)
        for index, xi in np.ndenumerate(x):
            if xi >= self.x.min() and xi <= self.x.max():
                # if it is in the data range, interpolate
                y[index] = self.interpolate(xi)
            else:
                # if it is not in the data range, extrapolate
                y[index] = self.extrapolate(xi)
        return y

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

    def normalize(self, xmin=None, xmax=None, norm='max'):
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
            if xmin is None:
                xmin = self.x.min()
            if xmax is None:
                xmax = self.x.max()
            self.y = self.y / \
                self.integrate(xmin, xmax)

    def average(self, xmin=None, xmax=None):
        if xmin is None:
            xmin = self.x.min()
        if xmax is None:
            xmax = self.x.max()
        mean = self.integrate(xmin, xmax) \
            / (xmax - xmin)
        return mean

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
        if x < self.x.min():
            x1 = self.x[0]
            x2 = self.x[1]
        elif x > self.x.max():
            x1 = self.x[-1]
            x2 = self.x[-2]
        # now find the slope
        m = (self.at(x1) - self.at(x2)) / (x1 - x2)
        # find the y change between closest point and new point
        dy = m * (x - x1)
        # find the new point
        return self.at(x1) + dy

    def find_nearest_down(self, x):
        idx = (np.abs(x - self.x)).argmin()
        return (self.x[idx - 1], self.y[idx - 1])

    def find_nearest_up(self, x):
        idx = (np.abs(x - self.x)).argmin()
        return (self.x[idx], self.y[idx])

    def integrate(self,x_min,x_max,quad='lin'):
        # for now, we'll just do simpsons rule until I write
        # more sophisticated
        return self.trapezoidal(x_min,x_max,quad);
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
    def plot(self,x=None,y=None,addto=None,linestyle=None,linecolor='black',
        yy=False,xerr=None,yerr=None):
        if addto is None:
            plot = ahp.ah2d();
        else:
            plot = addto;
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
            plot.add_line(plot_x,plot_y,name=self.name,linewidth=4.0,linecolor=linecolor,
                linestyle='-');
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
                plot.add_line(x,y,xerr=self.u_x,yerr=self.u_y,name=self.name,linestyle=linestyle,linecolor=linecolor);
            else:
                plot.add_line_yy(x,y,xerr=self.u_x,yerr=self.u_y,name=self.name,linestyle=linestyle,linecolor=linecolor);
        return plot;
    def decimate(self,R):
        pad_size = math.ceil(float(self.x.size)/R)*R - self.x.size;
        arr_x_padded = np.append(self.x, np.zeros(pad_size)*np.NaN);
        self.x = nanmean(arr_x_padded.reshape(-1,R), axis=1);
        arr_y_padded = np.append(self.y, np.zeros(pad_size)*np.NaN);
        self.y = nanmean(arr_y_padded.reshape(-1,R), axis=1);
    def fit_exp(self):
        def exp_func(coeffs=None,x=None):
            return np.exp(np.polyval(coeffs,x));
        coeffs = np.polyfit(self.x,np.log(self.y),1);
        self.fun = exp_func;
        self.coeffs = coeffs;
        self.fit_exp_bool = True;
    def fit_gen(self,fun,guess=None,u_y=None):
        self.fun = fun;
        fit = curve_fit(fun, self.x, self.y, p0 = guess,sigma=u_y,absolute_sigma=True);
        self.coeffs = fit[0];
        self.fit_exp_bool = False;
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
        self.coeffs = coeffs;
    def plot_fit(self,xmin=None,xmax=None,addto=None,linestyle=None):
        if addto is None:
            plot = ahp.ah2d();
        else:
            plot = addto;
        if xmin is None:
            xmin = self.x.min();
        if xmax is None:
            xmax = self.x.max();
        x = np.linspace(xmin,xmax,num=1000);
        y = self.fit_at(x);
        plot.add_line(x,y,name=self.name+'fit',linestyle=linestyle);
        return plot;
