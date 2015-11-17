import sys
import os
sys.path.append(os.environ['HOME'] + '/code')
from pym import func as pymc
import numpy as np

# use numpy to create a monotonic function to play with
x_data = np.linspace(0., 2., 1000)
y_data = np.power(x_data, 2)

# define these data as ahm.curves to expose the interface to the numerical
# methods
y = pymc.curve(x_data, y_data, name='$x^{2}$')

# Plot the unmodified function, shade the integral, and add a pointer with the
# integral value
plot = y.plot(linecolor='#285668', linestyle='-')
plot.fill_between(x_data, np.zeros_like(y_data), y_data, fc='#ccccff')
plot.add_data_pointer(1.5, point=1.5,
                      string=r'$\int f \left( x \right) dx = %.2f$' %
                      (y.integrate(0, 2)), place=(0.5, 3.))
plot.lines_on()
plot.markers_off()

# now normalize the curve with respect to the integral
y.normalize('int')
# Plot the modified function, shade the integral, and add a pointer with the
# integral value
plot = y.plot(addto=plot, linecolor='#FC8D82', linestyle='-')
plot.fill_between(x_data, np.zeros_like(y.x), y.y, fc='#ffdddd')
plot.add_data_pointer(1.25, point=0.125,
                      string=r'$\int f_{norm} \left( x \right) dx = %.2f$' %
                      (y.integrate(0, 2)), place=(0.25, 1.5))
plot.lines_on()
plot.markers_off()
plot.xlabel('$x$')
plot.ylabel('$y$')

# export it to a websvg (which doesnt convert text to paths)
plot.export('../../images/int_norm', formats=['websvg'], sizes=['2'])
