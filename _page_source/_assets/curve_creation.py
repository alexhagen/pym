import sys
import os
sys.path.append(os.environ['HOME'] + '/code')
from pym import func as pymc
import numpy as np

# use numpy to create some trigonometric functions across two periods
x_data = np.linspace(0., 4. * np.pi, 1000)
y1_data = np.sin(x_data)
y2_data = np.cos(x_data)

# define these data as ahm.curves to expose the interface to the numerical
# methods
y1 = pymc.curve(x_data, y1_data, name='$\sin \left( x \right)$')
y2 = pymc.curve(x_data, y2_data, name='$\cos \left( x \right)$')

# Plot these using the function pym.plot which returns a pyg.ah2d object
plot = y1.plot(linecolor='#285668', linestyle='-')
y2.plot(linecolor='#FC8D82', linestyle='-', addto=plot)

# make it pretty with some shading, lines, changing limits, and labels
plot.fill_between(x_data, y1_data, y2_data, fc='#ccccff')
plot.lines_on()
plot.markers_off()
plot.ylim(-1.1, 1.1)
plot.xlim(0., 4. * np.pi)
plot.xlabel('$x$')
plot.ylabel('$y$')

# export it to a websvg (which doesnt convert text to paths)
plot.export('../../images/curve_plotting', formats=['websvg'], sizes=['2'])
