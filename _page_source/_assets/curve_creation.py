import sys
import os
sys.path.append(os.environ['HOME'] + '/code')
from pym import func as pymc
import numpy as np

x_data = np.linspace(0., 4. * np.pi, 1000)
y1_data = np.sin(x_data)
y2_data = np.cos(x_data)

y1 = pymc.curve(x_data, y1_data, name='$\sin \left( x \right)$')
y2 = pymc.curve(x_data, y2_data, name='$\cos \left( x \right)$')

plot = y1.plot(linecolor='#285668', linestyle='-')
y2.plot(linecolor='#FC8D82', linestyle='-', addto=plot)

plot.fill_between(x_data, y1_data, y2_data, fc='#ccccff')

plot.lines_on()
plot.markers_off()
plot.ylim(-1.1, 1.1)
plot.xlim(0., 4. * np.pi)

plot.xlabel('$x$')
plot.ylabel('$y$')

plot.export('../../images/curve_plotting', formats=['websvg'], sizes=['2'])
