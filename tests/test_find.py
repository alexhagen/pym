import sys
import numpy as np
sys.path.append("/Users/ahagen/code")
from pym import func as ahf

x = np.arange(0., 24.)
y = np.append(np.arange(0., 12.), np.arange(11., -1., -1.))

curve = ahf.curve(x, y)
print curve.find(8.5)
print curve.at(8.5)
print curve.at(14.5)
