---
layout: post
title: pym
description: "a math library in python"
category: code
tags: [python, numerical methods]
image:
  feature: "http://alexhagen.github.io/pyg/images/pyg_banner_2.png"
---

# pym - A math and numerical methods library in python

Last Updated on 11/12/15

Written by [Alex Hagen](http://alexhagen.github.io)

Hosted at [github.com/alexhagen/pym](http://github.com/alexhagen/pym)

Documentation at [alexhagen.github.io/pym/docs](docs/)

`pym` (pronounced <i>pim</i>) is a pretty simple numerical methods library for
python.  It can be used to do interpolation, extrapolation, integration,
normalization, etc.  In general, it is a replication of Brian Bradie's book
*A Friendly Introduction to Numerical Methods* in code.  Usage should be fairly
simple, with the documentation [documentation](docs/) providing a practical
guide to using the library.

## Pym Demonstrations and Screenshots

### Installation

To install `pym`, all we have to do is install `numpy`, `scipy`, and
`matplotlib`, then download `pym` to our code directory (or wherever, really).
To do this, we can use

```bash

$ pip install numpy
$ pip install scipy
$ pip install matplotlib
$ pip install colours
$ cd ~/code
$ git clone https://github.com/alexhagen/pym.git

```

and then, we can use the library within any script by using

```python
import sys
import os
sys.path.append(os.environ['HOME'] + '/code')
from pym import func as pymc

# use pym from here on
```

### Curve creation and graphing using [pyg](../pyg/)

The basis of `pym` is the `curve` class, which holds x and y data, as well as
its associated error.  We can create a function with a sinusoid in it by using
the following code

```python
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
```

![curve_plotting](http://alexhagen.github.io/pym/images/curve_plottingweb.svg)

### Integration and normalization

### Curve arithmetic
