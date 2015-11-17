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
`matplotlib`,then download `pym` to our code directory (or wherever, really).
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

### Integration and normalization

### Curve arithmetic
