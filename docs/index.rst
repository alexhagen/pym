.. pym documentation master file, created by
   sphinx-quickstart on Wed Nov 11 13:36:16 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pym - a Python Math library
===============================

Pym is a library for some numerical methods written in python for general use.
It strives to be lightweight and easy to use, but if you haven't read the
introduction and motivation `here <http://alexhagen.github.io/pym/>`_, you
might want to.  In general, you should use this library if you want to use
easy integration, interpolation, extrapolation, and some graphing methods.
Many of the numerical method algorithms come from Brian Bradie's great book,
A Friendly Introduction to Numerical Analysis, and the theory and support for
those methods can be found in that book. This documentation will give cursory
explanations of the numerical methods used, but is written to be more
of a practical guide for coders. Good luck and happy coding!


func.curve
==========

.. autoclass:: func.curve
  :members:
  :show-inheritance:

.. include:: doc_coverage.rst

Indices and tables
==================

Test Coverage
-------------

.. raw:: html

  <iframe width=100% height=225px
   frameBorder="0" src="coverage/index.html"></iframe>

Contents:

.. toctree::
   :maxdepth: 2

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
