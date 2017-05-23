<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

pym - A math and numerical methods library in python[¶](#pym---A-math-and-numerical-methods-library-in-python){.anchor-link} {#pym---A-math-and-numerical-methods-library-in-python}
============================================================================================================================

Last Updated on 11/12/15

Written by [Alex Hagen](http://alexhagen.github.io)

Hosted at [github.com/alexhagen/pym](http://github.com/alexhagen/pym)

Documentation at [alexhagen.github.io/pym/docs](docs/)

`pym` (pronounced *pim*) is a pretty simple numerical methods library
for python. It can be used to do interpolation, extrapolation,
integration, normalization, etc. In general, it is a replication of
Brian Bradie's book *A Friendly Introduction to Numerical Methods* in
code. Usage should be fairly simple, with the documentation
[documentation](docs/) providing a practical guide to using the library.

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Pym Demonstrations and Screenshots[¶](#Pym-Demonstrations-and-Screenshots){.anchor-link} {#Pym-Demonstrations-and-Screenshots}
----------------------------------------------------------------------------------------

### Installation[¶](#Installation){.anchor-link} {#Installation}

To install `pym`, all we have to do is install `numpy`, `scipy`, and
`matplotlib`, then download `pym` to our code directory (or wherever,
really). To do this, we can use

<div class="highlight">

    $ pip install numpy
    $ pip install scipy
    $ pip install matplotlib
    $ pip install colours
    $ cd ~/code
    $ git clone https://github.com/alexhagen/pym.git

</div>

and then, we can use the library within any script by using

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[1\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    from pym import func as pym

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Curve creation and graphing[¶](#Curve-creation-and-graphing){.anchor-link} {#Curve-creation-and-graphing}

The basis of `pym` is the `curve` class, which holds x and y data, as
well as its associated error. We can create a function with a sinusoid
in it by using the following code

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[2\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    from pym import func as pym
    import numpy as np

    # use numpy to create some trigonometric functions across two periods
    x_data = np.linspace(0., 4. * np.pi, 1000)
    sin_data = np.sin(x_data)
    cos_data = np.cos(x_data)

    # define these data as ahm.curves to expose the interface to the numerical
    # methods
    sin = pym.curve(x_data, sin_data, name='$\sin \left( x \right)$')
    cos = pym.curve(x_data, cos_data, name='$\cos \left( x \right)$')

    # Plot these using the function pym.plot which returns a pyg.ah2d object
    plot = sin.plot(linecolor='#285668', linestyle='-')
    plot = cos.plot(linecolor='#FC8D82', linestyle='-', addto=plot)

    # make it pretty with some shading, lines, changing limits, and labels
    plot.fill_between(sin.x, np.zeros_like(sin.y), sin.y, fc='#ccccff')
    plot.fill_between(cos.x, np.zeros_like(cos.y), cos.y, fc='#ffcccc')
    plot.lines_on()
    plot.markers_off()
    plot.ylim(-1.1, 1.1)
    plot.xlim(0., 4. * np.pi)
    plot.xlabel(r'x-coordinate ($x$) [$cm$]')
    plot.ylabel(r'y-coordinate ($y$) [$cm$]')

    # export it to a websvg (which doesnt convert text to paths)
    plot.export('_static/curve_plotting', ratio='silver')
    plot.show('A pretty chart from data made for a pym curve')

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_html rendered_html output_subarea">

<div class="pygfigure"
name="A pretty chart from data made for a pym curve"
style="text-align: center; max-width: 800px; margin-left: auto; margin-right: auto;">

![](_static/curve_plotting.svg?594072040)
<div class="figurecaption" style="margin: auto; text-align: center;">

**Figure 1:** A pretty chart from data made for a pym curve

</div>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Integration and normalization[¶](#Integration-and-normalization){.anchor-link} {#Integration-and-normalization}

One of the useful options of `pym` is the ability to normalize a
function, either according to its maximum, or according to its integral.
The following is and example of this integration, showing that after
integration, we attain an integral of 1.0.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[3\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    # use numpy to create a monotonic function to play with
    x_data = np.linspace(0., 2., 1000)
    y_data = np.power(x_data, 2)

    # define these data as ahm.curves to expose the interface to the numerical
    # methods
    y = pym.curve(x_data, y_data, name='$x^{2}$')

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
    plot.xlabel(r'x-coordinate ($x$) [$cm$]')
    plot.ylabel(r'y-coordinate ($y$) [$cm$]')
    plot.ylim(0.0, 4.0)
    plot.xlim(0.0, 2.0)

    # export it to a websvg (which doesnt convert text to paths)
    plot.export('_static/int_norm', ratio='silver')
    plot.show('Normalized curves have a total integral of 1.0')

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_html rendered_html output_subarea">

<div class="pygfigure"
name="Normalized curves have a total integral of 1.0"
style="text-align: center; max-width: 800px; margin-left: auto; margin-right: auto;">

![](_static/int_norm.svg?83514618)
<div class="figurecaption" style="margin: auto; text-align: center;">

**Figure 2:** Normalized curves have a total integral of 1.0

</div>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Curve arithmetic[¶](#Curve-arithmetic){.anchor-link} {#Curve-arithmetic}

`pym` makes it easy to do simple artimetic operations on curves. The
arithmetic all happens after copying the curve, so you don't lose
anything in place. The example below illustrates the common identity
\$\$\\sin\^{2}\\left( \\theta \\right) + \\cos\^{2}\\left( \\theta
\\right) = 1\$\$.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[4\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    one = sin * sin + cos * cos
    one.name = r'$\sin^{2}\left( \theta \right) + \cos^{2}\left( \theta \right) = 1$'
    sin2 = sin * sin
    cos2 = cos * cos

    plot = one.plot(linestyle='-', linecolor='#999999')
    plot = sin2.plot(linestyle='--', linecolor='#FC8D82', addto=plot)

    plot.fill_between(sin2.x, np.zeros_like(sin2.y), sin2.y, fc='#ffcccc', name=r'$\sin^{2} \left( \theta \right)')
    plot.fill_between(cos2.x, sin2.y, sin2.y + cos2.y, fc='#ccccff', name=r'$\sin^{2} \left( \theta \right)')

    plot.markers_off()
    plot.lines_on()

    plot.xlim(0, 12)
    plot.ylim(0, 1.1)
    #plot.legend(loc=1)

    plot.export('_static/identity', ratio='silver')
    plot.show('Trigonometric identity and its contributions from $\cos^{2}$ and $\sin^{2}$')

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_html rendered_html output_subarea">

<div class="pygfigure"
name="Trigonometric identity and its contributions from $\cos^{2}$ and $\sin^{2}$"
style="text-align: center; max-width: 800px; margin-left: auto; margin-right: auto;">

![](_static/identity.svg?729022304)
<div class="figurecaption" style="margin: auto; text-align: center;">

**Figure 3:** Trigonometric identity and its contributions from
\$\\cos\^{2}\$ and \$\\sin\^{2}\$

</div>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Subclassing[¶](#Subclassing){.anchor-link} {#Subclassing}
------------------------------------------

`pym` is easily subclassable. I personally like to define classes for
data I download from instruments and write a load script and add some
properties to the object. The following example shows elevation plotting
for the Pacific Crest Trail using the wonderful
[postholer.com](www.postholer.com).

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[5\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    import urllib
    from bs4 import BeautifulSoup

    class trail(pym.curve):
        def __init__(self, trail_name):
            # first we have to download the trail off of postholer.com
            trail_string = trail_name.replace(' ', '-')
            url = 'http://www.postholer.com/databook/{trail}'.format(trail=trail_string)
            page = urllib.urlopen(url)
            pagestr = page.read()
            self.soup = BeautifulSoup(pagestr, 'lxml')
            # then we have to find the table with the mileage and elevation data
            for table in self.soup.find_all("table"):
                for row in table.find_all("tr"):
                    for cell in row.find_all("td"):
                        if cell.string == "Elev":
                            self.table = table
                            break
            # then we read the mileage and elevation data into lists
            mile = []
            elev = []
            for row in self.table.find_all("tr")[1:]:
                mile.extend([float(row.find_all("td")[1].string)])
                elev.extend([float(row.find_all("td")[5].string)])
            # finally, we initalize the parent class ``curve`` of this object with the data downloaded
            # and the name
            super(trail, self).__init__(mile, elev, name=trail_name)
            
    # lets download three long distance western trails
    pct = trail('pacific crest trail')
    cdt = trail('continental divide trail')
    ct = trail('colorado trail')
    # now that we've initialized the ``trail``s, we can treat them as curves
    plot = pct.plot(linestyle='-', linecolor='#7299c6')
    plot = cdt.plot(linestyle='-', linecolor='#baa892', addto=plot)
    plot = ct.plot(linestyle="-", linecolor='#3f4b00', addto=plot)
    plot.xlabel("Miles since trail start ($s$) [$\unit{mi}$]")
    plot.ylabel("Elevation ($E$) [$\unit{ft}$]")
    plot.lines_on()
    plot.markers_off()
    plot.ylim(0, 12000)
    plot.legend(loc=2)
    plot.export('_static/trail_elevations')
    plot.show('First section elevation of several long distance hiking trails')

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_html rendered_html output_subarea">

<div class="pygfigure"
name="First section elevation of several long distance hiking trails"
style="text-align: center; max-width: 800px; margin-left: auto; margin-right: auto;">

![](_static/trail_elevations.svg?1838488066)
<div class="figurecaption" style="margin: auto; text-align: center;">

**Figure 4:** First section elevation of several long distance hiking
trails

</div>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Fitting[¶](#Fitting){.anchor-link} {#Fitting}
----------------------------------

`pym` has a quick interface for fitting functions to its curves, and
then plotting these.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[6\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    # coming soon

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Interpolation and error propagation[¶](#Interpolation-and-error-propagation){.anchor-link} {#Interpolation-and-error-propagation}
------------------------------------------------------------------------------------------

`pym` uses a linear interpolation backend to make its curve objects
continuous, and it also propagates the error throughout when operations
are performed on it.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[7\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    # coming soon

</div>

</div>

</div>

</div>

</div>
