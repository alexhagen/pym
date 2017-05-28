
# pym - A math and numerical methods library in python

Last Updated on 5/28/17

Written by [Alex Hagen](http://alexhagen.github.io)

Hosted at [github.com/alexhagen/pym](http://github.com/alexhagen/pym)

Documentation at [alexhagen.github.io/pym/](http://alexhagen.github.io/pym/)

`pym` (pronounced <i>pim</i>) is a pretty simple numerical methods library for python.  It can be used to do interpolation, extrapolation, integration, normalization, etc.  In general, it is a replication of Brian Bradie's book *A Friendly Introduction to Numerical Methods* in code.

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
from pym import func as pym
```

### Curve creation and graphing

The basis of `pym` is the `curve` class, which holds x and y data, as well as
its associated error.  We can create a function with a sinusoid in it by using
the following code


```python
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
```



                <div class='pygfigure' name='A pretty chart from data made for a pym curve' style='text-align: center; max-width: 800px; margin-left: auto; margin-right: auto;'>
                    <img style='margin: auto; max-width:100%; width:1250.000000px; height: auto;' src='_static/curve_plotting.svg?750480266' />
                    <div style='margin: auto; text-align: center;' class='figurecaption'><b>Figure 1:</b> A pretty chart from data made for a pym curve</div>
                </div>
            


### Integration and normalization

One of the useful options of `pym` is the ability to normalize a function,
either according to its maximum, or according to its integral.  The following is
and example of this integration, showing that after integration, we attain an
integral of 1.0.


```python
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
```



                <div class='pygfigure' name='Normalized curves have a total integral of 1.0' style='text-align: center; max-width: 800px; margin-left: auto; margin-right: auto;'>
                    <img style='margin: auto; max-width:100%; width:1250.000000px; height: auto;' src='_static/int_norm.svg?1900945700' />
                    <div style='margin: auto; text-align: center;' class='figurecaption'><b>Figure 2:</b> Normalized curves have a total integral of 1.0</div>
                </div>
            


### Curve arithmetic

``pym`` makes it easy to do simple artimetic operations on curves.  The arithmetic all happens after copying the curve, so you don't lose anything in place.  The example below illustrates the common identity $$\sin^{2}\left( \theta \right) + \cos^{2}\left( \theta \right) = 1$$


```python
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
```



                <div class='pygfigure' name='Trigonometric identity and its contributions from $\cos^{2}$ and $\sin^{2}$' style='text-align: center; max-width: 800px; margin-left: auto; margin-right: auto;'>
                    <img style='margin: auto; max-width:100%; width:1250.000000px; height: auto;' src='_static/identity.svg?683575342' />
                    <div style='margin: auto; text-align: center;' class='figurecaption'><b>Figure 3:</b> Trigonometric identity and its contributions from $\cos^{2}$ and $\sin^{2}$</div>
                </div>
            


## Subclassing

``pym`` is easily subclassable.  I personally like to define classes for data I download from instruments and write a load script and add some properties to the object. The following example shows elevation plotting for the Pacific Crest Trail using the wonderful [postholer.com](www.postholer.com).


```python
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
```



                <div class='pygfigure' name='First section elevation of several long distance hiking trails' style='text-align: center; max-width: 800px; margin-left: auto; margin-right: auto;'>
                    <img style='margin: auto; max-width:100%; width:1250.000000px; height: auto;' src='_static/trail_elevations.svg?519946448' />
                    <div style='margin: auto; text-align: center;' class='figurecaption'><b>Figure 4:</b> First section elevation of several long distance hiking trails</div>
                </div>
            


## Fitting

``pym`` has a quick interface for fitting functions to its curves, and then plotting these.

A cool example of this uses the *Google Trends* database of search trends.  If you search for "Trail Running", or anything outdoorsy, you get back a graph that looks periodic.  Below I have an example using the downloaded U.S. results for "Trail Running" since 2004.  My hypothesis is that the interest peaks in the nice weather (summer), and is at it's nadir during the winter.


```python
import time
import matplotlib.dates as mdates
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange

# Download everything and process into two columns
arr = np.loadtxt('_static/trail_running_search.csv', dtype=str, skiprows=3, delimiter=',')
dates = []
scores = []
for row in arr:
    dates.extend([float(time.mktime(time.strptime(row[0], '%Y-%m')))])
    scores.extend([float(row[1])])
# Now create a curve object with this data
search = pym.curve(dates, scores, name='trail running search')
# The data looks sinusoidal but also with a (sort of) linear increase, so lets make a function that would fit that
def sin_fun(x, a, b, c, T):
    return a * np.sin((x) / (T / (2.0 * np.pi))) + b*x + c
# In general we don't need to guess much, but the period of the function is pretty important
# my hypothesis is that the period is a year.  The rest of the values are just eye-balling off of the chart
T = 365.0 * 24.0 * 60.0 * 60.0
search.fit_gen(sin_fun, guess=(10.0, 1.0E-8, 60.0, T))
# Now that we have the fit, lets plot it
plot = search.plot(linestyle='-', linecolor='#dddddd')
plot = search.plot_fit(linestyle='-', linecolor='#999999', addto=plot)
plot.lines_on()
plot.markers_off()
# And add descriptive statistics to the chart
period = search.coeffs[3]
minx = search.find_min()[0, 0]
miny = search.min()
plot.add_data_pointer(minx + 1.5 * period, point=search.fit_at(minx + 1.5 * period), place=(1.07E9, 90.0),
                      string=time.strftime('Max interest occurs in %B each year', time.gmtime(minx + 1.5 * period)))
plot.add_data_pointer(minx + 2.0 * period, point=search.fit_at(minx + 2.0 * period), place=(1.2E9, 40.0),
                      string=time.strftime('Min interest occurs in %B each year', time.gmtime(minx + 2.0 * period)))
plot.add_hmeasure(minx + 4.0 * period, minx + 8.0 * period, 0.90 * search.fit_at(minx + 4.0 * period),
                  string=r'$4\cdot T \sim %.2f \unit{yr}$' % (4.0*period/60.0/60.0/24.0/365.0))
# label the chart and convert the epoch seconds to years
plot.xlabel('Date ($t$) [$\unit{s}$]')
plot.ylabel('Interest ($\mathbb{I}$) [ ]')
times = [float(time.mktime(time.strptime(str(_y), '%Y'))) for _y in np.arange(2004, 2018, 2)]
times_f = [time.strftime('%Y', time.gmtime(_x)) for _x in times]
plot.ax.xaxis.set_ticks(times)
plot.ax.xaxis.set_ticklabels(times_f)
# finally, lets export
plot.export('trail_running', ratio='silver')
plot.show('Fitting Trail Running Trends with a sinusoid to show its periodic nature')
```



                <div class='pygfigure' name='Fitting Trail Running Trends with a sinusoid to show its periodic nature' style='text-align: center; max-width: 800px; margin-left: auto; margin-right: auto;'>
                    <img style='margin: auto; max-width:100%; width:1250.000000px; height: auto;' src='trail_running.svg?1301352968' />
                    <div style='margin: auto; text-align: center;' class='figurecaption'><b>Figure 5:</b> Fitting Trail Running Trends with a sinusoid to show its periodic nature</div>
                </div>
            


## Interpolation and error propagation

``pym`` uses a linear interpolation backend to make its curve objects continuous, and it also propagates the error throughout when operations are performed on it.


```python
# coming soon
```
