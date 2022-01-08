# satellite-visualization

A cross-platform application to calculate & view the trajectory and live position of any earth-orbiting satellite in 3D.

This is developed upon a sample solution to the *Atlas* astro-hackathon conducted by STAC during the annual intra-college technical fest of IIT Mandi, *Utkarsh*, in Dec 2021 (one of the myriad possibilities, there wasn't any specific correct/expected solution).

### Running the program

You can run this code on your system with an installation of Python 3.7 or newer :

1. Clone this repository locally and `cd` into its folder.

        git clone https://github.com/STAC-IITMandi/satellite-visualization.git && cd satellite-visualization
2. It is strongly [recommended](https://kivy.org/doc/stable/gettingstarted/installation.html#create-virtual-environment) to install Kivy (one of the dependencies) in a virtual environment. <br> Run `python -m venv ./.venv` to create a new one in a subfolder called `.venv` (will remain untracked due to `.gitignore`).
3. [Activate](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) the venv and then install the dependencies, as listed below. All of them are installable from PyPI (`pip install -r requirements.txt`).
4. Run `python app.py`
5. The contents of the `data/` folder will very likely be outdated. If the displayed TLE epoch in the app window is more than a week old, (or the satellite position incorrect), you can update the values as decribed in the **Implementation** section below. Then restart the app.


### Dependencies

```
numpy>=1.17.0
kivy>=2.0.0
astropy>=4.3.0
jplephem>=2.15
sgp4>=2.20
```
Links : [numpy](https://numpy.org/install/), [kivy](https://kivy.org/doc/stable/gettingstarted/installation.html), [astropy](https://docs.astropy.org/en/stable/install.html), [jplephem](https://pypi.org/project/jplephem/), [sgp4](https://pypi.org/project/sgp4/)

- The `[base]` module of `kivy` is sufficient (extras/media/gstreamer and other libraries for audio or video playback are not needed)
- `jplephem` is not used directly in the app, but required by astropy for part of its functionality.
- In case you want to create an executable file from this code, following the directions for [kivy apps on your target OS](https://kivy.org/doc/stable/guide/packaging.html) is sufficient. Note : there have been some issues with specific versions of [`astropy`](https://github.com/astropy/astropy/issues?q=is%3Aissue+pyinstaller) in the frozen/compiled form (using [`pyinstaller`](https://pyinstaller.readthedocs.io/en/stable/)).

#### Other Sources -

[Earth Textures](./earth/) - [NASA Visible Earth Library](https://visibleearth.nasa.gov/images/73909/december-blue-marble-next-generation-w-topography-and-bathymetry)

[TLE Data](./data/) - [Celestrak NORAD database](http://celestrak.com/NORAD/elements/)


### Implementation

The app propagates the satellite's trajectory by itself, using the `sgp4` library, based on the Two-Line Element Set (TLE) format as source for the orbital parameters. Theoretically, any earth-orbiting small body's orbit can be calculated, given its valid TLE.

To perform the complex transformations between the results of SGP4 propagation (position in a cartesian inertial frame of reference), and Earth-based coordinates such as Latitude/Longitude or Altitude/Azimuth, `astropy` is used.

Currently, the TLE sets are in static text files, which are not updated by the app. Due to gradually increasing [errors](https://en.wikipedia.org/wiki/Simplified_perturbations_models#cite_note-revisiting-3), the app's predictions will become noticeably inaccurate within a few days. They must be refreshed separately, on the timescale of days (which is how often updated TLEs are usually made publicly available). This is an example to automate fetching new data from their website every alternate day :
```shell
# crontab
# m h D M W command
0 0 */2 * * curl http://celestrak.com/NORAD/elements/visual.txt > ./data/celestrak-TLEs-100brightest.txt
```

Finally, the GUI framework used is Kivy, which was advantageous here due to its versatility (code written in Python, can be compiled into an executable for multiple Operating Systems, even mobile & touch devices). However the 3D/OpenGL capabilities of Kivy appear not to be very well documented online and so some parts of the code seem to somehow just work magically. 

### Contributing

Contributions are welcome ! If you're interested in helping build this project, please do take a look at the issues to see if any improvements are in progress... Guidelines are described [here](./CONTRIBUTING.md). We appreciate your enthusiasm and effort, no matter whether the change is simple or complex !

