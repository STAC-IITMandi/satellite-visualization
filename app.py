
from kivy.app import App
from kivy.clock import Clock, ClockEvent
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.logger import Logger
from kivy.resources import resource_find
from kivy.graphics.transformation import Matrix
from kivy.graphics.opengl import glEnable, glDisable, GL_DEPTH_TEST
from kivy.graphics import *
from kivy.properties import *

import numpy as np
import datetime
import threading
import re
import sys

from globegenerator import spherical_mesh_tex as sph
from globegenerator import cube_mesh as cube
from satellite import Satellite



def normalise(vec:np.ndarray, up:bool=True) -> np.ndarray :
    v = vec / np.linalg.norm(vec)
    if up and vec[1] < 0:
        return -v
    else :
        return v


class NumEntry(TextInput):
    
    autovalidate = BooleanProperty(True)
    valid = BooleanProperty(False)
    ontext_callbacks = ListProperty([])
    minval = NumericProperty(None, allownone=True)
    maxval = NumericProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super(NumEntry, self).__init__(**kwargs)
        self.multiline = False
        self.write_tab = False

    def insert_text(self, substring:str, from_undo:bool=False) -> str:
        if re.search("[^\d\.\+\-eE]", substring):
            return super(NumEntry, self).insert_text('', from_undo=from_undo)
        return super(NumEntry, self).insert_text(substring, from_undo=from_undo)

    def on_text(self, widget, text:str):
        if self.autovalidate:
            try:
                f = float(text)
                if self.minval is not None and f < self.minval :
                    self.valid = False
                elif self.maxval is not None and f > self.maxval :
                    self.valid = False
                else :
                    self.valid = True
            except ValueError:
                self.valid = False
        for fn in self.ontext_callbacks:
            fn(widget, text)

    def on_valid(self, widget, val):
        self.foreground_color = [0,0,0,1] if val else [1,0,0,1]

    def get(self):
        try:
            return float(self.text) if self.valid else None
        except ValueError:
            return None




class Renderer(Widget):


    def __init__(self, **kwargs):
        self.canvas = RenderContext(compute_normal_mat=True, 
                            with_depthbuffer=True,
                            size=self.size)
        super(Renderer, self).__init__(**kwargs)

        self.sat = Satellite(App.get_running_app().default_sat)
        self.geo_lat = None
        self.geo_long = None
        self.update2_thread = None
        self.update2_callback = None

        x0, y0, z0 = 0, 1, 2
        self.loc = np.array([x0, y0, z0])
        self.vertical = normalise(np.array([0, -z0, -y0]))
        self.horizontal = normalise(np.cross(-self.loc, self.vertical), up=False)
        self.current_touches = []
        self.modelrotate = True

        self.simpleshadercanv = RenderContext(compute_normal_mat=True)
        self.lambertshadercanv = RenderContext(compute_normal_mat=True)
        self.simpleshadercanv.shader.source = resource_find('lines.glsl')
        self.lambertshadercanv.shader.source = resource_find('solids.glsl')

        self.instr = InstructionGroup()
        self.instr.add(self.simpleshadercanv)
        self.instr.add(self.lambertshadercanv)

        self.ax_len = 1.5
        self.setup_scene()


    def setup_scene(self):
        self.canvas.add(self.instr)
        with self.canvas:
            self.cb = Callback(self.setup_gl_context)
            PushMatrix()
            x, y, z = self.loc

            with self.simpleshadercanv :
                Callback(self.setup_gl_context)
                Color(0,0,0,1)
                PushMatrix()
                self.t1 = Translate(-x, -y, -z)
                self.rot1 = Rotate(self.sat.framerot, 0,1,0)
                UpdateNormalMatrix()

                # Draw the TEME frame axes (x,y,z)
                ChangeState(lineColor=(1.,0.,0.,1.))
                self.aX = Mesh(vertices=[0.,0.,0.,self.ax_len,0.,0.], indices=[0,1],
                        mode='lines', fmt=[(b'v_pos', 3, 'float')])
                ChangeState(lineColor=(0.,1.,0.,1.))
                self.aY = Mesh(vertices=[0.,0.,0.,0.,self.ax_len,0.], indices=[0,1],
                        mode='lines', fmt=[(b'v_pos', 3, 'float')])
                ChangeState(lineColor=(0.,0.,1.,1.))
                self.aZ = Mesh(vertices=[0.,0.,0.,0.,0.,self.ax_len], indices=[0,1],
                        mode='lines', fmt=[(b'v_pos', 3, 'float')])

                # Draw line at the current GeoLocation (lat/long)
                ChangeState(lineColor=(1.,1.,0.,1.))
                self.geoptr = Mesh(vertices=[], indices=[0,1],
                        mode='lines', fmt=[(b'v_pos', 3, 'float')])

                # Draw the satellite's orbit path
                ChangeState(lineColor=(0.,1.,1.,1.))
                self.orbmesh = Mesh(
                    vertices = list(self.sat.orbitpath),
                    indices = range(int(self.sat.orbitpath.size / 3)),
                    mode = 'line_strip',
                    fmt = [(b'v_pos', 3, 'float')])
                PopMatrix()
                Callback(self.reset_gl_context)

            
            with self.lambertshadercanv :
                Callback(self.setup_gl_context)
                Color(1,1,1,1)

                # Draw the Earth
                PushMatrix()
                self.t2 = Translate(-x, -y, -z)
                self.rot2 = Rotate(0,0,1,0)
                UpdateNormalMatrix()
                self.globe = Mesh(vertices=sph['v'],
                    indices=sph['f'], 
                    mode='triangles', fmt=sph['format'],
                    source=resource_find('world.topo.bathy.200412.3x5400x2700.jpg'),
                )
                PopMatrix()

                # Draw the satellite
                PushMatrix()
                self.t3 = Translate(-x, -y, -z)
                self.rot3 = Rotate(self.sat.framerot, 0,1,0)
                self.t4 = Translate(*self.sat.pos)
                UpdateNormalMatrix()
                self.satbox = Mesh(vertices=cube['v'], indices=cube['f'],
                         mode='triangles', fmt=cube['format'])
                PopMatrix()
                Callback(self.reset_gl_context)

            PopMatrix()
            self.cb = Callback(self.reset_gl_context)

        # Update the parameters every frame
        # Spin the model, update viewpoint/FOV etc
        self.updateevt1 = Clock.schedule_interval(self.update_glsl, 1 / 60.)
        self.updateevt2 = Clock.schedule_interval(self.update_orbit, 5.)
        self.updateevt3 = Clock.schedule_interval(self.update_sat, 0.5)


    def setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def togglerotation(self, value):
        self.modelrotate = value

    def update_glsl(self, delta):
        
        # The renderer canvas always covers the entire window
        # Necessary take into account its actual size/pos and aspect ratio
        # To prevent distortion or overlap when other widgets are present
        rr = App.get_running_app().root
        w, h, h0 = self.width, max(self.height, 1.), rr.height
        px, py = self.pos
        asp, frac_t, frac_b = w/h,  2*(h0-py)/h-1,  2*(0-py)/h-1
        k = 8
        proj = Matrix().view_clip(-asp/k, asp/k, frac_b/k, frac_t/k, 0.25, 90, 1)

        # The earth (0,0,0) is always centered on screen
        # You can move around it and zoom in/zoom out
        # Compute view matrix based on observer location
        x, y, z = self.loc
        p, q, r = self.vertical
        persp = Matrix().look_at(x, y, z, 0., 0., 0., p, q, r)

        self.lambertshadercanv['theTexture'] = 0
        self.canvas['projection_mat'] = proj
        self.simpleshadercanv['projection_mat'] = proj
        self.lambertshadercanv['projection_mat'] = proj
        self.canvas['modelview_mat'] = persp
        self.simpleshadercanv['modelview_mat'] = persp
        self.lambertshadercanv['modelview_mat'] = persp
        if self.modelrotate and not self.current_touches:
            self.rot1.angle += delta * 50
            self.rot2.angle += delta * 50
            self.rot3.angle += delta * 50
        self.t1.x = -x; self.t1.y = -y; self.t1.z = -z
        self.t2.x = -x; self.t2.y = -y; self.t2.z = -z
        self.t3.x = -x; self.t3.y = -y; self.t3.z = -z


    def update_orbit(self, delta):
        self.sat.obstime = datetime.datetime.now(datetime.timezone.utc)
        self.sat.orbitpath = self.sat.get_orbit().flatten()
        self.sat.pos = x, y, z = tuple(self.sat.orbitpath[:3])
        fr = self.sat.get_earthrotation()
        self.rot1.angle += fr - self.sat.framerot
        self.rot3.angle += fr - self.sat.framerot
        self.sat.framerot = fr
        self.t4.x = x; self.t4.y = y; self.t4.z = z
        self.orbmesh.vertices = list(self.sat.orbitpath)
        self.orbmesh.indices = range(int(self.sat.orbitpath.size / 3))


    def update_sat(self, delta):
        current_time = datetime.datetime.now()
        self.sat.update(datetime.datetime.now(datetime.timezone.utc))
        l1 = App.get_running_app().root.info1
        l2 = App.get_running_app().root.info2
        l1.text = "[color=aaaaaa]TLE epoch " + \
            f"{self.sat.tle_epoch.isoformat(' ')[:-7]}[/color]\n" + \
            f"{current_time.isoformat(' ')[:-4]} Local\n" + \
            f"{self.sat.obstime.isoformat(' ')[:-10]} UTC"
        lat, long, alt = self.sat.get_geoposition()
        l2.text = f"Lat   {lat:.5f}°\nLon   {long:.5f}°\nAlt   {alt:.3f} km"
        self.update_sat_2()


    def update_sat_2(self, force=False):
        # Find next_transits_from(self.loc_*) if that valid; and update the GUI
        # Expensive computation (takes 1-2 sec) -> seperate thread to reduce lag
        l3 = App.get_running_app().root.info3
        ha, md = 5, 10
        def recompute(lat, long):
            l3.text = "Calculating Next transit..."
            t = self.sat.next_transits_from(lat, long, 
                horizonangle=ha, maxdays=md, localtime=False,)
            if len(t) :
                l3.text = f"Next transit for {lat}°, {long}°\nat " + \
                    str(t[0])[:19].replace('T',' ') + " UTC"
            else :
                l3.text = f"Not visible from {lat}°, {long}° in the next {md} days"
        
        if type(self.geo_lat) is float and type(self.geo_long) is float:
            al, az, di = self.sat.direction_in_sky(self.geo_lat, self.geo_long)
            if al > ha :
                l3.text = f"Currently visible from {self.geo_lat}°, {self.geo_long}°" + \
                    f"\nAltitude {al:.2f}°  Azimuth {az:.2f}°"
            elif force or ('Currently visible' in l3.text):
                if not ( isinstance(self.update2_thread, threading.Thread) and \
                              self.update2_thread.is_alive() ):
                    self.update2_thread = threading.Thread(target=recompute,
                                        args=(self.geo_lat, self.geo_long))
                    self.update2_thread.start()
        self.update2_callback = None


    def select_sat(self, text):
        self.sat = Satellite(text)
        self.canvas.remove(self.instr)
        self.simpleshadercanv.clear()
        self.lambertshadercanv.clear()
        self.updateevt1.cancel()
        self.updateevt2.cancel()
        self.updateevt3.cancel()
        self.setup_scene()
        self.update_sat_2(True)
        self.update_latlong(None, None)


    def update_latlong(self, widget, text):
        rr = App.get_running_app().root
        if widget is rr.latinput.__self__ :
            self.geo_lat = widget.get()
        elif widget is rr.longinput.__self__ :
            self.geo_long = widget.get()
        if rr.latinput.valid and rr.longinput.valid:
            if isinstance(self.update2_callback, ClockEvent) :
                self.update2_callback.cancel()
            self.update2_callback = Clock.schedule_once(
                lambda dt: self.update_sat_2(True), 1.0)
            self.geoptr.vertices = [0., 0., 0., 
                *self.sat.get_cartesianposition(self.geo_lat, self.geo_long)]
        else :
            self.geoptr.vertices = []


    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos):
            return
        if len(self.current_touches)==2:
            # Handle pinch/zoom type of gestures
            ta, tb = self.current_touches
            oppx = ta.dx < 0 and tb.dx > 0 or ta.dx > 0 and tb.dx < 0
            oppy = ta.dy < 0 and tb.dy > 0 or ta.dy > 0 and tb.dy < 0
            if oppx and oppy:
                l0, l1 = np.hypot((tb.px-ta.px, tb.x-ta.x),
                                  (tb.py-ta.py, tb.y-ta.y),)
                zoom_sensitivity = 10
                self.zoom_observer((l1-l0)/zoom_sensitivity)
                return
        # Modify observer location in world space based on touch events
        pan_sensitivity = 200
        dx, dy = touch.dx/pan_sensitivity, touch.dy/pan_sensitivity
        self.orbit_observer(dx, dy)


    def orbit_observer(self, dx, dy):
        x, y, z = self.loc
        l, m, n = self.vertical
        p, q, r = self.horizontal
        # X component of touch rotates about vertical axis; Y about horizontal
        mat = Matrix().rotate(-dx, l,m,n).rotate(dy, p,q,r)
            # Matrix().rotate(dx,0,1,0).rotate(dy,p,q,r)
        # Recompute axes
        self.loc = x, y, z = np.array(mat.transform_point(x,y,z))
        if x == 0:
            self.vertical = normalise(np.array([0, -z, -y]))
        elif y == 0:
            self.vertical = np.array([0, 0, 1])
        else :
            self.vertical = normalise(np.array([1, -x/y-z**2/x/y, z/x]))
        self.horizontal = normalise(np.cross(-self.loc, self.vertical), up=False)


    def zoom_observer(self, factor_plusorminus):
        # Polynomial zoom factor to "slow down" zoom rate as observer gets
        # too close to earth's surface or too far away (vs. linear scale)
        m = (self.loc ** 2).sum() + 1e-8
        step = self.loc / 200 * min(300/m, m, 10)
        self.loc -= (factor_plusorminus * step).astype(self.loc.dtype)


    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
                if touch.button == 'scrolldown':
                    self.zoom_observer(+1)
                elif touch.button == 'scrollup':
                    self.zoom_observer(-1)
            else :
                self.current_touches.append(touch)
            touch.grab(self)
        return super(Renderer, self).on_touch_down(touch)


    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
        if touch in self.current_touches:
            self.current_touches.remove(touch)
        return super(Renderer, self).on_touch_up(touch)



class SatelliteApp(App):

    def __init__(self, *args, **kwargs):
        super(SatelliteApp, self).__init__(*args, **kwargs)
        self.sat_choices = list(Satellite.TLEs.keys())
        if 'ISS (ZARYA)' in self.sat_choices :
            self.default_sat = 'ISS (ZARYA)'
        elif len(self.sat_choices) :
            self.default_sat = self.sat_choices[0]
        else :
            Logger.error('Satellite: A data file was not loaded or had no valid TLEs')
            sys.exit(1)



if __name__ == '__main__' :

    # Satellite.load('./data/celestrak-TLEs-geosync.txt')
    # Satellite.load('./data/celestrak-TLEs-gps_op.txt')
    # Satellite.load('./data/celestrak-TLEs-100brightest.txt')
    Satellite.load(resource_find('./data/misc-80.txt'))
    A = SatelliteApp()
    A.run()

