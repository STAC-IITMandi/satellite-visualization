
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
from kivy.graphics.opengl import glEnable, glDisable, GL_DEPTH_TEST
from kivy.graphics import *

import numpy as np
import datetime

from globegenerator import spherical_mesh_tex as sph
from globegenerator import cube_mesh as cube
from satellite import Satellite



def normalise(vec, up:bool=True) -> np.ndarray :
    v = vec / np.linalg.norm(vec)
    if up and vec[1] < 0:
        return -v
    else :
        return v



class Renderer(Widget):
    def __init__(self, **kwargs):
        self.canvas = RenderContext(compute_normal_mat=True, 
                            with_depthbuffer=True,
                            size=self.size)
        super(Renderer, self).__init__(**kwargs)

        self.sat = Satellite(App.get_running_app().default_sat)

        x0, y0, z0 = 0, 1, 2
        self.loc = np.array([x0, y0, z0])
        self.vertical = normalise(np.array([0, -z0, -y0]))
        self.horizontal = normalise(np.cross(-self.loc, self.vertical), up=False)
        self.current_touch = None

        self.simpleshadercanv = RenderContext(compute_normal_mat=True)
        self.lambertshadercanv = RenderContext(compute_normal_mat=True)
        self.simpleshadercanv.shader.source = 'lines.glsl'
        self.lambertshadercanv.shader.source = 'solids.glsl'

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
                    source='world.topo.bathy.200412.3x5400x2700.jpg', index=0,
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
        self.updateevt2 = Clock.schedule_interval(self.update_orbit, 10.)
        self.updateevt3 = Clock.schedule_interval(self.update_sat, 0.5)


    def setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)


    def update_glsl(self, delta):
        
        # The renderer canvas always covers the entire window
        # Necessary take into account its actual size/pos and aspect ratio
        # To prevent distortion or overlap when other widgets are present
        rr = App.get_running_app().root
        w, h = self.width, max(self.height, 1.)
        asp, frac = w / h, rr.height / h
        k = 3
        proj = Matrix().view_clip(-asp/k, asp/k, -1/k, frac/k, 0.5, 50, 1)

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
        if not self.current_touch:
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
        y = 2000 if self.sat.satrec.epochyr < 57 else 1900
        td = datetime.datetime(y+self.sat.satrec.epochyr, 1, 1) + \
            datetime.timedelta(days=self.sat.satrec.epochdays)
        l1.text = f"[color=aaaaaa]TLE epoch {td.isoformat(' ')[:-7]}[/color]\n" + \
            f"{current_time.isoformat(' ')[:-4]} Local\n" + \
            f"{self.sat.obstime.isoformat(' ')[:-10]} UTC"
        lat, long, alt = self.sat.get_geoposition()
        l2.text = f"Lat   {lat:.5f}°\nLon   {long:.5f}°\nAlt   {alt:.3f} km"


    def select_sat(self, text):
        self.sat = Satellite(text)
        self.canvas.remove(self.instr)
        self.simpleshadercanv.clear()
        self.lambertshadercanv.clear()
        self.updateevt1.cancel()
        self.updateevt2.cancel()
        self.updateevt3.cancel()
        self.setup_scene()


    def on_touch_move(self, touch):
        # Modify observer location in world space based on touch events
        sensitivity = 200
        dx, dy = touch.dx/sensitivity, touch.dy/sensitivity
        x, y, z = self.loc
        l, m, n = self.vertical
        p, q, r = self.horizontal
        # X component of touch rotates about vertical axis; Y about horizontal
        mat = Matrix().rotate(-dx, l,m,n).rotate(dy, p,q,r)
            # Matrix().rotate(dx,0,1,0).rotate(dy,p,q,r)
        # Recompute axes
        self.loc = np.array(mat.transform_point(x,y,z))
        x, y, z = self.loc
        if x == 0:
            self.vertical = normalise(np.array([0, -z, -y]))
        elif y == 0:
            self.vertical = np.array([0, 0, 1])
        else :
            self.vertical = normalise(np.array([1, -x/y-z**2/x/y, z/x]))
        self.horizontal = normalise(np.cross(-self.loc, self.vertical), up=False)


    def on_touch_down(self, touch):
        step = self.loc / np.linalg.norm(self.loc) / 10
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
                if touch.button == 'scrolldown':
                    self.loc -= step.astype(self.loc.dtype)
                elif touch.button == 'scrollup':
                    self.loc += step.astype(self.loc.dtype)
            else :
                self.current_touch = touch
            touch.grab(self)
        return super(Renderer, self).on_touch_down(touch)


    def on_touch_up(self, touch):
        if touch.grab_current is self:
            self.current_touch = None
            touch.ungrab(self)
        return super(Renderer, self).on_touch_up(touch)



class SatelliteApp(App):

    def __init__(self, *args, **kwargs):
        super(SatelliteApp, self).__init__(*args, **kwargs)
        self.sat_choices = list(Satellite.TLEs.keys())
        self.default_sat = 'ISS (ZARYA)'



if __name__ == '__main__' :

    # Satellite.load('./data/celestrak-TLEs-geosync.txt')
    # Satellite.load('./data/celestrak-TLEs-gps_op.txt')
    Satellite.load('./data/celestrak-TLEs-100brightest.txt')
    A = SatelliteApp()
    A.run()

