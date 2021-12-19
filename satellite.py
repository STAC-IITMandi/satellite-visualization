
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.logger import Logger
from kivy.graphics.transformation import Matrix
from kivy.graphics.opengl import glEnable, glDisable, GL_DEPTH_TEST
from kivy.graphics import *

from globegenerator import spherical_mesh_tex as sph
from globegenerator import cube_mesh as cube

import astropy.coordinates as ac
import astropy.units as u
import sgp4.api
import numpy as np

import re
import datetime

START_TIME = datetime.datetime.now(datetime.timezone.utc)

## International Space station
iss = ('1 25544U 98067A   21343.88482639  .00000143  00000+0  10791-4 0  9990',
       '2 25544  51.6415 193.9729 0004414 287.4837 253.6360 15.48949833315824')
## Atlas
# iss = ('1 38770U 12048N   21344.72361328 0.00002400  00000-0  14519-3 0    08',
#        '2 38770  64.6612 252.4486 0170517 317.7074  42.2926 14.98942986    08')
## INSAT (geostationary)
# iss = ('1 13129U 82031A   21348.37651804 -.00000335  00000-0  00000+0 0  9993',
#        '2 13129  10.7807 313.5418 0015742   8.6672   4.2201  1.00210069149927')
## PSLV rocket body
# iss = ('1 41470U 16027B   21347.42402392  .00002207  00000-0  39076-3 0  9994',
#        '2 41470  17.7589  90.9353 6026464 310.9522  10.7267  4.01028694 81752')


def get_orbit(satellite, when):
    t = when
    jds, frs = [], []
    detail = 360
    period = datetime.timedelta(minutes = 2 * np.pi / satellite.nm)
    for i in range(detail):
        jd,fr=sgp4.api.jday(*t.timetuple()[:6])
        jds.append(jd); frs.append(fr)
        t += period/detail
    e, p, v = satellite.sgp4_array(np.array(jds), np.array(frs))
    for ec in e :
        if ec :
            Logger.error(f"SGP4 : Computational error - {sgp4.api.SGP4_ERRORS[ec]}")
    p = np.stack([p[:,1],p[:,2],p[:,0]]).T
    return p / satellite.radiusearthkm


def get_rotation(when):
    TEMEframerotation = ac.TEME(
        (6400*u.km, 0*u.m, 0*u.m), obstime=when).transform_to(
        ac.ITRS(obstime=when)
    )
    angle = TEMEframerotation.earth_location.geodetic.lon.value - 90.0
    return angle


def normalise(vec, up=True):
    v = vec / np.linalg.norm(vec)
    if up and vec[1] < 0:
        return -v
    else :
        return v


class Renderer(Widget):
    def __init__(self, **kwargs):
        self.canvas = RenderContext(compute_normal_mat=True)
        super(Renderer, self).__init__(**kwargs)

        self.tle = iss
        self.satrec = sgp4.api.Satrec.twoline2rv(*self.tle)
        self.period = datetime.timedelta(days = 1.0/float(self.tle[1][52:63]))
        self.obstime = datetime.datetime.now(datetime.timezone.utc)
        self.orbitpath = get_orbit(self.satrec, START_TIME).flatten()
        self.satpos = tuple(self.orbitpath[:3])
        self.framerot = get_rotation(self.obstime)

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
                self.rot1 = Rotate(self.framerot, 0,1,0)
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
                self.orbmesh = Mesh(vertices = list(self.orbitpath),
                    indices = range(int(self.orbitpath.size / 3)),
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
                self.rot3 = Rotate(self.framerot, 0,1,0)
                self.t4 = Translate(*self.satpos)
                UpdateNormalMatrix()
                self.satbox = Mesh(vertices=cube['v'], indices=cube['f'],
                         mode='triangles', fmt=cube['format'])
                PopMatrix()
                Callback(self.reset_gl_context)

            PopMatrix()
            self.cb = Callback(self.reset_gl_context)

        # Update the parameters every frame
        # Spin the model, update viewpoint/FOV etc
        Clock.schedule_interval(self.update_glsl, 1 / 60.)
        Clock.schedule_interval(self.update_satellite, 5.)


    def setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def update_glsl(self, delta):
        asp = self.width / float(self.height)
        proj = Matrix().view_clip(-asp/3, asp/3, -1/3, 1/3, 1, 50, 1)
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

    def update_satellite(self, delta):
        self.obstime = datetime.datetime.now(datetime.timezone.utc)
        self.orbitpath = get_orbit(self.satrec, START_TIME).flatten()
        self.satpos = x, y, z = tuple(self.orbitpath[:3])
        fr = get_rotation(self.obstime)
        self.rot1.angle += fr - self.framerot
        self.rot3.angle += fr - self.framerot
        self.framerot = fr
        self.t4.x = x; self.t4.y = y; self.t4.z = z
        self.orbmesh.vertices = list(self.orbitpath)
        self.indices = range(int(self.orbitpath.size / 3))

    def on_touch_move(self, touch):
        # Modify observer location in world space based on touch events
        sensitivity = 200
        dx, dy = touch.dx/sensitivity, touch.dy/sensitivity
        x, y, z = self.loc
        l, m, n = self.vertical
        p, q, r = self.horizontal
        # X component of touch rotates about vertical axis; Y about horizontal
        mat = Matrix().rotate(-dx, l,m,n).rotate(dy, p,q,r)
            # mat = Matrix().rotate(dx,0,1,0).rotate(dy,p,q,r)
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
                touch.grab(self)
                self.current_touch = touch
            return True

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            self.current_touch = None
            touch.ungrab(self)
            return True




class RendererApp(App):
    def build(self):
        return Renderer()


if __name__ == "__main__":
    RendererApp().run()