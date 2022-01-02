# Python 3.7

from __future__ import annotations

from kivy.logger import Logger
import astropy.coordinates as ac
import astropy.units as u
import astropy.time
import sgp4.api
import numpy as np

import re
import datetime
import typing
import os



class Satellite :

    TLEs = {}

    @staticmethod
    def load(source:typing.Union[str, bytes, os.PathLike]):
        with open(source, 'r') as f:
            text = f.read()
        pat = re.compile(r"(.*)\n(1.*)\n(2.*)\n")
        for match in re.finditer(pat, text):
            n, l1, l2 = match.groups()
            n = n.strip()
            if n in Satellite.TLEs :
                Satellite.TLEs[n].append((l1.strip(), l2.strip()))
            else :
                Satellite.TLEs[n] = [(l1.strip(), l2.strip())]
        duplicates, d = [], {}
        for s in Satellite.TLEs :
            if isinstance(Satellite.TLEs[s], list):
                if len(Satellite.TLEs[s])==1:
                    Satellite.TLEs[s] = Satellite.TLEs[s][0]
                else :
                    for i, si in enumerate(Satellite.TLEs[s], 1):
                        d[s+f" ({i})"] = si
                    duplicates.append(s)
        for s in duplicates :
            Satellite.TLEs.pop(s)
        Satellite.TLEs = {**Satellite.TLEs, **d}
    

    def __init__(self, name:str, when:datetime.datetime=None):
        self.name = name
        self.satrec = sgp4.api.Satrec.twoline2rv(*self.TLEs[name])
        self.period = datetime.timedelta(minutes = 2 * np.pi / self.satrec.nm)
        self.obstime = when or datetime.datetime.now(datetime.timezone.utc)
        self.orbitpath = self.get_orbit().flatten()
        self.pos = tuple(self.orbitpath[:3])
        self.framerot = self.get_earthrotation()


    def get_orbit(self, when:datetime.datetime=None, detail:int=360,
            from_jdates:tuple[typing.Sequence, typing.Sequence] = None, 
            opengl_format:bool=True, ) -> np.ndarray :
        if not from_jdates:
            t = when or self.obstime
            jd,fr=sgp4.api.jday(*t.timetuple()[:6])
            T = self.period.seconds / 86400.0 
            frs, jds = np.modf(np.linspace(jd+fr+0.5, jd+fr+0.5+T, detail))
            jds -= 0.5
        else :
            jds, frs = from_jdates
        e, p, v = self.satrec.sgp4_array(np.array(jds), np.array(frs))
        if e.any():
            for i in np.where(e):
                Logger.error(f"SGP4 : Computational error - {sgp4.api.SGP4_ERRORS[e[i]]}")
        if opengl_format:
            # axis orientations differ
            p = np.stack([p[:,1],p[:,2],p[:,0]]).T / self.satrec.radiusearthkm
        return p


    def get_earthrotation(self, when:datetime.datetime=None) -> float:
        t = when or self.obstime
        TEMEframerotation = ac.TEME(
            (6400*u.km, 0*u.m, 0*u.m), obstime=t).transform_to(
            ac.ITRS(obstime=t)
        )
        angle = TEMEframerotation.earth_location.geodetic.lon.value - 90.0
        return angle
        

    def get_geoposition(self, when:datetime.datetime=None) -> tuple[float,float,float]:
        t = when or self.obstime
        Re = self.satrec.radiusearthkm
        xyz = (self.pos[2]*Re, self.pos[0]*Re, self.pos[1]*Re,)
        geol = ac.TEME(xyz * u.km, obstime=t).transform_to(ac.ITRS(obstime=t)
                ).earth_location.geodetic
        return (geol.lat.value, geol.lon.value, geol.height.value)


    def update(self, when:datetime.datetime=None, set_new:bool=True) -> tuple[float,float,float]:
        t = when or self.obstime
        if set_new and when:
            self.obstime = when
        jd,fr = sgp4.api.jday(*t.timetuple()[:6])
        e, p, v = self.satrec.sgp4(jd, fr)
        if e:
            Logger.error(f"SGP4 : Computational error - {sgp4.api.SGP4_ERRORS[e]}")
        Re = self.satrec.radiusearthkm
        pos = (p[1]/Re, p[2]/Re, p[0]/Re)
        if set_new :
            self.pos = pos
        return pos


    def direction_in_sky(self, lat:float, long:float, alt:float = 0,
            when:datetime.datetime=None) -> tuple[float,float,float]:
        t = when or self.obstime
        pos = self.update(when, False) if when else self.pos
        el = ac.EarthLocation.from_geodetic(long, lat, alt)
        Re = self.satrec.radiusearthkm
        xyz = (pos[2]*Re, pos[0]*Re, pos[1]*Re,)
        aa = ac.TEME(xyz * u.km, obstime=t).transform_to(
            ac.AltAz(location=el, obstime=t)
        )
        return (aa.alt.value, aa.az.value, aa.distance.value)


    def next_transits_from(self, lat:float, long:float, alt:float=0, 
            horizonangle:float=5, start:datetime.datetime=None, maxdays:float=14, 
            localtime=False) -> np.ndarray[np.datetime64]:

        el = ac.EarthLocation.from_geodetic(long, lat, alt)
        Re = self.satrec.radiusearthkm
        t = start or self.obstime.replace(tzinfo=None)
        detail = 90
        dtimes = astropy.time.Time(
            np.arange(t, t+datetime.timedelta(days=maxdays), self.period/detail)
        )
        o1 = self.get_orbit(opengl_format=False, from_jdates=(dtimes.jd1, dtimes.jd2))

        # Eliminate values where it definitely won't pass over based on latitude
        # Then exact computation performed for the remaining
        geol = ac.TEME(o1.T * u.km, obstime=dtimes).transform_to(
            ac.ITRS(obstime=dtimes)).earth_location
        approx_fov = np.degrees(np.arccos(Re / (geol.height.value + Re)))
        subset = np.where(np.abs(geol.lat.value - lat) <= approx_fov)[0]

        if len(subset):
            selection = ac.TEME(o1[subset].T * u.km, obstime=dtimes[subset])
            aa = selection.transform_to(ac.AltAz(location=el, obstime=dtimes[subset]))
            x = dtimes[subset][np.where(aa.alt.value > horizonangle)[0]]
            if localtime :
                offset = datetime.datetime.now() - datetime.datetime.utcnow()
                x += offset
            return x.value
        else :
            return np.array([], dtype='datetime64[ns]')




if __name__ == "__main__":
    Satellite.load('./data/celestrak-TLEs-100brightest.txt')
    s = Satellite('ISS (ZARYA)')
