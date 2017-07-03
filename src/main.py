# -*- coding: utf-8 -*-

import numpy as mNumpy
import pygame as mPygame
import time as mTime
import solarsystem as mSolarSystem

from solarsystem.earth import atmosphere as mAtmosphere
from solarsystem.earth import terrasphere as mTerrasphere


u   = mAtmosphere.UGrd(mSolarSystem.shape)
v   = mAtmosphere.VGrd(mSolarSystem.shape)
w   = mAtmosphere.WGrd(mSolarSystem.shape)
T   = mAtmosphere.TGrd(mSolarSystem.shape)
rao = mAtmosphere.RGrd(mSolarSystem.shape)
q   = mAtmosphere.QGrd(mSolarSystem.shape)

p   = mAtmosphere.PRel(mSolarSystem.shape)
dH  = mAtmosphere.dHRel(mSolarSystem.shape)
dQ  = mAtmosphere.dQRel(mSolarSystem.shape)

tl  = mTerrasphere.TLGrd(mSolarSystem.shape)
si  = mTerrasphere.SIGrd(mSolarSystem.shape)
tc  = mTerrasphere.TotalCloudage(mSolarSystem.shape)

mContinent = mTerrasphere.continent()


def evolve():
    s = mNumpy.sqrt(u.curval * u.curval + v.curval * v.curval + w.curval * w.curval + 0.00001)
    dt = 100 / mNumpy.max(s)
    if dt > 1:
        dt = 1
    mSolarSystem.t = mSolarSystem.t + dt
    print '----------------------------------------------------'
    print mSolarSystem.t, dt
    print 'speed: ', mNumpy.max(s), mNumpy.min(s), mNumpy.mean(s)
    tmp = 0.5 * s[:, :, 0] + 0.5 * s[:, :, 1]
    print 'wind:       ', mNumpy.max(tmp), mNumpy.min(tmp), mNumpy.mean(tmp)
    tmp = T.curval[:, :, 0] - 273.15
    print 'temperature ', mNumpy.max(tmp), mNumpy.min(tmp), mNumpy.mean(tmp)
    tmp = p.curval[:, :, 0] / 101325
    print 'pressure    ', mNumpy.max(tmp), mNumpy.min(tmp), mNumpy.mean(tmp)
    tmp = rao.curval[:, :, 0]
    print 'rao         ', mNumpy.max(tmp), mNumpy.min(tmp), mNumpy.mean(tmp)
    tmp = q.curval[:, :, 0]
    print 'humidity    ', mNumpy.max(tmp), mNumpy.min(tmp), mNumpy.mean(tmp)
    tmp = tc.curval[:, :, 0]
    print 'cldg', mNumpy.max(tmp), mNumpy.min(tmp), mNumpy.mean(tmp)

    u.evolve(dt)
    v.evolve(dt)
    w.evolve(dt)
    T.evolve(dt)
    rao.evolve(dt)
    q.evolve(dt)

    p.evolve(dt)
    dH.evolve(dt)
    dQ.evolve(dt)

    tl.evolve(dt)
    si.evolve(dt)
    tc.evolve(dt)


def flip():
    u.swap()
    v.swap()
    w.swap()
    T.swap()
    rao.swap()
    q.swap()

    p.swap()
    dH.swap()
    dQ.swap()

    tl.swap()
    si.swap()
    tc.swap()


def normalize(array, minv, maxv):
    val = (array - minv) / (maxv - minv + 0.001) * 255
    return val * (val > 0) * (val < 256)


if __name__ == '__main__':
    mWidth = mSolarSystem.shape[0]
    mHeight = mSolarSystem.shape[1]
    mTileSize = 6
    mGap = int(12 / mSolarSystem.dLongitude)
    wind_size = mTileSize * mGap

    mPygame.init()
    mScreen = mPygame.display.set_mode((mWidth * mTileSize, mHeight * mTileSize))
    mBackground = mPygame.Surface(mScreen.get_size())
    tilep = mPygame.Surface((mTileSize, mTileSize))
    tilew = mPygame.Surface((wind_size, wind_size))
    tilew.set_alpha(128)

    mClock = mPygame.time.Clock()

    first_gen = True
    timer = 12

    running = True
    lasttile = 0
    while running == True:
        mClock.tick(5)
        mTime.sleep(1)
        mPygame.display.set_caption('FPS: ' + str(mClock.get_fps()))
        for event in mPygame.event.get():
            if event.type == mPygame.QUIT:
                running = False

        evolve()

        mapc = mContinent[:, :, 0]
        bmap = si.curval[:, :, 0]
        tmap = T.curval[:, :, 0]
        cmap = tc.curval[:, :, 0]
        umap = 0.5 * u.curval[:, :, 0] + 0.5 * u.curval[:, :, 1]
        vmap = 0.5 * v.curval[:, :, 0] + 0.5 * v.curval[:, :, 1]
        wmap = 0.5 * w.curval[:, :, 0] + 0.5 * w.curval[:, :, 1]
        smap = mNumpy.sqrt(umap * umap + vmap * vmap + 0.001)

        bcmap = normalize(bmap, 0, mNumpy.max(bmap))
        tcmap = normalize(tmap, 200, 374)
        ccmap = normalize(cmap, 0, 1)
        scmap = normalize(smap, 0, mNumpy.max(smap))
        ucmap = normalize(umap, 0, mNumpy.max(umap))
        vcmap = normalize(vmap, 0, mNumpy.max(vmap))
        wcmap = normalize(wmap, 0, mNumpy.max(wmap))

        r = (tcmap * 2 / 3 + 72 * mapc) * (tmap > 273.15) + (128 + tcmap / 2 + 72 * mapc) * (tmap <= 273.15)
        g = (128 + ccmap - 72 * mapc) * (tmap > 273.15) + (256 + ccmap - 72 * mapc) * (tmap <= 273.15)
        b = (128 + ccmap - 72 * mapc) * (tmap > 273.15) + (256 + ccmap - 72 * mapc) * (tmap <= 273.15)
        bcmap = bcmap + 100
        r = r * bcmap / (255 + 100)
        g = g * bcmap / (255 + 100)
        b = b * bcmap / (255 + 100)

        for ixlng in range(mSolarSystem.shape[0]):
            for ixlat in range(mSolarSystem.shape[1]):
                uval = umap[ixlng, ixlat]
                vval = vmap[ixlng, ixlat]
                sval = smap[ixlng, ixlat]

                bcolor = bcmap[ixlng, ixlat]
                scolor = scmap[ixlng, ixlat]
                tcolor = tcmap[ixlng, ixlat]
                ccolor = ccmap[ixlng, ixlat]
                ucolor = ucmap[ixlng, ixlat]
                vcolor = vcmap[ixlng, ixlat]
                wcolor = wcmap[ixlng, ixlat]

                rval = r[ixlng, ixlat]
                gval = g[ixlng, ixlat]
                bval = b[ixlng, ixlat]
                rval = rval * (rval > 0) * (rval <= 255) + 255 * (rval > 255)
                gval = gval * (rval > 0) * (gval <= 255) + 255 * (gval > 255)
                bval = bval * (rval > 0) * (bval <= 255) + 255 * (bval > 255)

                try:
                    tilep.fill((rval, gval, bval))
                except:
                    print rval, gval, bval
                tilep.set_alpha((255 - ccolor) / 2)
                mScreen.blit(tilep, (ixlng * mTileSize, ixlat * mTileSize))

                if ixlng % mGap == 0 and ixlat % mGap == 0:
                    length = wind_size / 2 * scolor / 256.0
                    tilew.fill((255, 255, 255))
                    size = length
                    if mNumpy.absolute(uval) >= mNumpy.absolute(vval):
                        alpha = mNumpy.arctan2(vval, uval)
                        mPygame.draw.aaline(tilew, (wcolor, ucolor, vcolor), [wind_size / 2.0 - size * mNumpy.cos(alpha), wind_size / 2.0 - size * mNumpy.sin(alpha)],
                                                                            [wind_size / 2.0 + size * mNumpy.cos(alpha), wind_size / 2.0 + size * mNumpy.sin(alpha)], True)
                    else:
                        alpha = mNumpy.arctan2(uval, vval)
                        mPygame.draw.aaline(tilew, (wcolor, ucolor, vcolor), [wind_size / 2.0 - size * mNumpy.sin(alpha), wind_size / 2.0 - size * mNumpy.cos(alpha)],
                                                                                           [wind_size / 2.0 + size * mNumpy.sin(alpha), wind_size / 2.0 + size * mNumpy.cos(alpha)], True)

                    mScreen.blit(tilew, (ixlng * mTileSize, ixlat * mTileSize))

        flip()
        mPygame.display.flip()

        if first_gen:
            timer -= 1
            if timer < 0:
                first_gen = False

    mPygame.quit()
