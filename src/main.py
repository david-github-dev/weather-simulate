# -*- coding: utf-8 -*-

import numpy as mNumpy
import pygame
import time

import solarsystem

from physicalconstant import SUN_CONST
from solarsystem.earth.atmosphere import UGrd, VGrd, WGrd, TGrd, RGrd, QGrd, PRel, dHRel, dQRel
from solarsystem.earth.terrasphere import TLGrd, SIGrd, continent, TotalCloudage


u = UGrd(solarsystem.shape)
v = VGrd(solarsystem.shape)
w = WGrd(solarsystem.shape)
T = TGrd(solarsystem.shape)
rao = RGrd(solarsystem.shape)
q = QGrd(solarsystem.shape)

p = PRel(solarsystem.shape)
dH = dHRel(solarsystem.shape)
dQ = dQRel(solarsystem.shape)

tl = TLGrd(solarsystem.shape)
si = SIGrd(solarsystem.shape)
tc = TotalCloudage(solarsystem.shape)

cntn = continent()


def evolve():
    s = mNumpy.sqrt(u.curval * u.curval + v.curval * v.curval + w.curval * w.curval + 0.00001)
    dt = 100 / mNumpy.max(s)
    if dt > 1:
        dt = 1
    solarsystem.t = solarsystem.t + dt
    print '----------------------------------------------------'
    print solarsystem.t, dt
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
    map_width = solarsystem.shape[0]
    map_height = solarsystem.shape[1]

    tile_size = 6
    gap = int(12 / solarsystem.dLongitude)
    wind_size = tile_size * gap

    pygame.init()
    screen = pygame.display.set_mode((map_width * tile_size, map_height * tile_size))
    background = pygame.Surface(screen.get_size())
    tilep = pygame.Surface((tile_size, tile_size))
    tilew = pygame.Surface((wind_size, wind_size))
    tilew.set_alpha(128)

    clock = pygame.time.Clock()

    first_gen = True
    timer = 12

    running = True
    lasttile = 0
    while running == True:
        clock.tick(5)
        time.sleep(1)
        pygame.display.set_caption('FPS: ' + str(clock.get_fps()))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        evolve()

        mapc = cntn[:, :, 0]
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
        g = (128 + ccmap - 72 * mapc) + (256 + ccmap - 72 * mapc) * (tmap <= 273.15)
        b = (128 + ccmap - 72 * mapc) + (256 + ccmap - 72 * mapc) * (tmap <= 273.15)
        bcmap = bcmap + 200
        r = r * bcmap / (255 + 200)
        g = g * bcmap / (255 + 200)
        b = b * bcmap / (255 + 200)

        for ixlng in range(solarsystem.shape[0]):
            for ixlat in range(solarsystem.shape[1]):
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
                rval = rval * (rval > 0) * (rval < 256) + 255 * (rval > 255)
                gval = gval * (rval > 0) * (gval < 256) + 255 * (gval > 255)
                bval = bval * (rval > 0) * (bval < 256) + 255 * (bval > 255)

                tilep.fill((rval, gval, bval))
                tilep.set_alpha((255 - ccolor) / 2)
                screen.blit(tilep, (ixlng * tile_size, ixlat * tile_size))

                if ixlng % gap == 0 and ixlat % gap == 0:
                    length = wind_size / 2 * scolor / 256.0
                    tilew.fill((255, 255, 255))
                    tilew.fill((255, 255, 255))
                    tilew.fill((255, 255, 255))
                    size = length
                    if mNumpy.absolute(uval) >= mNumpy.absolute(vval):
                        alpha = mNumpy.arctan2(uval, vval)
                        pygame.draw.aaline(tilew, (wcolor, ucolor, vcolor), [wind_size / 2.0 - size * mNumpy.cos(alpha), wind_size / 2.0 - size * mNumpy.sin(alpha)],
                                                                            [wind_size / 2.0 + size * mNumpy.cos(alpha), wind_size / 2.0 + size * mNumpy.sin(alpha)], True)
                    else:
                        alpha = mNumpy.arctan2(vval, uval)
                        pygame.draw.aaline(tilew, (wcolor, ucolor, vcolor), [wind_size / 2.0 - size * mNumpy.sin(alpha), wind_size / 2.0 - size * mNumpy.cos(alpha)],
                                                                                           [wind_size / 2.0 + size * mNumpy.sin(alpha), wind_size / 2.0 + size * mNumpy.cos(alpha)], True)

                    screen.blit(tilew, (ixlng * tile_size, ixlat * tile_size))

        flip()
        pygame.display.flip()

        if first_gen:
            timer -= 1
            if timer < 0:
                first_gen = False

    pygame.quit()
