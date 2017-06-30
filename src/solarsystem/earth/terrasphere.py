# -*- coding: utf-8 -*-

import numpy as mNumpy
import cv2

from os import path

import solarsystem

from solarsystem.earth import Relation, Grid
from solarsystem import shape, zero, bottom, theta, phi, dSr, mAltitude, mLongitude, mLatitude
from physicalconstant import EARTH_STEFAN_BOLTZMANN_CONSTANT, EARTH_WATER_HEAT_CAPACITY, EARTH_ROCK_HEAT_CAPACITY, EARTH_WATER_DENSITY, EARTH_ROCK_DENSITY, SUN_CONST


if not path.exists('data/continent.npy'):
    im = cv2.imread('data/earth-continent.png', 0)
    mNumpy.save('data/continent', im > 250)

cntndata = mNumpy.array(mNumpy.load('data/continent.npy'), dtype=mNumpy.float64).T
cntndata = (cv2.resize(cntndata, (shape[1], shape[0])))[:, :, mNumpy.newaxis]


def continent():
    return (cntndata > 0.9)


def relu(x):
    return x * (x > 0)


def zinit(**kwargs):
    return mNumpy.copy(zero)


def tinit():
    return 278.15 * bottom


class TLGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(TLGrd, self).__init__('lt', lng_size, lat_size, alt_size, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        contnt = continent()
        capacity = EARTH_WATER_HEAT_CAPACITY * (1 - contnt) + EARTH_ROCK_HEAT_CAPACITY * contnt
        density = EARTH_WATER_DENSITY * (1 - contnt) + EARTH_ROCK_DENSITY * contnt

        return (si + EARTH_STEFAN_BOLTZMANN_CONSTANT * T * T * T * T / 2 - EARTH_STEFAN_BOLTZMANN_CONSTANT * lt * lt * lt * lt) / (capacity * density) * bottom


class SIGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(SIGrd, self).__init__('si', lng_size, lat_size, alt_size, initfn=zinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        albedo = 0.7 * (lt > 273.15) + 0.1 * (lt < 273.15) # considering ice and soil

        doy = mNumpy.mod(solarsystem.t / 3600 / 24, 365.24)
        hod = mNumpy.mod(solarsystem.t / 3600 - mLongitude / 15.0, 24)
        ha = 2 * mNumpy.pi * hod / 24
        decline = - 23.44 / 180 * mNumpy.pi * mNumpy.cos(2 * mNumpy.pi * (doy + 10) / 365)
        sza_coeff = mNumpy.sin(phi) * mNumpy.sin(decline) + mNumpy.cos(phi) * mNumpy.cos(decline) * mNumpy.cos(ha)

        return albedo * relu(sza_coeff) * SUN_CONST * tc * bottom


class TotalCloudage(Relation):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(TotalCloudage, self).__init__('tc', lng_size, lat_size, alt_size, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        dT = solarsystem.earth.context['T'].drvval
        cloudage = mNumpy.sqrt(q) * (dT < 0) * (q > 0.0001)

        ratio = 1 - cloudage
        ratio_total = mNumpy.copy(bottom)
        for ix in range(32):
            ratio_total[:, :, 0] = ratio_total[:, :, 0] * ratio[:, :, ix]

        return 1 - ratio_total

