# -*- coding: utf-8 -*-

import numpy as mNumpy
import cv2 as mOpenCV
from os import path as mPath

import physicalconstant as mPhysicalConstant
import solarsystem as mSolarSystem
from solarsystem import shape, zero, bottom, theta, phi, dSr, mAltitude, mLongitude, mLatitude


if not mPath.exists('data/continent.npy'):
    mImage = mOpenCV.imread('data/earth-continent.png', 0)
    mNumpy.save('data/continent', mImage > 200)

mContinentData = mNumpy.array(mNumpy.load('data/continent.npy'), dtype=mNumpy.float64).T
mContinentData = (mOpenCV.resize(mContinentData, (shape[1], shape[0])))[:, :, mNumpy.newaxis]


def continent():
    return (mContinentData > 0.9)


def relu(x):
    return x * (x > 0)


def zinit(**kwargs):
    return mNumpy.copy(zero)


def tinit():
    return 278.15 * bottom


class TLGrd(mSolarSystem.earth.Grid):
    def __init__(self, shape):
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        super(TLGrd, self).__init__('lt', mLongitudeSize, mLatitudeSize, mAltitudeSize, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        contnt = continent()
        capacity = mPhysicalConstant.EARTH_WATER_HEAT_CAPACITY * (1 - contnt) + mPhysicalConstant.EARTH_ROCK_HEAT_CAPACITY * contnt
        density = mPhysicalConstant.EARTH_WATER_DENSITY * (1 - contnt) + mPhysicalConstant.EARTH_ROCK_DENSITY * contnt

        return (si + mPhysicalConstant.EARTH_STEFAN_BOLTZMANN_CONSTANT * T * T * T * T / 2 - mPhysicalConstant.EARTH_STEFAN_BOLTZMANN_CONSTANT * lt * lt * lt * lt) / (capacity * density) * bottom


class SIGrd(mSolarSystem.earth.Grid):
    def __init__(self, shape):
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        super(SIGrd, self).__init__('si', mLongitudeSize, mLatitudeSize, mAltitudeSize, initfn=zinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        albedo = 0.7 * (lt > 273.15) + 0.1 * (lt < 273.15) # considering ice and soil

        doy = mNumpy.mod(mSolarSystem.t / 3600 / 24, 365.24)
        hod = mNumpy.mod(mSolarSystem.t / 3600 - mLongitude / 15.0, 24)
        ha = 2 * mNumpy.pi * hod / 24
        decline = - 23.44 / 180 * mNumpy.pi * mNumpy.cos(2 * mNumpy.pi * (doy + 10) / 365)
        sza_coeff = mNumpy.sin(phi) * mNumpy.sin(decline) + mNumpy.cos(phi) * mNumpy.cos(decline) * mNumpy.cos(ha)

        return albedo * relu(sza_coeff) * mPhysicalConstant.SUN_CONST * tc * bottom


class TotalCloudage(mSolarSystem.earth.Relation):
    def __init__(self, shape):
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        super(TotalCloudage, self).__init__('tc', mLongitudeSize, mLatitudeSize, mAltitudeSize, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        dT = mSolarSystem.earth.context['T'].drvval
        cloudage = mNumpy.sqrt(q) * (dT < 0) * (q > 0.0001)

        ratio = 1 - cloudage
        ratio_total = mNumpy.copy(bottom)
        for ix in range(32):
            ratio_total[:, :, 0] = ratio_total[:, :, 0] * ratio[:, :, ix]

        return 1 - ratio_total

