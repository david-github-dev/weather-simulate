# -*- coding: utf-8 -*-

import numpy as mNumpy

import solarsystem as mSolarSystem
import physicalconstant as mPhysicalConstant
from solarsystem.earth import terrasphere as mTerrasphere
from solarsystem import earth as mEarth
from solarsystem import R
from solarsystem import dLongitude, dLatitude, dAltitude, one, zero, bottom, top, r, theta, phi, Th, Ph, dSr, dSph, dSth, dV, dpath, div
from solarsystem.earth import Omega, gamma, gammad, cv, cp, R, miu, M, niu_matrix


def relu(x):
    return x * (x > 0)


def zinit(**kwargs):
    return mNumpy.copy(zero)


def uinit(**kwargs):
    return 20 * mNumpy.random.random(mSolarSystem.shape) - 10


def vinit(**kwargs):
    return 20 * mNumpy.random.random(mSolarSystem.shape) - 10


def winit(**kwargs):
    return mNumpy.copy(zero)


def tinit(**kwargs):
    return 288.15 - gamma * mSolarSystem.mAltitude + 2 * mNumpy.random.random(mSolarSystem.shape) - 1


def pinit(**kwargs):
    return 101325 * mNumpy.exp(- mPhysicalConstant.EARTH_STANDARD_GRAVITY * M * mSolarSystem.mAltitude / 288.15 / 8.31447) + 100 * mNumpy.random.random(mSolarSystem.shape) - 50


def rinit(**kwargs):
    t = tinit()
    p = pinit()
    return p / R / t


class UGrd(mEarth.Grid):
    def __init__(self, shape):
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        super(UGrd, self).__init__('u', mLongitudeSize, mLatitudeSize, mAltitudeSize, initfn=uinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        a_th, _, _ = mNumpy.gradient(p * dSth) / (r * mNumpy.cos(phi)) / rao / dV

        f_th = mNumpy.gradient(rao * u * dSth * u)[0] / (r * mNumpy.cos(phi))
        f_ph = mNumpy.gradient(rao * u * dSth * v)[1] / r
        f_r = mNumpy.gradient(rao * u * dSth * w)[2]

        f = 0.0004 * (f_th + f_ph + f_r) / rao / dV

        return u * v / r * mNumpy.tan(phi) - u * w / r - 2 * Omega * (w * mNumpy.cos(phi) - v * mNumpy.sin(phi)) + a_th - f


class VGrd(mEarth.Grid):
    def __init__(self, shape):
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        super(VGrd, self).__init__('v', mLongitudeSize, mLatitudeSize, mAltitudeSize, initfn=vinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        _, a_ph, _ = mNumpy.gradient(p * dSph) / r / rao / dV

        f_th = mNumpy.gradient(rao * v * dSph * u)[0] / (r * mNumpy.cos(phi))
        f_ph = mNumpy.gradient(rao * v * dSph * v)[1] / r
        f_r  = mNumpy.gradient(rao * v * dSph * w)[2]

        f = 0.0004 * (f_th + f_ph + f_r) / rao / dV * r

        return - u * u / r * mNumpy.tan(phi) - v * w / r - 2 * Omega * u * mNumpy.sin(phi) + a_ph - f


class WGrd(mEarth.Grid):
    def __init__(self, shape):
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        super(WGrd, self).__init__('w', mLongitudeSize, mLatitudeSize, mAltitudeSize, initfn=winit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        _, _, a_r = mNumpy.gradient(p * dSr) / rao / dV

        f_th = mNumpy.gradient(rao * w * dSr * u)[0] / (r * mNumpy.cos(phi))
        f_ph = mNumpy.gradient(rao * w * dSr * v)[1] / r
        f_r = mNumpy.gradient(rao * w * dSr * w)[2]

        f = 0.0004 * (f_th + f_ph + f_r) / rao / dV * dAltitude

        dw = (u * u + v * v) / r + 2 * Omega * u * mNumpy.cos(phi) - mPhysicalConstant.EARTH_STANDARD_GRAVITY + a_r - f
        return dw * (1 - bottom) * (1 - top) + (w > 0) * dw * bottom + (w < 0) * dw * top


class RGrd(mEarth.Grid):
    def __init__(self, shape):
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        super(RGrd, self).__init__('rao', mLongitudeSize, mLatitudeSize, mAltitudeSize, initfn=rinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        u_th, _, _ = mNumpy.gradient(u)
        _, v_ph, _ = mNumpy.gradient(v)
        _, _, w_r = mNumpy.gradient(w)

        return rao * (u_th * dSth + v_ph * dSph + w_r * dSr * (1 - bottom)) / dV


class TGrd(mEarth.Grid):
    def __init__(self, shape):
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        super(TGrd, self).__init__('T', mLongitudeSize, mLatitudeSize, mAltitudeSize, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        dp = mSolarSystem.earth.context['p'].drvval
        return dH / dV / rao / cp + dp / rao / cp


class QGrd(mEarth.Grid):
    def __init__(self, shape):
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        super(QGrd, self).__init__('q', mLongitudeSize, mLatitudeSize, mAltitudeSize, initfn=zinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        u_th, _, _ = mNumpy.gradient(u)
        _, v_ph, _ = mNumpy.gradient(v)
        _, _, w_r = mNumpy.gradient(w)

        return q * (u_th * dSth + v_ph * dSph + w_r * dSr * (1 - bottom)) / dV + dQ


class PRel(mEarth.Relation):
    def __init__(self, shape):
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        super(PRel, self).__init__('p', mLongitudeSize, mLatitudeSize, mAltitudeSize, initfn=pinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        return rao * R * T


class dQRel(mEarth.Relation):
    def __init__(self, shape):
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        super(dQRel, self).__init__('dQ', mLongitudeSize, mLatitudeSize, mAltitudeSize)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        dT = mSolarSystem.earth.context['T'].drvval
        cntnt = mTerrasphere.continent()

        return + 0.00001 * T * T * T / 273.15 / (373.15 - T) / (373.15 - T) * (dT > 0) * (1 - cntnt) + 0.000001 * (dT > 0) * cntnt - 0.000001 * (dT < 0) * (q > 0.0001)


class dHRel(mEarth.Relation):
    def __init__(self, shape):
        mLongitudeSize, mLatitudeSize, mAltitudeSize = shape
        super(dHRel, self).__init__('dH', mLongitudeSize, mLatitudeSize, mAltitudeSize)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        doy = mNumpy.mod(mSolarSystem.t / 3600 / 24, 365.24)
        hod = mNumpy.mod(mSolarSystem.t / 3600 - mSolarSystem.mLongitude / 15.0, 24)
        ha = 2 * mNumpy.pi * hod / 24
        decline = - 23.44 / 180 * mNumpy.pi * mNumpy.cos(2 * mNumpy.pi * (doy + 10) / 365)
        sza_coeff = mNumpy.sin(phi) * mNumpy.sin(decline) + mNumpy.cos(phi) * mNumpy.cos(decline) * mNumpy.cos(ha)

        dT = mSolarSystem.earth.context['T'].drvval
        absorbS = mNumpy.sqrt(q) * (dT < 0) * (q > 0.0001)
        absorbL = mNumpy.sqrt(mNumpy.sqrt(q)) * (dT < 0) * (q > 0.0001)

        reachnessS = mNumpy.ones((mSolarSystem.shape[0], mSolarSystem.shape[1], mSolarSystem.shape[2], mSolarSystem.shape[2]))
        for ix in range(32):
            for jx in range(ix, 32):
                for kx in range(ix, jx):
                    reachnessS[:, :, ix, jx] = reachnessS[:, :, ix, jx] * (1 - absorbS[:, :, kx])

        reachnessL = mNumpy.ones((mSolarSystem.shape[0], mSolarSystem.shape[1], mSolarSystem.shape[2], mSolarSystem.shape[2]))
        for ix in range(32):
            for jx in range(ix, 32):
                for kx in range(ix, jx):
                    reachnessL[:, :, ix, jx] = reachnessL[:, :, ix, jx] * (1 - absorbL[:, :, kx])

        income_s = relu(sza_coeff) * mPhysicalConstant.SUN_CONST * top

        lt = lt[:, :, 0::32]
        income_l = mPhysicalConstant.EARTH_STEFAN_BOLTZMANN_CONSTANT * lt * lt * lt * lt * dSr * (lt > 0)
        outcome = mPhysicalConstant.EARTH_STEFAN_BOLTZMANN_CONSTANT * T * T * T * T * dSr * (T > 0)

        fusion = + (lt >= 273.15) * (lt < 275) * dSr * bottom * 0.01 * mPhysicalConstant.EARTH_WATER_DENSITY * 333550 \
                 - (lt > 271) * (lt <= 273.155) * dSr * bottom * 0.01 * mPhysicalConstant.EARTH_WATER_DENSITY * 333550

        income = mNumpy.copy(zero)
        for ix in range(32):
            for jx in range(32):
                income[:, :, ix] += outcome[:, :, jx] * reachnessL[:, :, ix, jx]
            income[:, :, ix] += income_l[:, :, ix] * reachnessL[:, :, ix, 0]
            income[:, :, ix] += income_s[:, :, -1] * reachnessS[:, :, ix, 31]

        return (income - outcome) - 2266000 * dQ * dV + fusion

