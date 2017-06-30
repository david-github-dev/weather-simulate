# -*- coding: utf-8 -*-

import numpy as mNumpy
from physicalconstant import EARTH_MEAN_RADIUS

t = 0

context = {
    't': t,
}

dLongitude = 2
dLatitude = 2
dAltitude = 500.0


mLatitude, mLongitude, mAltitude = mNumpy.meshgrid(mNumpy.arange(-89.5, 89.5, dLatitude), mNumpy.arange(-180, 181, dLongitude), mNumpy.arange(0, 16000, dAltitude))

shape = mLongitude.shape
one = mNumpy.ones(mLongitude.shape)
zero = mNumpy.zeros(mLongitude.shape)
bottom = mNumpy.zeros(mLongitude.shape)
top = mNumpy.zeros(mLongitude.shape)
mNorthPole = mNumpy.zeros(mLongitude.shape)
mSouthPole = mNumpy.zeros(mLongitude.shape)

bottom[:, :, 0] = 1
top[:, :, -1] = 1

mNorthPole[:, 0, :] = 1
mSouthPole[:, -1, :] = 1

dth = mNumpy.pi / 180 * dLongitude * one
dph = mNumpy.pi / 180 * dLatitude * one
dr = 1 * dAltitude * one

r = EARTH_MEAN_RADIUS + mAltitude
theta = mLongitude * mNumpy.pi / 180
phi = mLatitude * mNumpy.pi / 180

rx = mNumpy.cos(phi) * mNumpy.cos(theta)
ry = mNumpy.cos(phi) * mNumpy.sin(theta)
rz = mNumpy.sin(phi)

R = mNumpy.concatenate([mNumpy.expand_dims(rx, 3), mNumpy.expand_dims(ry, 3), mNumpy.expand_dims(rz, 3)], axis=3)

thx = mNumpy.sin(phi) * mNumpy.cos(theta)
thy = mNumpy.sin(phi) * mNumpy.sin(theta)
thz = - mNumpy.cos(phi)

Th = mNumpy.concatenate([mNumpy.expand_dims(thx, 3), mNumpy.expand_dims(thy, 3), mNumpy.expand_dims(thz, 3)], axis=3)

phx = - mNumpy.sin(theta)
phy = mNumpy.cos(theta)
phz = mNumpy.copy(zero)

Ph = mNumpy.concatenate([mNumpy.expand_dims(phx, 3), mNumpy.expand_dims(phy, 3), mNumpy.expand_dims(phz, 3)], axis=3)

dLr = 1 * dr
dLph = r * dph
dLth = r * mNumpy.cos(phi) * dth
dSr = dLph * dLth
dSph = dLr * dLth
dSth = dLr * dLph
dV = dLr * dLph * dLth


def dpath(df_th, df_ph, df_r):
    return df_r[:, :, :, mNumpy.newaxis] * R + r[:, :, :, mNumpy.newaxis] * df_th[:, :, :, mNumpy.newaxis] * Th + r[:, :, :, mNumpy.newaxis] * mNumpy.cos(phi)[:, :, :, mNumpy.newaxis] * df_ph[:, :, :, mNumpy.newaxis] * Ph


def grad(f):
    f_th, f_ph, f_r = mNumpy.gradient(f)
    return f_r[:, :, :, mNumpy.newaxis] * R + f_th[:, :, :, mNumpy.newaxis] / r[:, :, :, mNumpy.newaxis] * Th + f_ph[:, :, :, mNumpy.newaxis] / (r * mNumpy.cos(phi))[:, :, :, mNumpy.newaxis] * Ph


def div(F):
    Fth = F[:, :, :, 0]
    Fph = F[:, :, :, 1] * mNumpy.cos(phi)
    Fr = F[:, :, :, 2] * r * r
    val_th, _, _ = mNumpy.gradient(Fth)
    _, val_ph, _ = mNumpy.gradient(Fph)
    _, _, val_r = mNumpy.gradient(Fr)

    return (val_th + val_ph) / r / mNumpy.cos(theta) + val_r / r / r


def curl(F):
    Fth = F[:, :, :, 0]
    Fph = F[:, :, :, 1]
    Fr = F[:, :, :, 2]

    val_r = (mNumpy.gradient(Fph * mNumpy.cos(phi))[0] - mNumpy.gradient(Fth)[1]) / r / mNumpy.cos(phi)
    val_th = (mNumpy.gradient(Fr)[1] / mNumpy.cos(phi) - mNumpy.gradient(r * Fph)[2]) / r
    val_ph = (mNumpy.gradient(r * Fth)[2] - mNumpy.gradient(Fr)[0]) / r

    return val_r[:, :, :, mNumpy.newaxis] * R + val_th[:, :, :, mNumpy.newaxis] * Th + val_ph[:, :, :, mNumpy.newaxis] * Ph


def laplacian(f):
    return div(grad(f))