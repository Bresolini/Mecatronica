#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:28:28 2020

@author: bernardo
"""

import numpy as np
from numpy import pi, sqrt,  log, array
from control import tf, zero, pole, evalfr, series, feedback
from control import minreal, rlocus, step_response
import matplotlib.pyplot as plt

plt.close('all')

ts = 0.74        # (s)     tempo de acomodaçao desejado
OS = 7/100   # (%/100) maximo percentual de overshoot
a  = 4        # ( )     constante de projeto (abaco do OGATA)

t= np.arange(0,5+1/64,1/64)

num1 = 5*array([1,0.5])
den1 = np.convolve([1,0],[1,2])
G = tf(num1,den1)

def param2(ts,OS,a=4):
    z = -log(OS)/sqrt(log(OS)**2 + pi**2)
    w = a/(ts*z)
    p1 = array( [-a/ts + 1j*w*sqrt(1-z**2)] )
    p2 = np.conjugate(p1)
    return z, w, p1, p2

def get_ang(pto, z, p, Print=True):
    theta = np.angle( pto-z )
    phi   = np.angle( pto-p )
    beta = pi + sum(theta) - sum(phi)
    if (Print):
        print('Angulos dos zeros: ')
        print(theta*180/pi)
        print('\nAngulos dos polos: ')
        print(phi*180/pi)
        print('\nBeta: ')
        print(beta*180/pi)
    # end if

    return theta, phi, beta

def get_Kc(pto, H):
    Kc = 1/abs( evalfr(H,pto) )
    return Kc


# Exemplo de uso:

z = zero(G)
p = pole(G)

zeta, wn, p1, p2 = param2(ts, OS, a)

phi, theta, beta = get_ang(p1,z,p)

zc = -2
thetac = np.angle(p1 - zc)

phic = thetac + beta
d = -np.imag(p1)/np.tan(phic)
pc = np.double(zc + d)

numc = [1,-zc]
denc = [1,-pc]
Gc = tf(numc,denc)

H = minreal( series(Gc,G), verbose=False)
rlocus(H)

Kc = get_Kc(p1,H)
print('\nGanho do compensador (Kc): ')
print(Kc)

Hmf = feedback(np.double(Kc)*H,1)
print('\nPolos de malha fechada: ')
print(pole(Hmf))

numP = [1, -np.real(pole(Hmf)[-1])]
denP = Hmf.num[0][0]
GP = tf(numP,denP)
HmfP = minreal( GP*Hmf, verbose=False )

_, y = step_response(Hmf, t)

plt.figure(2)
plt.plot(t,y)
plt.xlim([t[0],t[-1]])
plt.grid(linestyle='--')
plt.show()