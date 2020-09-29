#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 12:23:27 2020

@author: bernardo
"""

import numpy as np
from numpy import pi, sqrt,  log, log10, logspace, array
from control import tf, bode, minreal, rlocus
import matplotlib.pyplot as plt

plt.close('all')


"""
# Defina as constantes usadas
T  = 1
Kc = 1
a  = 2

w = logspace(-5,3,1000)
"""
# Compensador em atraso
"""
    Considere o compensador

                        s + z         T*s + 1
            Gc(s) = Kc ------- = Kc -----------
                        s + p        a*T*s + 1
    em que a > 1 (isto eh, z > p).


                                            ^
                                            |
                                            -
                                           /|
                                         /  |
                                       /    |
                                     /      |
                                   /        |
                                 /          -
                               /           /|
                             /           /  |
                           /           /    |
                         /           /      |
                       /           /        |
                     /           /          |
                   /  φ        /  θ         |
    -------------o-----------x--------------|------->
                -z          -p

    A fase resultante eh dada por φ - θ < 0,
    por isso eh chamado de atraso de fase (LAG COMPENSATION).
"""

def param2(ts,OS,a=4):
    z = -log(OS)/sqrt(log(OS)**2 + pi**2)
    w = a/(ts*z)
    p1 = array( -a/ts + 1j*w*sqrt(1-z**2) )
    p2 = np.conjugate(p1)
    return z, w, p1, p2

def get_ang(pto, z, p):
    theta = np.angle( pto-z )
    phi   = np.angle( pto-p )
    beta = pi + sum(theta) - sum(phi)
    return theta, phi, beta
"""
numAtraso = Kc*array([T,   1])
denAtraso =    array([a*T, 1])
GAtraso = tf(numAtraso, denAtraso)

# Obtendo a resposta em frequencia
mag, fase, omega = bode(GAtraso,w,Plot=False)

# Valores maximos
wm = 1/(T*sqrt(a))
phim = 180*np.arcsin( (1-a)/(1+a) )/pi

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(omega,20*log10(mag))
plt.xscale('log')
plt.xlim([w[0],w[-1]])
plt.grid(linestyle=':',which='both')

plt.subplot(2,1,2)
plt.plot(omega,180*fase/pi)
plt.xscale('log')
plt.xlim([w[0],w[-1]])
plt.grid(linestyle=':',which='both')
plt.plot(wm,phim,'r.')
plt.show()
"""

# Exemplo de uso:

z = [-0.5]
p = [0, -2]

zeta, wn, p1, p2 = param2(1, 0.1)
pto = array(-3.5 + 4.77*1j)
phi, theta, beta = get_ang(p1,z,p)
print('Angulos dos zeros: ')
print(phi*180/pi)
print('\nAngulos dos polos: ')
print(theta*180/pi)

print('\n\n beta: ')
print(beta*180/pi)


zc = -6
thetac = np.angle(pto - zc)

phic = thetac + beta
d = np.real(pto)/np.tan(phic)
pc = np.double(zc + d)

numc = [1,-zc]
denc = [1,-pc]
Gc = tf(numc,denc)

num1 = 5*array([1,0.5])
den1 = np.convolve([1,0],[1,2])
G = tf(num1,den1)

H = minreal(Gc*G)
rlocus(H)
