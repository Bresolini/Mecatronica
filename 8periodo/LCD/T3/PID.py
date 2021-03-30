#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:05:15 2021

@author: bernardo
"""

import numpy as np
from numpy.linalg import inv
import control as ct
import matplotlib.pyplot as plt
import ctrl

plt.close('all')

Show = False
Save = True

ts = 0.09
OS = 0.001
Ts = 0.001
n1  = 8000
n2  = 150

# Defina a funçao transf. do processo
numG = np.array([0.3161])          # Numerador
denG = np.array([1,2.303,3.793])   # Denominador

# Funçao transferencia do processo
G = ct.tf(numG, denG)

# ABORDAGEM PID via LGR

z, w, pd = ctrl.param(ts, OS)
Kc, C1 = ctrl.PID_LGR(G, pd[0], Print=False)
H1 = ct.feedback(Kc*C1*G)

if Show:
    ct.rlocus(G*C1, xlim=(-60, 1))
    plt.scatter(np.real(pd), np.imag(pd), c='k', marker='x')

    zz, pp = ct.zero(H1), ct.pole(H1)
    plt.scatter(np.real(pp[0:2]), np.imag(pp[0:2]), c='#DC1C13', marker='x')
    ax = plt.gca().axes
    ax.set_ylim(-30,30)
    if Save:
        plt.savefig('Fig/rlocusCG.eps', format='eps',
                    dpi=3000, bbox_inches='tight')

r1 = np.ones(n1)
Cz1 = ctrl.PIDz(Kc*C1, Ts)

t1, y1, u1 = ctrl.simulaCzGs(G, Cz1, n1, r1, ulim=(-24,24))

# ABORDAGEM PID empírica
t_ = np.arange(0, 5+0.01, 0.01)
t_, y_, _ = ct.forced_response(G, t_, 3*np.ones_like(t_))

_K, _T, _L = ctrl.getKTL(t_, y_, A=3, Plot=False)
tau = _L/(_T+_L)
Kp, Ti, Td = 1.2*_T/_K/_L, 2*_L, 0.5*_L

#Kp = 1.35*_T/_K/_L*(1+0.18*tau/(1-tau))
#Ti = (2.5 - 2*tau)/(1 - 0.39*tau)*_L
#Td = (0.37-0.37*tau)/(1-0.81*tau)*_L

n2=8000
Ts2 = 0.001

Kp, Ti, Td = ctrl.sintoniaPID(_K, _T, _L, metodo='ITAE')
Cz2 = ctrl.discretePID(Kp, Ti, Td, Ts2)

r2 = np.ones(n2)
t2, y2, u2 = ctrl.simulaCzGs(G, Cz2, n2, r2, ulim=(-24,24), qnt=10)

Kp, Ti, Td = ctrl.sintoniaPID(_K, _T, _L)
Cz3 = ctrl.discretePID(Kp, Ti, Td, Ts2)
t3, y3, u3 = ctrl.simulaCzGs(G, Cz3, n2, r2, ulim=(-24,24), qnt=10)


#ctrl.PlotData(t1, y1, r1, u1, True)
ctrl.PlotData(t2, y2, r2, u2, True)
ctrl.PlotData(t3, y3, r2, u3, True)


"""
plt.savefig('Fig/RespPID_LGR.eps',
            format='eps',
            dpi=3000,
            bbox_inches='tight')
"""
