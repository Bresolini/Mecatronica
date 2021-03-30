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
import ctrl # Adicionar ctrl.py a pasta deste código

plt.close('all')

Show = False
Save = True

ts = 0.09
OS = 0.001
Ts = 0.001
n  = 5000
ulim = (-12,12)

# Defina a funçao transf. do processo
numG = np.array([0.3161])          # Numerador
denG = np.array([1,2.303,3.793])   # Denominador

# Funçao transferencia do processo
G = ct.tf(numG, denG)

# ABORDAGEM PID via LGR

z, w, pd = ctrl.param(ts, OS) # Polo desejado
Kc, C1 = ctrl.PID_LGR(G, pd[0], Print=False) # Projeto
H1 = ct.feedback(Kc*C1*G) # malhe fechada

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

r = np.ones(n) # Sinal de referência
Cz1 = ctrl.discretePID(Kc*C1, Ts) # Controlador 1

# SIMULAÇÂO C1
t, y1, u1 = ctrl.simulaCzGs(G, Cz1, n, r, ulim=ulim)

# SINTONIA PID
# Obter K, T e L
t_ = np.arange(0, 5+0.01, 0.01)
t_, y_, _ = ct.forced_response(G, t_, 3*np.ones_like(t_))
_K, _T, _L = ctrl.getKTL(t_, y_, A=3, Plot=False)

# Controlador 2: sintonia PID ITAE
Kp, Ti, Td = ctrl.sintoniaPID((_K, _T, _L), met='ITAE', modo='servo')
Cz2 = ctrl.PIDz(Kp, Ti, Td, Ts)

# SIMULAÇÂO C2
t, y2, u2 = ctrl.simulaCzGs(G, Cz2, n, r, ulim=ulim, qnt=10)

# Controlador 3: Sintonia PID Ziegler-Nichols
Kp, Ti, Td = ctrl.sintoniaPID((_K, _T, _L))
Cz3 = ctrl.PIDz(Kp, Ti, Td, Ts)

# SIMULAÇÂO
t, y3, u3 = ctrl.simulaCzGs(G, Cz3, n, r, ulim=ulim, qnt=10)

# PLOTAGEM
ylab = ('$y_{LGR}(t)$','$y_{ITAE}(t)$','$y_{ZN}(t)$')
ulab = ('$u_{LGR}(t)$','$u_{ITAE}(t)$','$u_{ZN}(t)$')

ctrl.PlotData(t, [y1,y2,y3], r, [u1,u2,u3], True, ylab, ulab)

"""
plt.savefig('Fig/RespPID_LGR.eps',
            format='eps',
            dpi=3000,
            bbox_inches='tight')
"""
