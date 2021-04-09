#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:52:50 2021

@author: bernardo
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ct
import ctrl

plt.close('all')

# Função usada para simular o sistema continuamente
def simCont(sys, t0, tf, qnt, u, x0):
    t = np.linspace(t0,tf+t0,qnt)
    io_sys = ct.tf2io(sys)
    t, y, x = ct.input_output_response(io_sys, t, u, x0, return_x=True)
    return y[-1], x.T[-1]

N = 8  # Ordem do atraso
# Defina o modelo do sistema e a taxa de amostragem
G  = ct.tf([1],[1,0.1])
nd, dd  = ct.pade(2.8, N) # Aproximação do atraso por Padè
delay = ct.tf(nd,dd)
Gn = G*delay # Sistema com atraso

Ts = 0.6 # Taxa de amostragem
qnt = 25 # Quantidade de pontos para a simulação contínua

# Quantidade de amostras
m = 50

# Defina os critérios de desempenho
ts = 20   # Tempo de acomodação
OS = 0.01  # Sobressinal (%/100)

# Polos de malha fechada
z, w, pd = ctrl.param(ts, OS) # Polo desejado
pnd = np.array([])       # Polo não dominante

# Polos em z
P = np.exp(Ts*np.concatenate((pd,pnd)))
P.sort() # Organiza em ordem crescente

# Conversão para espaço de estados contínuo e depois discreto
sys = ct.tf2ss(G)     # Sistema em espaço de estados
sysz = sys.sample(Ts) # Sistema discretizado
n = len(G.den[0][0])-1 # Ordem do sistema

# Forma ampliada
Ad = np.vstack((np.hstack((sysz.A, np.zeros((n,1)))),
                np.hstack((-sysz.C*sysz.A, np.eye(1)))))
Bd = np.concatenate((sysz.B, -sysz.C*sysz.B))

# Controlabilidade
Ctrb = ct.ctrb(Ad,Bd)

# Verifica a controlabilidade e observabilidade
assert np.linalg.matrix_rank(Ctrb)==n+1, 'Não há controlabilidade'

# Projeto do controladdor com integrador
K  = ct.place(Ad,Bd,P) # projeto
Kp = K[0,0:-1]         # Ganho para os estados do sistema
Ka = K[0,-1]           # Ganho do integrador

# Defina as condições iniciais dos estados
x0sys = np.array([0]) # Cond. inicial do sistema
xa0   = 0  # Cond. inicial do integrador

# Pré-alocação
r  = np.ones(m)        # Referência
r[0] = 0
u  = np.zeros_like(r)  # Sinal de controle
xa = np.zeros_like(r)  # Estado do integrador

yn = np.zeros(m+1)     # Saída do sistema
y  = np.zeros_like(yn)
xm = np.zeros((m+1,n)) # Estados do sistema
xd = np.zeros((m+1,N))
xn = np.zeros((m+1,n+N))

# Adicionando as condições iniciais
xm[0], xa[0] = x0sys, xa0

t0 = 0 # Tempo de início da simulação
t  = np.asarray(range(m))*Ts # Tempo da simulação

for k in range(m)[1:]:
    xa[k] = xa[k-1] + r[k] - y[k]
    uk = -Kp*xm[k] - Ka*xa[k]
    u[k] = uk[0,0]

    # Modelo de projeto
    ymk, xmk = simCont(G,     t0, Ts, qnt, u[k], xm[k])
    # Sistema c/ atraso
    ynk, xnk = simCont(Gn,    t0, Ts, qnt, u[k], xn[k])
    # Atraso
    ydk, xdk = simCont(delay, t0, Ts, qnt, ymk, xd[k])

    yn[k+1] = ynk # Saída do sistema
    y[k+1] = ynk + ymk - ydk # Saída p/ o controlador

    xm[k+1] = xmk # Modelo
    xd[k+1] = xdk # Atraso
    xn[k+1] = xnk # Sistema

    t0 += Ts

ylabels=['$y_{controlador}$','$y_{real}(t)$']
ctrl.PlotData(t, [y[:-1], yn[:-1]], r, [u,u], True, ylabels)
