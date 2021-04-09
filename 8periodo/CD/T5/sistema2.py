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
    io_sys = ct.LinearIOSystem(sys)
    t, y, x = ct.input_output_response(io_sys, t, u, x0, return_x=True)
    return y[-1], x.T[-1]

# Defina o modelo do sistema e a taxa de amostragem
G  = ct.tf(1, np.convolve([1,2],[1,0.2,0.65]))
Ts = 0.9

# Quantidade de amostras
m = 20

# Defina os critérios de desempenho
ts = 10   # Tempo de acomodação
OS = 0.1  # Sobressinal (%/100)

# Polos de malha fechada
z, w, pd = ctrl.param(ts, OS) # Polo desejado
pnd = np.array([-5,-4])       # Polo não dominante

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

# Controlabilidade e Observabilidade
Ctrb, Obsv = ct.ctrb(Ad,Bd), ct.obsv(sysz.A, sysz.C)

# Verifica a controlabilidade e observabilidade
assert np.linalg.matrix_rank(Ctrb)==n+1, 'Não há controlabilidade'
assert np.linalg.matrix_rank(Obsv)==n, 'Não há observabilidade'

# Projeto do controladdor com integrador
K  = ct.place(Ad,Bd,P) # projeto
Kp = K[0,0:-1]         # Ganho para os estados do sistema
Ka = K[0,-1]           # Ganho do integrador

Acl = Ad-Bd*K
#eigAcl = np.linalg.eig(Acl)[0]
#eigAcl.sort()

# Polos desejados do Observador
Pobs = np.array([Ts/4, Ts/4.1, Ts/4.2])
Ke   = ct.place(sysz.A.T, sysz.C.T, Pobs) # Proj. Observador

# Defina as condições iniciais dos estados
x0sys = np.array([0, 0, 0]) # Cond. inicial do sistema
x0obs = np.array([0, 0, 0]) # Cond. inicial do observador
xa0   = 0  # Cond. inicial do integrador

# Pré-alocação
r  = np.ones(m)        # Referência
r[0] = 0
u  = np.zeros_like(r)  # Sinal de controle
xa = np.zeros_like(r)  # Estado do integrador

y  = np.zeros(m+1)     # Saída do sistema
x  = np.zeros((m+1,n)) # Estados do sistema
xe = np.zeros_like(x)  # Estados do observador

# Adicionando as condições iniciais
x[0], xe[0], xa[0] = x0sys, x0obs, xa0

t0 = 0 # Tempo de início da simulação
t  = np.asarray(range(m))*Ts # Tempo da simulação

for k in range(m)[1:]:
    xek = xe[k].reshape((n,1))
    xa[k] = xa[k-1] + r[k] - y[k]
    uk = -Kp*xek - Ka*xa[k]
    u[k] = uk[0,0]

    y1, x1 = simCont(sys, t0, Ts, 100, u[k], x[k])
    y[k+1] = y1
    x[k+1] = x1
    xe[k+1] = (sysz.A*xek + sysz.B*uk + Ke.T*(y[k] -sysz.C*xek)).reshape(n)

    t0 += Ts

ylabels=['$y(t)$']
ctrl.PlotData(t, y[:-1], r, u, True, ylabels)
