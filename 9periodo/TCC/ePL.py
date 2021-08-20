#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 18:32:55 2021

@author: bernardo
"""

import numpy as np
from numpy.linalg import norm, inv
from numpy.random import rand
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

# Função de normalização linear
# Como a eq. da página 97 do Lim08
def normalize(vec):
    vec = np.asarray(vec)

    vmax = np.max(vec)
    vmin = np.min(vec)

    vnorm = (vec-vmin)/(vmax-vmin)

    return vnorm, vmax, vmin

def get_data(data, p=1):
    # Lendo os valores de entrada e saída
    x_data = np.asarray(data.iloc[:,0:p])
    y_data = np.asarray(data.iloc[:,p:p+1])

    # Vetor coluna
    x_data = x_data.reshape((x_data.shape[0],1))
    y_data = y_data.reshape((y_data.shape[0],1))

    # Normalizando os valores
    xn, *_ = normalize(x_data)
    yn, *_ = normalize(y_data)

    return xn, yn

################################################################
################################################################
################################################################
# Importando os dados dos vetores de dados coletados
df = pd.read_csv('BoxJenkins.csv')

xn, yn = get_data(df)

# Define o ponto de ínicio e término da aprendizagem
ki, kf = 6, 206

# Dimensão do vetor de entrada
p = 1
# Número incial de regras
c = 3

x = xn[ki:]
y = yn[ki:]


# Inicialização

# Inicia os centros dos clusters
# Adotei como os c valores anteriores ao ponto inicial
V = xn[ki-c+1:ki+1]

# Parâmetros
alf  = 1    # Taxa de aprendizagem   (valor não informado pelo autor)
tau  = 0.16 # Limiar para novas regras (informado)
beta = 5*tau  # Taxa de atualização    (informado)
r    = 49  # Parâmetro gaussiano      (informado)
lamb = 1 - (tau/beta + 1)/2 +0.5 # Limiar de remoção (informado)

beta = 0.16
alf = 1
# Inicia o índice de alerta
a = np.zeros((c,1))

# Q0
X0 = np.matrix([[1, xn[ki-1,0]],[1, xn[ki,0]]])
Y0 = np.matrix([[yn[ki-1,0]],[yn[ki,0]]])
Q0 = inv(X0.T@X0)

gam0_i = Q0@X0.T@Y0
for j in range(1,c):
    gam0_i = np.hstack((gam0_i,gam0_i[:,0]))


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# ---------------------------- TREINAMENTO ---------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Inicia Qk e gam, y
Qk = Q0
bk = X0.T@Y0
Gam = gam0_i
yk = Y0[-1]
Yi = X0[-1]@Gam

# Pré-alocação
yy = np.zeros_like(y)
yy[0] = yk

# Previsão
for k in range(1,len(x)):
    z = x[k] # igual a xk do Lim08
    d = norm(V-z, axis=1).reshape((V.shape[0],1)) # Distância do centro
    rho = 1 - d/1            # Compatibilidade
    a = a + beta*(1-rho-a) # Alerta
    mu = np.exp(-r*d**2)   # Grau de ativação gaussiano

    # Calculando Qk+1 pela equação (6.15) do Lim08
    # O cálculo é feito recursivamente
    Xi = np.asmatrix(np.hstack(([1],z)))
    Qk = Qk - Qk@Xi.T@Xi@Qk/(1+Xi@Qk@Xi.T)
    bk = bk + Xi

    if min(a) >= tau:
        c += 1
        print(f'Novo grupo criado! Agora são {c} clusters.')
        print(f'Adição feita na iteração {k}\n')
        Gam = np.hstack((Gam, Gam*mu))
        V = np.vstack((V, z))

        a = np.vstack((a,[0]))
        mu = np.vstack((mu, [1]))
    else:
        s = np.argmax(rho)
        V[s] = V[s] + alf*rho[s]**(1-a[s])*(z-V[s])
        Gam[:,s] = Gam[:,s] + Qk@Xi.T*(y[k] - Xi@Gam[:,s])


    # --------------------------------------------------------
    # Aqui seria feito a exclusão de grupos, mas no exemplo
    # não foi preciso, logo decidiu-se pulá-la
    # --------------------------------------------------------

    Yi = np.asarray(Xi@Gam)
    yk = sum((mu.T*Yi)[0])/sum(mu)

    yy[k] = yk


t = np.arange(ki, len(xn))-ki
#plt.plot(range(ki,kf), Yn[ki:kf])

fig, axs = plt.subplots(2,1,sharex=True)
axs[0].plot(t, y, label='Real')
axs[0].plot(t, yy, label='Previsto')
axs[0].set_xlim(0,290)
axs[0].set_ylabel('%CH$_4$')
axs[0].grid(linestyle='--')
axs[0].legend()

axs[1].plot(t, x)
axs[1].set_xlabel('$k$')
axs[1].set_ylabel('%CO$_2$')
