#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 19:32:18 2020

@author: B. Bresolini

Este codigo recebe informaçoes de 'pto_op.csv' e 'modelos.csv' e computa o
controlador via metodo polinomial, compensando os zeros de fase nao minima
e o polo nao dominante atribuido. Ademais, se calcula o valor maximo e mi-
nimo do sinal de controle.

Deve-se atribuir os valores de ts e OS desejados para que o codigo opere.
Deste modo, o projeto do controlador e feito visando um comporamento de
sistema de segunda ordem subamortecido.
"""

import numpy as np
from numpy.linalg import inv
from math import log, pi, sqrt
import pandas as pd
import csv
import matplotlib.pyplot as plt
from control import tf, feedback, minreal, series, dcgain, zero, pole
from control import step_response, bode, forced_response

plt.close('all')
ts = 230    # (s)     tempo de acomodaçao desejado
OS = 0.01   # (%/100) overshoot desejado

# Obtendo os dados do ponto de operaçao
with open('pto_op.csv', 'r') as f:
    reader_op  = csv.reader(f, delimiter=',')
    headers_op = next(reader_op)
    data_op    = np.array(list(reader_op)).astype(float)
yop, uop, tss, dt, max_dy, max_du = data_op[-1]

# Obtendo os dados do modelo
df = pd.read_csv('modelos.csv', index_col='Modelos')
print(df.iloc[:,0:3])
print('\n')

K1, T1, L1 = df.loc['1 ordem'][0:3]
K2, T2, L2 = df.loc['2 ordem'][0:3]

# Informando o maior e menor valor do sinal de controle de malha fechada admitido
print('MIN. VALALOR DE U ADMITIDO: -%0.2f' %(uop))
print('MAX. VALALOR DE U ADMITIDO: +%0.2f\n' %(100-uop))


def sylv(NUM,DEN):
    n, m = len(NUM)-1, len(DEN)-1
    N  = max([n,m])
    S = np.zeros( [ 2*N, 2*N ] )
    for i in range(0,N):
        S[i][i:m+1+i] = DEN[::-1]
        S[i+N][i:n+1+i] = NUM[::-1]

    S = S.transpose()
    return S

z = -log(OS)/sqrt(log(OS)**2 + pi**2)

# Indo no abaco do OGATA, FIG. 5.11. p. 158
a  = 4      #         Constante retirada do abaco
zw = a/ts   #         zeta*omega
w  = zw/z   # (rad/s) Freq. natural


# Defina a funçao transf. do processo
numG = np.array([ K2 ])      # Numerador
denG = np.convolve([T2,1],[T2,1])   # Denominador
# Funçao transferencia do processo
G = tf(numG,denG)

n = len(denG)-1   # Irden de G

# Polos desejados
p = np.zeros(2*n-1,dtype=complex)
p[0] = zw + 1j*w*sqrt(1-z**2)   # Polo dominante
p[1] = np.conj(p[0])

# Polos nao dominantes uma decada abaixo
for i in range( 2,2*n-1 ):
    p[i] = 10*zw + (i-2)
#end for

# Polinomio desejado
D = np.convolve([ 1, p[0] ], [1, p[1]] )
for i in range(2,len(p)):
    D = np.convolve(D, [ 1, p[i] ] )
#end for
D = np.real(D[::-1])   # Inverte o vetor


# Matriz de Sykvester
E = sylv(numG,denG)
#print('Matriz de Sylvester')
#print(E)

# Soluçao do sistema linear
M = inv(E).dot(D)
#print('Coef')
#print(M)

# Denominador de C
A = M[0:n]
A = A[::-1]   # Reverter os valores

# Numerador de C
B = M[n:len(M)]
B = B[::-1]

# Controlador
C = tf(B,A)

# Controlador em serie
H1 = minreal(feedback(G*C,1),verbose=False)   # TF de malha fechada (serie)
kh1 = dcgain(H1)                 # Ganho estatico
H1 = (1/kh1)*H1                  # Malha fechada ganho unitario

# Controlador em paralelo
H2 = minreal(feedback(G,C),verbose=False)    # TF de malha fechada (realimentacao)
kh2 = dcgain(H2)                # Ganho estatico
H2 = (1/kh2)*H2                 # Ganho unitario


# Simulaçao
t = np.arange(0,2*ts+dt,dt)

# O sinal de referencia maximo e minumo sao max_dy e -max_dy
r_max =  max_dy*np.ones(t.shape)
r_min = -max_dy*np.ones(t.shape)

# Controlador em serie
zH1 = zero(H1)   # Armazenando os zeros
denGc1 = 1       # Denominador inicial de Gc1
for i in range( 0, len(zH1) ):
    if (zH1[i] < 0):   # Verifica se o zero 茅 de fase m铆nima
        denGc1 = np.convolve(denGc1, [ 1, -zH1[i] ])
    # end if
# end for
Gc1 = tf([1, np.real(p[-1])], denGc1)
Gc1 = (1/dcgain(Gc1))*Gc1
Hc1 = minreal( series(Gc1,H1),verbose=False )

# Obtençao da resposta para o maior sinal de referencia
_T,  Y1max, _X = forced_response(H1, t, r_max)   # Sem compensaçao
_T, Y1cmax, _X = forced_response(Hc1,t, r_max)   # Com compensaçao

# Obtençao da resposta para o menor sinal de referencia
_T,  Y1min, _X = forced_response(H1, t, r_min)   # Sem compensaçao
_T, Y1cmin, _X = forced_response(Hc1,t, r_min)   # Com compensaçao

# Controlador na realimentaçao
zH2 = zero(H2)   # Armazenando os zeros
denGc2 = 1       # Denominador inicial de Gc2
for i in range( 0, len(zH2) ):
    if (zH2[i] < 0):   # Verifica se o zero 茅 de fase m铆nima
        denGc2 = np.convolve(denGc2, [ 1, -zH2[i] ])
    # end if
# end for
Gc2 = tf([1, np.real(p[-1])], denGc2)
Gc2 = (1/dcgain(Gc2))*Gc2
Hc2 = minreal( series(Gc2,H2), verbose=False )

# Obtençao da resposta para o maior sinal de referencia
_T,  Y2max, _X = forced_response(H2, t, r_max)
_T, Y2cmax, _X = forced_response(Hc2,t, r_max)

# Obtençao da resposta para o menor sinal de referencia
_T,  Y2min, _X = forced_response(H2, t, r_min)
_T, Y2cmin, _X = forced_response(Hc2,t, r_min)

# Funçao transferencia de U(s)/R(s)
Gu1 = minreal( C*Gc1/( kh1*(G*C + 1) ) )
Gu2 = minreal(   Gc2/( kh2*(G*C + 1) ) )


# Obtendo o sinal de controle
_T,  U1min, _X = forced_response(Gu1,t, r_min)
_T,  U1max, _X = forced_response(Gu1,t, r_max)

_T,  U2min, _X = forced_response(Gu2,t, r_min)
_T,  U2max, _X = forced_response(Gu2,t, r_max)


if min(U1min) < min(U1max):
    minU1 = min(U1min)
else:
    minU1 = min(U1max)

if max(U1min) > max(U1max):
    maxU1 = max(U1min)
else:
    maxU1 = max(U1max)

if maxU1 > (100-uop) or minU1 < -uop:
    print('VALOR MAX/MIN DE U1 ATINGIDO: %0.2f e %0.2f' %(maxU1,minU1))
else:
    print('SINAL DE CONTROLE EM SERIE OK!\n\t MAX: %0.2f e MIN: %0.2f' %(maxU1,minU1))

if min(U2min) < min(U2max):
    minU2 = min(U2min)
else:
    minU2 = min(U2max)

if max(U2min) > max(U2max):
    maxU2 = max(U2min)
else:
    maxU2 = max(U2max)

if maxU2 > (100-uop) or minU2 < -uop:
    print('VALOR MAX/MIN DE U2 ATINGIDO: %0.2f e %0.2f' %(maxU2,minU2))
else:
    print('SINAL DE CONTROLE NA REALIMENTAÇAO OK!\n\t MAX: %0.2f e MIN: %0.2f' %(maxU2,minU2))

# Plotar
plt.figure(1)
# Sem compensaçao
plt.plot(t,Y1min)
plt.plot(t,Y1max)

plt.plot(t,Y2min)
plt.plot(t,Y2max)

plt.xlabel('$t$ (s)')
plt.ylabel('$y$')
plt.xlim([0,2*ts])
plt.rc('legend', fontsize=12)
plt.title('Sem compensaçao')
plt.legend(['$H_1$ min', '$H_1$ max','$H_2$ min', '$H_2$ max'])
plt.grid(linestyle='--')


plt.figure(2)
# Com compensaçao

# Minimo
plt.subplot(2,2,1)
plt.plot(t,Y1cmin)
plt.plot(t,Y2cmin)
plt.xlabel('$t$ (s)')
plt.ylabel('$y(t)$')
plt.xlim([0,2*ts])
plt.grid(linestyle='--')
plt.rc('legend', fontsize=8)
plt.legend(['$y_1$ min', '$y_2$ min'])

# Sinal de controle
plt.subplot(2,2,3)
plt.plot(t,U1min, '#f0a500')
plt.plot(t,U2min,'#fe7171')
plt.xlabel('$t$ (s)')
plt.ylabel('$u(t) (\%)$')
plt.xlim([0,2*ts])
plt.grid(linestyle='--')
plt.rc('legend', fontsize=8)
plt.legend(['$u_1$ min', '$u_2$ min'])

# Maximo
plt.subplot(2,2,2)
plt.plot(t,Y1cmax)
plt.plot(t,Y2cmax)
plt.xlabel('$t$ (s)')
plt.ylabel('$y(t)$')
plt.xlim([0,2*ts])
plt.grid(linestyle='--')
plt.rc('legend', fontsize=8)
plt.legend(['$y_1$ max', '$y_2$ max'])


plt.subplot(2,2,4)
plt.plot(t,U1max, '#f0a500')
plt.plot(t,U2max,'#fe7171')
plt.xlabel('$t$ (s)')
plt.ylabel('$u(t) (\%)$')
plt.xlim([0,2*ts])
plt.grid(linestyle='--')
plt.rc('legend', fontsize=8)
plt.legend(['$u_1$ max', '$u_2$ max'])
plt.show()