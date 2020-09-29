#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:42:09 2020

@author: Lucas Silva de Oliveira

Edited on Tue Sep 01 10:44:59 2020
@author: Bernardo Bresolini

Esse código simula a dinâmica em malha aberta de um tanque
"""

import numpy as np #Importando a biblioteca numpy - para ambiente matemático com vetores
import matplotlib.pyplot as plt #importando a biblioteca matplotlib - para plotagem
plt.close('all') #fechando todas as janelas com figuras

# Funçao degrau
def degrau(tstep,tf,dt,amp=1,t0=0):
    """ Funçao que retorna um vetor correspondente a um degrau em 'tsep'
        com inicio em 't0' (igual a 0, normalmente) e vai ate 'tf'
        com espaçamento de 'dt'. Amplitude eh de 'amp'.

    Parameters:
    tstep (float): Tempo no qual iniciara o degrau
    tf    (float): Tempo final do vetor
    dt    (float): Intervalo, em segundos, dos valores (prefira valores de base 2).
    amp   (float): Amplitude do sinal, normalmente eh 1;
    t0    (float): Tempo inicial do vetor, normalmente eh 0.

    Returns:
    vetor[float]: Returnando o vetor u do degrau

   """
    t=np.arange(t0,tf+dt,dt)
    u=amp*np.heaviside(t-tstep,1)
    return u

# Definição de algumas constantes do processo.

dt = 0.25 # (s) período de amostragem
tf = 1000 # (s) duração do teste
t = np.arange(0,tf+dt,dt) #vetor de tempo do experimento

# Defina o ponto de operaçao (0 cm a 70 cm)
hop = 36

# Calculo da entrada uop necessaria para alcançar hop
""" O sistema escolhido eh dado pela eq. 
    $$ \dot{h}(t) = \frac{ 16,998 u(t) - 12,714 h(t) - 462,893}{3019}$$"""

uop = (462.893+12.741*hop)/16.998   # valor entre 0 a 100


amp = uop #amplitude do sinal de controle, esse deve ser um valor pertencente ao intervalo 0<= amp <= 100
u = amp*np.ones(len(t)) #Vetor do sinal de controle

h = np.empty(len(t)) #pre-alocação na memória do estado do sistema - h=altura da coluna de líquido em (cm). Atenção:
# o tanque possui limitação física, desse modo a altura deve permanecer ao intervalo: 0 <= h <= 70 cm.
h.fill(np.nan)
h[0] = 0 #Condição inicial do sistema

for i in range(len(t)-1): #simulação do sistema em malha aberta
    h[i+1] = h[i] + dt*(16.998*u[i] + 354.781 - (12.741*h[i]+817.874))/3019
    
plt.figure(1)
plt.subplot(2,1,1) #Plotando a saída do sistema
plt.plot(t,h,'b')
#plt.plot(t,ref,'k--')
plt.xlabel('Tempo (s)')
plt.ylabel('h (cm)')
plt.xlim((0,tf))
plt.ylim((0,40))
plt.yticks([0,10,20,30,hop,40,50])
plt.grid(linestyle='--')

plt.subplot(2,1,2) #Plotando o sinal de controle
line=plt.plot(t,u,'b')
plt.xlabel('Tempo (s)')
plt.ylabel('u (%)')
plt.xlim((0,tf))
plt.ylim((0,100))
plt.yticks([0,20,40,uop,60,80,100])
plt.grid(linestyle='--')
plt.show()


# Variaçao maxima do sinal de controle
minY = 0
maxY = 70
max_dy = 0.07*(maxY-minY)
max_du = 12.741*max_dy/16.998

# Sinal de controle variando em 7% do fundo de escala
ts = 1000   # Inicialmente definindo o tempo de acomodaçao

uu = uop + degrau(ts,10*ts,dt,0.25*max_du) + degrau(2*ts,10*ts,dt,0.25*max_du) \
+ degrau(3*ts,10*ts,dt,0.25*max_du) + degrau(4*ts,10*ts,dt,0.25*max_du) \
+ degrau(5*ts,10*ts,dt,-1.25*max_du) + degrau(6*ts,10*ts,dt,-0.25*max_du) \
+ degrau(7*ts,10*ts,dt,-0.25*max_du) + degrau(8*ts,10*ts,dt,-0.25*max_du) \
+ degrau(9*ts,10*ts,dt,max_du) 

# Vetor de referencia
r  = ( 16.998*uu - 462.893 )/(12.741)   # valor entre 0 a 70


# Limites inferiores e superiores do sinal de controle
u_inf = (uop-max_du)*np.ones(len(uu))
u_sup = (uop+max_du)*np.ones(len(uu))

# Vetor do tempo
tt = np.arange(0,10*ts+dt,dt)



h = np.empty(len(tt)) #pre-alocação na memória do estado do sistema
h.fill(np.nan)
h[0] = 0 #Condição inicial do sistema
for i in range(len(tt)-1): #simulação do sistema em malha aberta
    h[i+1] = h[i] + dt*(16.998*uu[i] + 354.781 - (12.741*h[i]+817.874))/3019


plt.figure(2)
plt.subplot(2,1,1) #Plotando a saida do sistema
plt.plot(tt,r)     #Referencia
plt.plot(tt,h)     #Saida
#Configuraçoes de plotagem
plt.xlabel('Tempo (s)')
plt.ylabel('h (cm)')
plt.xlim((0,10*ts))
plt.ylim((0,45))
plt.yticks([0,10,20,30,40])
plt.grid(linestyle='--')

plt.subplot(2,1,2) #Plotando o sinal de controle
plt.plot(tt,u_inf,'k--') #Limite inferior
plt.plot(tt,u_sup,'k--') #Limite superior
plt.plot(tt,uu)    #Sinal de controle
#Configuraçoes de plotagem
plt.xlabel('Tempo (s)')
plt.ylabel('u (%)')
plt.xlim((0,10*ts))
plt.ylim((50,60))
plt.yticks([50,u_inf[0],uop,55,u_sup[0],60])
plt.grid(linestyle='--')
plt.show()

"""
amp=3
tstep, t0,tf,dt = 3, 0,100,0.25
t = np.arange(t0,tf+dt,dt)
u1 = degrau(tstep,tf,dt)
u2 = degrau(5,tf,dt,2)
plt.plot(t,u1)
plt.plot(t,u2)
plt.plot(t,u1+u2)
"""