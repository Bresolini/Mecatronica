"""
Created on Sun Jun 27 19:20:39 2021

@author: Bernardo Bresolini

Exemplo de estimador estendido de mínimos quadrados (EMQ) para obtenção de um modelo ARMAX.
Este código se basea-se do "Exemplo 7.2.1. Polarização devida a ruído MA --- análise" até o "Exemplo 7.2.5. Resíduos como estimativa do ruído"
do livro do AGUIRRE: Introdução à Identificação de Sistemas. p. 292-299.

Esclarecimentos:
O arquivo armax.py (que contém as funções usadas) deve estar na mesma pasta deste arquivo.
O arquivo ruido.mat (que contém o vetor u e nu da simulação do código ``bias31.m`` do Aguirre) deve estar na mesma pasta deste arquivo.
Vale ressaltar que o arquivo .mat quando importado com scipy.io.loadmat deve ter sido salvo no MATLAB com o flag -v7. Exemplo
    >>>> save('ruido.mat', '-v7') 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import armax

a, b = 0.7, 0.5

data = loadmat('ruido.mat')

u, nu = data["u"], data["nu"]
u, nu = np.asarray(u), np.asarray(nu)

# Pré-alocação
e = np.zeros_like(nu)

# Ruído colorido
for k in range(1, len(nu)):
    e[k] = 0.8*nu[k-1] + nu[k]

# Obtém a saída do modelo
# y(k) = a y(k-1) + b u(k-1) + e(k)
# Note que como o ruído é colorido, a saída não é precisamente
# de um modelo ARX, mas sim ARMAX.
y = armax.sim_ARX([1,a], [b], u, e)

# Teste com vários valores de ne
psiE, thE, xiE = armax.EMQ(y, u, (1,1,2))
