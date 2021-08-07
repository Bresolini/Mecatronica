"""
Created on Sun Jun 27 19:20:39 2021

@author: Bernardo Bresolini

Exemplo de estimador estendido de mínimos quadrados (EMQ) para obtenção de um modelo ARMAX.
Este código se basea-se do "Exemplo 7.2.1. Polarização devida a ruído MA --- análise" até o "Exemplo 7.2.5. Resíduos como estimativa do ruído"
do livro do AGUIRRE: Introdução à Identificação de Sistemas. p. 292-299.
"""

import numpy as np
import matplotlib.pyplot as plt

def EMQ(y, u, n=(1,1,2), imax=10):
    """
Implementa o estimador estendido de mínimos quadrados (EMQ) para obtenção de modelo ARMAX. O método é pseudolinear e utiliza os resíduos como aproximação para o ruído do sinal de saída.

Sejam as amostras y(k) e u(k). O vetor regressores é
psiE(k-1) = [y(k-1) .. y(k-ny) | u(k-1) .. u(k-nu) | xi(k-1) .. xi(k-ne)]
de modo que
    y(k) = psiE(k-1)*thE
sendo `thE' os coeficientes do modelo a serem determinados.

A resolução pela pseudoinversa é a iteração de
    thE = inv(psiE.T@psiE)@psi.T@yk

Parâmetros
----------
y : array_like
    Vetor com os valores medidos da saída.
u : array_like
    Vetor com os valores medidos da entrada.
n : tuplet, list .. de 3 valores
    Ordem dos polinômios em y, u e xi, respectivamente. n = (ny, nu, ne).
imax : int
    Quantidade de vezes para a iteração.
    Padrão é 10.

Retorna
-------
psiE : np.matrix
    Matriz com os vetores regressores extendido.
thE : np.matrix
    Matriz coluna com os coeficientes do modelo.
xiE : np.matrix
    Matriz com os resíduos obtidos de xiE = yk - psiE@thE

Notas
-----
O algoritmo não testa a convergência dos dados.

    """
    # Tornando os vetores de cados em np.array
    y, u = np.asarray(y), np.asarray(u)

    # Transforma os sinais em um matrix-coluna
    # devido a implementação realizada, esta deve ser a forma
    y = y.reshape((max(y.shape),1))
    u = u.reshape(y.shape)

    # Maior ordem do sistema será usada como ponto de partida para y(k)
    k0 = max(n)

    # Valor de ne
    ne = n[2]

    # Resolve para a estrutura ARX
    # Nota, como em Python os arrays começam de 0, deve ser subtraído 1
    psi, th, res = MQ(y, u, n[0:2], k0-1)
    # y[k]
    yk = y[k0:]

    # Se ne == 1, não é preciso deslocar os resíduos
    if n[2] == 1:
        xiE = res
    else:
    # Desloca os resíduos em ne - 1 e insere zeros no lugar
        xiE = np.vstack((np.zeros((ne-1, 1)), res[:-ne+1]))

    # Resolução iterativa, admitindo que os resíduos são aproximações
    # para o ruído, Assim é feita iterações até convergir ξ = ν.
    # O método não é convexo e pode ter mínimos locais.
    for k in range(imax):
        psiE = psi[ne:]
        # Faz psiE = [ psi | xi(k-1)  xi(k-2) .. x(k-ne)]
        for i in range(1,ne+1):
            psiE = np.hstack((psiE, xiE[ne-i:-i]))

        # Transofmra em np.matrix
        psiE = np.asmatrix(psiE)

        # Resolve para os coeficientes
        thE = np.linalg.inv(psiE.T@psiE).dot(psiE.T).dot(yk[ne:])
        # Recalcula os resíduos
        xiE = np.asmatrix(np.vstack((np.zeros((ne,1)), yk[ne:] - psiE@thE)))

    return psiE, thE, xiE
  
a, b = 0.7, 0.5

# Cria o gerador de números aleatórios
g = np.random.Generator(np.random.PCG64(1337))
h = np.random.Generator(np.random.PCG64(1338))

# Cria o sinal de controle como um sinal de distribuição normal
# Mostrando que o método é válido para qualquer tipo de entrada aplicada
u  = g.normal(size=(502,1))
# Cria vetor de ruído branco com distribuição normal
nu = 0.2*h.normal(loc=0, scale=1, size=(502,1))

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
 


