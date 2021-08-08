#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 19:20:39 2021

@author: Bernardo Bresolini

Biblioteca desenvolvida durante o TCC em Identificação de Sistemas.
Ela contém funções úteis para a implementação das técnicas de estimação
de parâmetros para séries temporais.

Histórico de revisões:
    Autor: Bernardo Bresolini
    Data: 30 de jun. 2021
    New: Criação com as funções `sim_ARX', `MQ' e `EMQ'.
"""

import numpy as np
import matplotlib.pyplot as plt

# Função que simula um dado modelo ARX com base nos polinômios A(q) e B(q)
def sim_ARX(polA,polB, u, nu, y0 = 0):
    """
Função usada para simular um modelo ARX sendo dados os polinômios A(q) e B(q), o sinal de controle e o ruído branco.

Para tanto, considerou-se o modelo ARX dado por
    $$ A(q) y(k) = B(q) u(k) + \nu(k) $$
sendo $A(q) = 1 - a_1 q^{-1} - \ldots - a_{n_y} q^{-n_y}$,
    $B(q) = b_1 q^{-1} + \ldots + b_{n_u} q^{-n_u}$.

Isolando y(k) segue
    $$y(k) = a_1 y(k-1) + \ldots + a_{n_y} y(k-n_y) +
             b_1 u(k-1) + \ldots + b_{n_u} u(k-n_u) + e(k)$$
em que $e(k)$ é o ruído filtrado.

Ou seja, para A(q) passe os valores na forma (1, a1, a2, ...)
              B(q) passe os valores na forma (b1, b2, b3, ...)

Parâmetros
----------
polA : array_like
    Coeficientes do polinômio A(q)
polB : array_like, ...
    Coeficientes do polinômio B(q)
u : array_like, ...
    Sinal de controle a ser aplicado ao modelo.
nu : array_like, ...
    Sinal do ruído de medição.

Retorna
-------
y : ndarray
    Saída do modelo para a entrada `u' e o ruído `nu'.
   """
   # Mudando para objetos np.array
    polA, polB = np.asarray(polA), np.asarray(polB)

    # Transforma o sinal em um vetor-linha
    # devido a implementação realizada, esta deve ser a forma
    u = u.reshape((max(u.shape)))

    # Lista para a iteração
    K = np.arange(len(u))
    # Pré-alocação
    y = np.zeros_like(u)
    y[0] = y0

    # Usado para obter o primeiro termo não nulo de ``polB''. Obtém o atraso puro
    #n = next((i for i, x in enumerate(polB) if x), None)

    # Vetores auxiliares para a equação a diferença
    yy, uu = y0*np.ones(len(polA)-1), np.zeros(len(polB))

    # Transforma o primeiro termo de ``polA'' em 1.
    polA = polA[1:]/polA[0]

    # Transforma o ruído na forma np.array
    nu = np.asarray(nu).reshape(u.shape)

    # Simulação
    for k in K[1:]:
        # Equação a diferença
        y[k] = sum(yy*polA) + sum(uu*polB) + nu[k]

        # Atualização dos valores
        uu = np.concatenate(([u[k]], uu[:-1])) # Atualiza y(k), y(k-1),
        yy = np.concatenate(([y[k]], yy[:-1])) # Atualiza u(k), y(k-1),

    # Transforma y como matriz-coluna
    y = np.asmatrix(y.reshape((max(y.shape), 1)))
    return y

def MQ(y, u, n=(1,1), k0=None):
    """
Estimador de modelo ARX pelo método dos mínimos quadrados pela pseudoinversa.

Sejam as amostras y(k) e u(k). O vetor regressores é
psi(k-1) = [y(k-1)  y(k-2) .. y(k-ny) | u(k-1)  u(k-2)  .. u(k-nu)]
de modo que
    y(k) = psi(k-1)*th
sendo `th' os coeficientes do modelo a serem determinados.

A resolução pela pseudoinversa é
    th = inv(psi.T@psi)@psi.T@y

Parâmetros
----------
y : array_like
    Vetor com os valores medidos da saída.
u : array_like
    Vetor com os valores medidos da entrada.
n : tuplet, list .. de 2 valores
    Ordem dos polinômios A(q) e B(q), sendo n = (ny, nu).
k0 : int ou None
    Ponto inicial para k. Caso seja None, é considerado como max(n)+1.
    Padrão é None.

Retorna
-------
psi : np.matrix
    Matriz com os vetores regressores.
th : np.matrix
    Matriz coluna com os coeficientes dos polinômios A(q) e B(q).
res : np.matrix
    Matriz coluna com os resíduos obtidos de res = y(k) - psi@th
    """
    # Caso não seja passado o ponto em que se deve começar
    # ele iniciará pelo maior valor entre ny e nu
    if k0 is None:
        k0 = max(n)
    else:
        assert k0 >= max(n)-1, 'k0 deve ser maior que max(n)-1!'

    # Tornando os vetores de cados em np.array
    y, u = np.asarray(y), np.asarray(u)

    # Transforma os sinais em um matrix-coluna
    # devido a implementação realizada, esta deve ser a forma
    y = y.reshape((max(y.shape),1))
    u = u.reshape(y.shape)

    yk = y[k0+1:]
    uk = u[k0+1:]

    # Matriz psi na forma psi = [ y(k-1) y(k-2) .. u(k-1) u(k-2)]
    # Começando por y(k-1), y(k-2)
    psi = y[k0:-1]
    for i in range(1, n[0]):
        psi = np.hstack((psi, y[k0-i:-i-1]))

    # Parte de i(k-1), u(k-2)
    psi = np.hstack((psi, u[k0:-1]))
    for i in range(1,n[1]):
        psi = np.hstack((psi, u[k0-i:-i-1]))

    # Transforma em np.matrix
    psi = np.asmatrix(psi)

    # Resolvendo pela pseudoinversa
    th = np.linalg.inv(psi.T@psi).dot(psi.T).dot(yk)

    # Calculando os resíduos
    res = yk - psi@th

    # retornando
    return psi, th, res

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
