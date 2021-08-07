#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:53:03 2021

@author: Bernardo Bresolini

Exemplo estimação de parêmetros de um modelo ARX.
Exemplo 5.5.2. Estimaçãao de parâmetros de um modelo ARX, p. 243 do livro do AGUIRRE: Introdução à Identificação de Sistemas.

"""
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Valores medidos da saída (y) e da entrada (u)
y = np.array([12.2, 11.8, 11.6, 11.6, 11.8, 12.2, 13.0, 14.4, 16.2, 15.8])
u = np.array([2.50, 2.50, 2.50, 2.50, 2.50, 2.23, 2.20, 2.20, 2.21, 2.20])

# Farei na forma intuitiva
# psi = [ y(k-1)  y(k-2)  u(k-1)  u(k-2)  u(k-3) ]

# No Aguirre temos
# psi = [ y(k-1)  y(k-2)  u(k-3)  u(k-1)  u(k-2) ]
# Contudo isso afeta somente na ordem de th, sendo th da forma
# th = [ a1 a2 b3 b1 b2]

# Formato dos polinômios, p = (ny, nu)
p = (2,3)

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

    # Tornando os vetores de dados em np.array
    y, u = np.asarray(y), np.asarray(u)

    # Transforma os sinais em um matrix-coluna
    # devido a implementação realizada, esta deve ser a forma
    y = y.reshape((max(y.shape),1))
    u = u.reshape(y.shape)

    yk = y[k0+1:]
    uk = u[k0+1:]

    # Matriz psi na forma psi = [ y(k-1) y(k-2) .. u(k-1) u(k-2) .. ]
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

# Identificando os parâmetros th
psi, th, res = ident_ARX(y, u, p)

print('Matriz ψ:', psi, sep='\n')
print('')
print('Parâmetros θ:', th, sep='\n')
print('')
print('Resíduos ξ:', res, sep='\n')

