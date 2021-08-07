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

def ident_ARX(y, u, p):
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
p : tuplet, list .. de 2 valores
    Ordem dos polinômios A(q) e B(q), sendo p = (ny, nu).

Retorna
-------
psi : np.matrix
    Matriz com os vetores regressores.
th : np.matrix
    Matriz coluna com os coeficientes dos polinômios A(q) e B(q).
res : np.matrix
    Matriz coluna com os resíduos obtidos de res = y(k) - psi@th
    """

    # Tornando os vetores de cados em np.array
    y, u = np.asarray(y), np.asarray(u)

    # Valores de y(k) em vetor coluna
    yk = np.matrix(y[max(p):]).T

    # Variáveis auxiliares
    ny, nu = p

    # Agora iniciaremos o vetor `psi'
    # Para isso, vamos usar um `for' em que a cada iteração `k' ele lê
    # os valores: y(k-1), y(k-2), ...
    #             u(k-1), u(k-2), ...
    # e salva na matriz `psi'.
    # Contudo na primeira iteração `k', fazemos com o `if'
    # devido a como é implementado o acesso aos vetores em NumPy.

    if ny > nu:
        psi = np.hstack((y[ny-1::-1], u[ny-1:ny-nu-1:-1]))
    elif ny < nu:
        psi = np.hstack((y[nu-1:nu-ny-1:-1], u[nu-1::-1]))
    else:
        psi = np.hstack((y[ny-1::-1], u[nu-1::-1]))

    # Para cada iteração `k', devemos  obter os valores:
    # k - 1, k - 2, ... k - ny   (do vetor y).
    # Logo estamos lendo o vetor no sentido inverso, por isso o `::-1'.
    # Ainda começamos em k - 1 e terminamos em k - ny.
    # Em Python isso equivale a terminar em k - ny - 1, já que o último digito não é lido.
    for k in range(max(p)+1, len(y)):
        aux = np.hstack((y[k-1:k-ny-1:-1], u[k-1:k-nu-1:-1]))
        psi = np.vstack((psi, aux))

    # Transforma `psi' em np.matrix
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

