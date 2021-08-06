#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:57:13 2021

@author: Bernardo Bresolini

Exemplo de geração de sinal PRBS de sequencia m.
Este código se basea na Seção 4.3.1, p. 195 do livro do AGUIRRE: Introdução à Identificação de Sistemas para gerar um Sinal PRBS de sequência m.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

b = 6  # Quantidade de bits
N = 2**b - 1 # Perído do sinal
m = 1  # Tamanho do intervalo
k = 2  # Quantidade de intervalos para o sinal

def PRBSm(b, N, m=1, Vmin=-1, Vmax=1, seed=12345):
    """
Gerador de sinal PRBS de sequência m, são dados o número de bits do registro de deslogamento `bits' e o tamanho da amostra `N'.

Parâmetros
----------
b : int
     Número de bits do registro de deslogamento.
N : int
    Tamanho do sinal gerado, isto é, quantidade de elementos presentes na saída da função.
m : int
    Fator multiplicador para o intervalo de dados, sendo usado para aumentar o intervalo entre bits.
seed : int
    Fonte que inicializaráo `BitGenerator`.

Retorna
-------
y : np.array
    Sinal PRBS de sequência m gerado.

Notas
-----
O gerador dos floats pseudoaleatórios é feita pelo Gerador PCG-64. Esta estrutura, PCG64, tem melhores propriedades estatísticas do que o MT19937, normalmente usado.

Para mais informações, consulte o módulo numpy.random().
    """

    # Cria o gerador de números aleatórios
    rng = np.random.Generator(np.random.PCG64(seed))

    y = np.zeros(N)       # Pré-alcação
    # Gera b números aleatórios entre 0 e 1 e transforma em True e False a depender do valor.
    x = rng.random(b)>0.5

    # Para 8 bits, deve ser feito a xor entre os bits 2, 3, 4 e 8.
    # Consulte a TAB, 4,1 do Aguirre, p. 197.
    if b == 8:
        # j = b - 4, k = b - 3, l = b - 2
        j, k, l = 4, 5, 6
        for i in range(1, int(N/m)):
            # Os intervalos de tamanho m, recebe o último bit de x
            y[m*(i-1):m*i] = x[-1]*np.ones(m)
            # Atualiza o primeiro bit de x
            # e desloca o restanto, perdendo o último.
            x = np.hstack((
                    x[-1] ^ x[b-j-1] ^ x[b-k-1] ^ x[b-l-1],
                    x[:b-1]))

    # Para as demais quantidades de bits
    else:
        # Normalmente, a XOR é feita o último e o penúltimo dígito
        j = 1

        # Exceti nos casos:
        if b == 5 or b == 11:
            j = 2
        elif b == 7 or b == 10:
            j = 3
        elif b == 9:
            j = 4

        for i in range(1, int(N/m)):
            # Os intervalos de tamanho m, recebe o último bit de x
            y[m*(i-1):m*i] = x[-1]*np.ones(m)
            # Atualiza o primeiro bit de x
            # e desloca o restanto, perdendo o último.
            x = np.hstack((x[-1] ^ x[b-j-1], x[:b-1]))


        ymap = (Vmax-Vmin)*y + Vmin

    return y, ymap, x

def FAC(y):
    """
Calcula a função de autocorrelação (FAC) amostral para o sinal.

O calculo é feito resolvendo
    \begin{equation}
        \hat{r}_y(k) = \frac{1}{N} \sum_{i=0}^{N-k} y(i) y(i+k),
        \qquad k = 0,\,1,\,\cdots,\,N-1
    \end{equation}
sendo N a quantidade de amostras.

Parâmetros
----------
y : array_like
     Sinal de saída cuja FAC amostral deseja ser calculada.

Retorna
-------
ry : np.array
    FAC amostral calculado.
    """
    y = np.asarray(y)

    # Quantidade de elementos em y
    N = len(y)

    ry = np.zeros(N) # Pré-alocação
    for k in range(N-1):
        for i in range(N-k):
            ry[k] = sum(y[:N-k]*y[k:N])/N

    return ry


t = np.arange(k*m*N)
y, ymap, x = PRBSm(b, k*m*N, m)

# Plotagem sinal
plt.step(t, ymap,c='tab:cyan',label='Mapeado')
plt.plot([63,63],[-1.1,1.1],'--', c='xkcd:pink red',linewidth=1)
plt.yticks([-1,-0.5,0,0.5,1])
plt.text(31.5,1.1,r"$N$", horizontalalignment="center",fontsize=12)
plt.text(94.5,1.1,r"$2N$", horizontalalignment="center",fontsize=12)

plt.xlabel('Tempo')
plt.ylabel('Saída')

# PLotagem FAC amostral
ry = FAC(ymap)
plt.figure(2)
plt.plot(np.hstack((-t[::-1],t)), np.hstack((ry[::-1],ry)),c='tab:cyan')
plt.xticks([-2*63, -1*63, 0, 1*63, 2*63])
plt.yticks([-1/len(ymap), 0.25, 0.5, 0.75, 1])
plt.grid(linestyle='--')


# Valores para a validação
nmin = list(ymap[:N]).count(-1.0)
nmax = list(ymap[:N]).count( 1.0)

print(f"\t       +V:\t +1",
      f"\t       -V:\t -1",
      f"\t        N:\t {N}",
      f"\tnum de -V:\t {nmin}",
      f"\tnum de +V:\t {nmax}",
      sep='\n')

