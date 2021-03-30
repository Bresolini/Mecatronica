#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bernardo Bresolini

Esta biblioteca reúne várias funções que foram construídas durante os trabalhos das disciplinas do eixo de Controle do curso em Eng. Mecatrônica.

As funções implementadas se mostraram bastante úteis durante os trabalhos.

Lista de funções
----------------
# Funções utilitárias


stepinfo(t, y, Print=True)
    Calcula os parâmetros da resposta temporal a um degrau para um sistema estável.

param(ts, OS, a=4, n=2):
    Cálculo do(s) polo(s) desejados com base nos critérios de desempenho.

zw(Mp, tp):
    Calcula os parâmetros de ζ e ωn com base no percentual máximo de overshoot e o tempo em que o pico do sinal ocorre. O sistema considerado é um sistema de segunda ordem.

sylv(num,den):
    Determina a matriz de Sylvester com o numerador e denominador dados.

polos_nondom(p1, n, pond=1):
    Retorna os polos não dominantes de um sistema de ordem ``n``.

makeSteps(lists, T, dt):
    Função que criar uma sequência de degraus uniformes a partir de uma lista com as amplitudes, o espaçamento e o tempo de cada degrau.

def getKTL(t, y, A=1, Plot=True):
    Função usada para determinar os parêmtros K, T e L para a sintonia de controladores PID pelo método Ziegler-Nichols. Os 3 parâmetros são determinados para a resposta a um degrau no sistema.

# Funções de plotagem


PlotStep(sys, t, nfig=1):
    Função para plotar a resposta temporal de um sistema linear estável para uma entrada de degrau. Além disso, são marcados no gráfico o tempo de acomodação, tempo de pico e pico do sinal.

PlotData(t,
         y,
         r,
         u,
         discrete=False,
         ylabels=None,
         ulabels=None,
         loc='lower right'):
    Função para plotar os dados de uma simulação marcando tss e OS.

Bode(G, wlim=None,Hz=True, tol=1e-1):
    Plota o diagrama de Bode de um sistema G e calcula os valores da banda de passagem, ganho em baixa frequência, ganho máximo e freq. de pico.

_genTxt(pol, x, dec):
    Função auxiliar para a função zpk, Gera o texto do numerador ou denominador com base nos zeros ou polos passados.

zpk(G, dec=4):
    Printa os zeros, polos e ganho de uma função transferência.

#  Funções de simulação


sim_cont(G, t0, Ts, qnt, Ue, x0):
    Função auxiliar usada para fazer a simulação de um sistema contínuo durante o tempo entre a amostragem do sistema discreto.

simulaCzGs(G,
           C,
           n,
           r=None,
           x0=0,
           ulim=None,
           qnt=20):
    Simula a implementação do controlador discreto C(z) ligado em série com o processo contínuo G(s) e com realimentação unitária.

update_uk(C, ek, _u, _e):
    Função auxiliar que implementa a equação a diferença do controlador e retorna o valor do sinal de controle u[k] considerando o controlador em série com o processo e a realimentação unitária.

# Construir PID


PID(K, Ti, Td, N=None):
    Cria um controlador PID com base nos parâmetros K, Ti, Td. É possível utilizar um filtro N no derivativo.

PIDz(K, Ti, Td, Ts):
    Gera o controlador PID discretizado com base nos parâmetros K, Ti, Td e Ts.

discretePID(C, Ts):
    Discretiza um controlador PID C(s) com uma taxa de amostragem Ts.

# Projetar PID


PID_Pol(G, D, N=None):
    Projeta um controlador PID utilizando o método polinomial.

PID_LGR(G, p, zPI=-0.5, Print=True, xlim=None):
    Projeta um controlador PID pelo método LGR.

sintoniaPID(param, met='ZN',C='PID', modo=None):
    Sintoniza um controlador PID, PI ou P com base nos parâmetros do modelo identificado e no método pedido.

# Funções em desenvolvimento

latexTable(t, y1, y2):
    Gerar o texto em LaTeX dos critérios de desempenho de duas respostas

find_nearest(array, value):
    Função usada para achar o índice em que um valor é o mais próximo possível de um array passado.


Última edição
-------------
30/03/2021.
"""
import numpy as np
from numpy.linalg import inv
import control as ct
import matplotlib.pyplot as plt


# Funções de utilitárias
def stepinfo(t, y, Print=True):
    """
    Calcula os parâmetros da resposta temporal a um degrau para um sistema estável.

    Considera-se:
    • Tempo de acomodação (``tss``)
    \t Tempo que leva o erro |``y(t)`` - ``yfinal``| variar somente em 2% de ``yfinal``.
    • Overshoot (``OS``)
    \t Percentual de overshoot/sobressinal relativo a ``yfinal``.
    • Undershoot (``US``)
    \t Percentual de underrshoot/subsinal relativo a ``yfinal``.
    • Tempo de subida (``tr``)
    \t Tempoo que leva para a resposta subir de 10% a 90% da resposta de estado estacionário.
    • Pico (``peak``)
    \t Valor máximo absoluto de ``y(t)``.
    • Tempo de Pico (``tpeak``)ts
    \t Tempo em que o pico do sinal é atingido.

    Usou-se como referência a função stepinfo do MATLAB, em que se pode
    obter mais detalhes em
    https://www.mathworks.com/help/control/ref/stepinfo.html.

    Nota: A função não interpola os valores e portanto, sua precisão está associada ao intervalo entre os valores.

    Parâmetros
    ----------
    t : array
        Tempo da simulação.
    y : array
        Resposta temporal.
    Print : bool
        Se Print=True, printa os resultados na saída. Caso contrário, não exibe os parâmetros da resposta.
        Padrão é True.

    Retorna
    -------
    tss : float
        Tempo de acomodação.
    OS : float
        Percentual de overshoot.
    US : float
        Percentual de undershoot.
    tr : float
        Tempo de subida.
    peak : float
        Pico absoluto da resposta temporal.
    tpeak : float
        Tempo em que ocorre o pico.
    """
    t0, yf = t[0], y[-1]
    n = len(y)
    if yf < 0:
        tpeak = t[np.argmin(y)]
        peak  = np.min(y)
    else:
        tpeak = t[np.argmax(y)]
        peak  = np.max(y)
    OS = 100*(peak/yf - 1)
    if min(y) < y[0]:
        US = -100*min(y)/yf
    else:
        US = 0
    tr = t[next(i for i in range(0, n-1) if abs(y[i]) > abs(0.9*yf))] \
         - t[next(i for i in range(0, n-1) if abs(y[i]) > abs(0.1*yf))]
    tss = t[next(n-i for i in range(2,n-1) \
          if abs(y[-i]-yf) > abs(0.02*yf))+1] - t0
    if Print:
        print(f'\t     Tempo de acomodação: {tss}')
        print(f'\t Percentual de overshoot: {OS}')
        print(f'\tPercentual de undershoot: {US}')
        print(f'\t         Tempo de subida: {tr}')
        print(f'\t           Pico do sinal: {peak}')
        print(f'\t           Tempo de pico: {tpeak}')
    return tss, OS, US, tr, peak, tpeak

def param(ts, OS, a=4, n=2):
    """
    Cálculo do(s) polo(s) desejados com base nos critérios de desempenho.

    Com base no critério de desempenho passado, são calculados os
    parâmetros e os polos do modelo de primeira ou segunda ordem
    (subamortecido).

    Parâmetros
    ----------
    ts : float
        Tempo de acomodação desejado.
    OS : float
        Percentual de overshoot desejado (em %/100). Para sistemas
        de primeira ordem, esse valor é desconsiderado.
    a : float
        Constante usada para o cálculo caso se deseje aumentar a
        precisão do cálculo. Para sistemas de primeira ordem, seu
        valor é de -ln(0,02). Em sistemas de segunda ordem, deve-se
        consultar o ábaco do OGATA, 5 ed., Cap. 5, p. 158.
        Padrão é 4.
    n : {1, 2}
        Ordem do sistema. Caso seja 1, será considerado um sistema de
        primeira ordem. Se for 2, será considerado um sistema de
        segunda ordem com polos conjugados complexos.

    Retorna
    -------
    Caso ``n`` = 1
    ==========
    tau : float
        Constante de tempo (τ) do sistema de primeira ordem.
    p1 : float
        Polo desejado do sistema.
    ==========
    Caso ``n`` == 2
    ==========
    z : float (0 < z < 1)
        Coeficiente de amortecimento (ζ).
    w : float
        Frequência natural do sistema (ωn).
    p1 : complex
        Polo desejado com parte imaginária positiva.
    p2 : complex
        Polo desejado com parte imaginária negativa.
    """
    assert n in [1, 2], 'Escolha uma ordem válida: 1 ou 2.'
    if n == 1:
        tau = ts/a
        p1  = -1/tau
        return tau, p1
    else:
        z = -np.log(OS)/np.sqrt(np.log(OS)**2 + np.pi**2)
        w = a/(ts*z)
        p1 = np.complex(-a/ts, w*np.sqrt(1-z**2))
        p = np.array([p1,np.conjugate(p1)])
        return z, w, p

def zw(Mp, tp):
    """
    Calcula os parâmetros de ζ e ωn com base no percentual máximo de overshoot e o tempo em que o pico do sinal ocorre. O sistema considerado é um sistema de segunda ordem.

    Parâmetros
    ----------
    Mp : float
        Percentual de overshoot/sobressinal.
    tp : float
        Tempo em que ocorre o pico do sinal.

    Retorna
    -------
    z : float
        Coeficiente de amortecimento (ζ).
    w : float
        Frequência natural (ωn).
    p : np.array
        Polos do sistema.
    """
    z = -np.log(Mp)/np.sqrt(pow(np.log(Mp),2)+pow(np.pi, 2))
    w = np.pi/( tp*np.sqrt(1-pow(z,2)) )

    p1 = -z*w + 1j*w*np.sqrt(1-pow(z,2))
    p = np.array([p1,np.conjugate(p1)])
    return z, w, p

def sylv(num,den):
    """
    Determina a matriz de Sylvester com o numerador e denominador dados.

    Parâmetros
    ----------
    num : array, ndarray, vetor
        Numerador do processo.
    den : array, ndarry, vetor
        Denominador do processo.

    Retorna
    -------
    S : ndarray (Matriz)
        Matriz de Sylvester.
    """
    n, m = len(num)-1, len(den)-1
    N  = max([n,m])
    S = np.zeros( [ 2*N, 2*N ] )
    for i in range(0,N):
        S[i][i:m+1+i] = den[::-1]
        S[i+N][i:n+1+i] = num[::-1]

    S = S.transpose()
    return S

def polos_nondom(p1, n, pond=1):
    """
    Retorna os polos não dominantes de um sistema de ordem ``n``.

    Os polos não dominantes devem estar localizados uma década abaixo
    do polo dominantes mais rápido.

    Parâmetros
    ----------
    p1 : float
        Polo dominante com a parte real mais negativa.
    n : int
        Ordem do sistema.
    pond : float
        Espaçamento linear entre os valores dos polos.
        Padrão é 1.

    Retorna
    -------
    p : array
        Polos não dominantes do sistema.
    """

    pond = abs(pond)
    p = 10*np.real(p1) - pond*np.arange(0, n-2)
    return p

def makeSteps(lists, T, dt):
    """
    Função que criar uma sequência de degraus uniformes a partir de uma lista com as amplitudes, o espaçamento e o tempo de cada degrau.

    Parâmetros
    ----------
    lists : tuple, list, array
        Lista de amplitudes desejadas em cada degrau.
    T : int, float
        Tempo final de cada degrau.
    dt : int, float
        Espaçamento de cada valor.

    Retorna
    -------
    t : ndarray
        Vetor com o tempo de simulação.
    u : ndarray
        Vetor com a sequencia de degrau.
    """
    n, m = len(lists), int(T/dt) # n° de steps, qnt de valores
    t = np.arange(0, n*T, dt)    # Tempo de simulação

    u = np.zeros_like(t) # Pré-alocação
    for i in range(n):
        u[i*m:(i+1)*m] = lists[i] # Seq. stesp
    u[-1] = lists[n-1] # Atualiza último valor

    return t, u

def getKTL(t, y, A=1, Plot=True):
    """
    Função usada para determinar os parêmtros K, T e L para a sintonia de controladores PID pelo método Ziegler-Nichols. Os 3 parâmetros são determinados para a resposta a um degrau no sistema.

    Parâmetros
    ----------
    t : np.array
        Tempo da coleta dos dados. O vetor deve ser equispaçado.
    y : np.array
        Resposta do sistema para uma a entrada em degrau de amplitude A.
    A : float, int
        Amplitude do degrau aplicado.
    Plot : bool
        Se for True, plotará a reta identificada e os pontos de interesse.
    """
    dt = t[1]-t[0]
    K = y[-1]/A - y[0]

    #P = 0.63*K
    dy = np.diff(y)
    i  = np.argmax(dy)

    a1 = max(dy)/dt
    a0 = y[i+1] - a1*t[i+1]

    # O atraso não pode ser negativo
    t1 = -a0/a1 if -a0/a1 > 0 else 0
    #B = find_nearest(y, P)
    t2 = (y[-1] - a0)/a1

    L = t1
    T = t2-t1

    if Plot:
        plt.plot(t, a1*t+a0, linewidth=1)
        plt.plot([t2, t2], [K*A, 0], 'k--', linewidth=1)
        r = K*A*np.ones_like(y)
        plt.plot(t, r, 'k--', linewidth=1)
        plt.plot(t, y)


        plt.scatter([t[i+1], t1, t2], [y[i+1], 0, K*A], c='#DC1C13')

        plt.grid(linestyle='--')
        plt.xlim(t[0], t[-1])
        plt.ylim(min(y), 1.2*max(y))
        plt.xlabel('tempo (s)')
        plt.ylabel('Amplitude')

    return K, T, L

# Funções de plotagem
def PlotStep(sys, t, nfig=1):
    """
    Função auxiliar para plotar a resposta temporal de um sistema linear estável para uma entrada de degrau. Além disso, são marcados no gráfico o tempo de acomodação, tempo de pico e pico do sinal.

    Parâmetros
    ----------
    sys : control.tf ou control.ss
        Sistema cuja resposta será plotada.
    t : np.array, list of float
        Tempo de simulação.
        Nota: para sistemas discretos, certifique que o espaço entre os valores do tempo sejam igual a taxa de amostragem do sistema.
    nfig : int
        Número da figura em que o plot será mostrado.
        Padrão é 1.

    Retorna
    -------
    y : np.array
        Resposta temporal para a entrada em degrau
    """

    t, y = ct.step_response(sys, t)
    tss, OS, US, tr, peak, tpeak = stepinfo(t, y)
    plt.figure(nfig)
    if ct.isctime(sys):
        plt.plot(t, y)
    else:
        t = np.concatenate(([0], t))
        y = np.concatenate(([0], y))

        plt.step(t, y, where='post')

    plt.xlim((t[0],t[-1]))
    #plt.ylim([0, 1.05*np.max(y)])
    # Tempo de acomodação
    plt.plot([tss, tss],
             [0, y[np.where(t==tss)][0]],
              '-.',
              linewidth=1,
              c='#43464b')
    plt.plot([tss],
             [y[np.where(t==tss)][0]],
             '.',
             markersize=8,
             c='#43464b')
    # Pico
    plt.plot([tpeak, tpeak],
             [0, peak],
              '-.',
              linewidth=1,
              c='#43464b')
    plt.plot([0, tpeak],
             [peak, peak],
              '-.',
              linewidth=1,
              c='#43464b')
    plt.plot([tpeak],
             [peak],
             '.',
             markersize=8,
             c='#43464b')
    plt.grid(linestyle='--')

    return y

def PlotData(t,
             y,
             r,
             u,
             discrete=False,
             ylabels=None,
             ulabels=None,
             loc='lower right'):
    """
    Função para plotar os dados de uma simulação.

    É criado 2 subplots, no primeiro será mostrado a resposta temporal de y comparada com o sinal de referência passado r. No segundo é mostrado o sinal de controle u.

    Parâmetros
    ----------
    t : np.array
        Tempo de simulação.
    y : np.array
        Resposta do sistema.
    r : np.array
        Sinal de referência passado.
    u : np.array
        Sinal de controle enviado a planta.
    discrete : bool
        Se for True, o sistema será plotado com plt.step. Caso contrário, será plotado com plt.plot.
        Padrão é False
    ylabels : list of strigs
        Textos usados na legenda das respostas. Se None, não haverá textos para as legendas.
        Padrão é None.
    ulabels : list of strings
        Textos usados na legenda dos sinais de controle. Se None, não haverá textos para as legendas.
        Padrão é None.
    loc : string
        Localização das legendas. Veja matplotlib.pyplot.legend() para mais informações.
        Padrão é 'lower right'.
    """
    # Transformar u e y em np.array
    u, y = np.asarray((u, y))
    n, *m = y.shape # forma inicial de y
    # Se y for um vetor e não uma matriz, transforma-o numa matriz linha
    y = y.reshape((1, n)) if m==[] else y
    u = u.reshape(y.shape) # Faz o mesmo para u
    n, _ = y.shape # Forma atual de y
    # Altera ylabels e ulabels para serem iteráveis, acessados por:
    # ylabels[i]
    ylabels = n*[None] if ylabels is None else ylabels
    ulabels = n*[None] if ulabels is None else ulabels

    # Pré-aloca os critérios de desempenho
    tss, OS, US, tr, peak, tpeak = np.zeros((6,n))
    # Cria a figura
    fig, ax = plt.subplots(2,1,sharex=True)
    # Configuração do plot
    ax[0].grid(linestyle='--')
    ax[0].set_ylabel('$y(t)$')
    ax[0].set_xlim([min(t), max(t)])

    ax[1].grid(linestyle='--')
    ax[1].set_xlabel('$t$ (s)')
    ax[1].set_ylabel('$u(t)$')

    # Iteração pela quantidade de respostas
    for i in range(n):
        # Critérios de desempenho
        tss[i], OS[i], US[i], tr[i], peak[i], tpeak[i] = stepinfo(t, y[i])
        # Se for discreto, use step e não plot
        if discrete:
            ax[0].step(t, y[i], label = ylabels[i])
            ax[1].step(t, u[i], label = ulabels[i])
        else:
            ax[0].plot(t, y[i], label = ylabels[i])
            ax[1].plot(t, u[i], label = ulabels[i])

        # Plot os tracejados e pontos dos critérios de desempenho
        # Tempo de acomodação
        ax[0].plot([tss[i], tss[i]],
                 [np.min(y), y[i][np.where(t==tss[i])][0]],
                  '-.',
                  linewidth=1,
                  c='#43464b')
        ax[0].plot([tss[i]],
                 [y[i][np.where(t==tss[i])][0]],
                 '.',
                 markersize=8,
                 c='#43464b')
        # Pico
        ax[0].plot([tpeak[i], tpeak[i]],
                 [np.min(y), peak[i]],
                  '-.',
                  linewidth=1,
                  c='#43464b')
        ax[0].plot([min(t), tpeak[i]],
                 [peak[i], peak[i]],
                  '-.',
                  linewidth=1,
                  c='#43464b')
        ax[0].plot([tpeak[i]],
                 [peak[i]],
                 '.',
                 markersize=8,
                 c='#43464b')

    # Plota a referência
    if discrete:
        ax[0].step(t, r, label = '$r(t)$')
    else:
        ax[0].plot(t, r, label = '$r(t)$')

    # Insere a legenda
    ax[0].legend(loc=loc)
    ax[1].legend(loc=loc)

def Bode(G, wlim=None,Hz=True, tol=1e-1):
    """
    Plota o diagrama de Bode de um sistema G e calcula os valores da banda de passagem, ganho em baixa frequência, ganho máximo e freq. de pico.

    Parâmetros
    ----------
    G : control.tf
        Função transferência a ser analisada.
    wlim : tuple, list .. de dois valores
        Limites do vetor de frequência gerados. Se Hz=True, os limites são em Hz. Caso contrário são em rad/s.
        Padrão é None, correspondente a (-4, 2)
    Hz : bool
        Se for True, plota a frequência em Hz.
        Padrão é True.
    tol : float
        Tolerância admitida para encontrar wB.
        Padrão é 0,1.

    Retonra
    -------
    KB : float
        Ganho em baixas frequências (dB).
    wB : float
        Frequência da banda de passagem. Se Hz=True, em Hz. Caso contrário é retornada em rad/s.
    Kpeak : float
        Ganho máximo do diagrama de Bode.
    wpeak : float
        Frequência em que ocorre Kpeak. Se Hz=True, em Hz. Caso contrário é retornada em rad/s.
    """
    Kss = ct.dcgain(G)         # Ganho estático
    Klw = 20*np.log10(Kss)     # Ganho estático (dB)
    Kb  = pow(10, (Klw-3)/20)  # Ganho da freq. de passagem
    KB  = 20*np.log10(abs(Kb)) # Ganho da freq. de passagem (dB)

    # Verifica se wlim é None
    if (wlim is None):
        wlim = (-4, 2)

    # Verifica se G é discreto e determina o vetor de valores w_
    # Se G for discreto, o vetor irá até próximo da freq. do período de amostragem
    if ct.isdtime(G):
        w_ = np.logspace(wlim[0], np.log10(1.6*np.pi*G.dt),200)
        wlim = (wlim[0], np.log10(np.ceil(2*G.dt)))
    else:
        w_ = np.logspace(wlim[0], wlim[1], 500)

    # Obtém os valores do diagrama de Bode e plota o gráfico
    mag, phase, w = ct.bode(G, w_, dB=True, Hz=Hz)
    # Determina o layour como estreito
    plt.tight_layout()

    #Converte mag para dB
    mag = 20*np.log10(mag)
    # Se Hz=True, converte w de rad/s para Hz
    if Hz:
        w = w/(2*np.pi)

    # Número de amostras de mag
    n = len(mag)
    # Determina wB
    wB = w[next(n-i for i in range(2,n-1) \
          if abs(mag[-i]-KB) < abs(tol))]

    Kpeak = np.max(mag)          # Ganho máximo
    wpeak = w[np.argmax(Kpeak)]  # Freq. do ganho máximo

    # Configurações de plotagem
    ax1,ax2 = plt.gcf().axes     # Obtendo os eixos do subplot

    plt.sca(ax1)                 # Plot da magnitude
    plt.xlim(( pow(10,wlim[0]), pow(10,wlim[1]) )) # xlim
    plt.plot((w_[0],wB),[KB,KB],'k--') # Horizontal
    plt.plot([wB,wB],[KB, plt.ylim()[0]],'k--')    # Vertical
    plt.plot(wB, KB, 'k.')

    plt.sca(ax2)                 # Plot da fase
    #plt.plot((w_[0],w[-1]),[0,0],'k--')   # Horizontal
    #plt.plot([wB,wB],plt.ylim(),'k--')    # Vertical

    return Klw, wB, Kpeak, wpeak

def _genTxt(pol, x, dec):
    """
    Função auxiliar para a função zpk, Gera o texto do numerador ou denominador com base nos zeros ou polos passados.

    Parâmetro
    ---------
    pol : np.array
        Polinômio a ser transformado em texto.
    x : char {'s', 'z'}.
        String da variável utilizada.
    dec : int
        Quantidade de casas depois da vírgula a serem usadas.

    Retorna
    -------
    a : string
        String com o texto do numerador ou denominador.
    """
    a = '' # Pré-alocação
    ii = iter(pol) # Iterador a ser usado
    for i in ii:
        if np.imag(i) <= pow(10, -dec): # Caso não haja parte complexa
            if abs(i) <= pow(10, -dec): # Caso raiz seja 0
                a += x
            elif i < 0: # Caso raiz seja negativa
                a += '(' + x + ' + ' + str(abs(i)) + ')'
            else:       # Caso raiz seja positiva
                a += '(' + x + ' - ' + str(abs(i)) + ')'
        else: # Caso haja parte imaginária
            aux = np.real(np.convolve([1,-i],[1,-np.conj(i)]))
            aux = np.round(aux, dec)
            if abs(aux[1]) <= pow(10, -dec): # Parte real = 0
                a += '(' + x + '² + ' + str(aux[2]) + ')'
            elif aux[1] > 0: # Parte real < 0
                a += '(' + x + '² + ' + str(aux[1]) + x + ' + ' \
                    + str(aux[2]) + ')'
            else:            # Parte real > 0
                a += '(' + x + '² - ' + str(abs(aux[1])) + x + ' + ' \
                    + str(aux[2]) + ')'
            # Admite-se que os polos complexos sejam conjugados, logo
            # deve-se pular o próximo valor
            next(ii)

    return a

def zpk(G, dec=4):
    """
    Printa os zeros, polos e ganho de uma função transferência.

    Parâmetros
    ----------
    G : control.TransferFunction
        Função transferência SISO do sistema a ser analisado.
    label : string
        Nome printado da função transferência.
        Padrão é 'G'.
    dec : int
        Quantidade de algarismos significativos.
        Padrão é 4.
    v : string
        Variável em que ``G`` é descrita.
        Padrão é 's'.

    Retorna
    -------
    z : vetor
        Vetor com os zeros do sistema
    p : vetor
        Vetor com os polos do sistema
    k : float
        Ganho estático do sistema.
    """
    G = ct.minreal(G, pow(10, -dec), verbose=False)

    z, p, k = ct.zero(G), ct.pole(G), np.real(ct.dcgain(G))
    z, p, k = np.round(z, dec), np.round(p, dec), np.round(k, dec)

    x = 's' if ct.isctime(G) else 'z'

    if len(z) == 0:
        a = '1'
    else:
        a = _genTxt(z, x, dec)

    if len(p) == 0:
        b = '1'
    else:
        b = _genTxt(p, x, dec)

    space = ' '
    dash  = '-'

    c = np.round(G.num[0][0][0]/G.den[0][0][0], dec)
    mid = 'G(' + x + ') = ' + str(c) + ' * '

    if len(b) > len(a):
        m, n = len(b), len(a)
        print(int(len(mid)+(m-n)/2-1)*space, a, int((m-n)/2)*space)
        print(mid + m*dash)
        print(len(mid)*space + b)
    else:
        m, n = len(a), len(b)
        print(len(mid)*space + a)
        print(mid + m*dash)
        print(int(len(mid)+(m-n)/2-1)*space, b, int((m-n)/2)*space)

    if not ct.isctime(G):
        print('Ts = ' + str(G.dt) + '.')

    print('Ganho estático: ' + str(k) + '.')

# Funções de simulação
def sim_cont(G, t0, Ts, qnt, Ue, x0):
    """
    Função auxiliar usada para fazer a simulação de um sistema contínuo durante o tempo entre a amostragem do sistema discreto.

    Esta função simula a resposta da planta real (contínua) durante um período de amostragem e retorna o último valor (quando se iniciaria a próxima amostragem). Ela é usada dentro da função ``ctrl.simulaCzGs()``.

    Parâmetros
    ----------
    G : control.tf
        Função transferência contínua da planta/processo.
    t0 : int, float
        Tempo inicial da simulação.
    Ts : int, float
        Taxa de amostragem do controlador.
    qnt : int
        Quantidade de valores a serem computados dentro de t0 a t0 + Ts. Quanto mais elevado, maior a precisão.
    Ue : int, float, np.array
        Sinal de controle durante t0 a t0 + Ts.
    x0 : np.array
        Estado do sistema a t = t0.

    Retorna
    -------
    t : np.array
        Tempo de simulação entre t0 e t0 + Ts.
    y : float
        Resposta do sistema em t0 + Ts.
    x : np.array
        Estado do sistema em t0 + Ts.
    """
    Ue = Ue*np.ones(qnt) if type(Ue) in [float, int] else Ue
    sys = ct.tf2io(G)
    t = np.linspace(t0, t0+Ts, qnt)
    _, y, x = ct.input_output_response(sys, t, Ue, x0, return_x=True)
    return t, y[-1], x.T[-1]

def simulaCzGs(G,
               C,
               n,
               r=None,
               x0=0,
               ulim=None,
               qnt=20):
    """
    Função que simula a implementação do controlador discreto C(z) ligado em série com o processo contínuo G(s) e com realimentação unitária.

    Parâmetros
    ----------
    G : control.tf
        Função transferência contínua do processo a ser controlado.
    C : control.tf
        Função transferência discretizada do controlador.
    n : int
        Número de amostras a serem simuladas.
    r : None, int, float ou np.array
        Sinal de referência a ser aplicado na malha fechada. Caso seja None será usado um vetor unitário. Se for um valor (int ou float), será criado um vetor de tamanho n contendo este valor em todos os casos.
        Padrão é None.
    x0 : 0 ou np.array
        Estado inicial do processo. Caso seja 0, então será criado um vetor nulo do tamanho da ordem de G(s).
        Padrão é 0.
    ulim : list, tuple, array .. de 2 valores
        Implementa a saturação no sinal de controle. Se for None, o sistema é considerado podendo ter valor ilimitado no sinal de controle.
        Padrão é None.
    qnt : int
        Quantidade de instantes na simulação contínua.
        Padrão é 20.

    Retorna
    -------
    t : np.array
        Instantes em que ocorreram as amostragens durante todo o período da simulação.
    Y : np.array
        Resposta discretizada do sistema.
    U : np.array
        Sinal de controle enviado a G(s).
    """
    Ts = C.dt # Taxa de amostragem
    t0 = 0    # Tempo inicial

    K = np.asarray(range(n)) # Vetor com as amostras
    # R recebe um vetor unitário do tamanho do vetor K
    R = np.ones_like(K) if type(r) in [int, float] or r is None else r
    t = Ts*K             # Tempo da simulação
    Y = np.zeros(n+1)    # Resposta
    U = np.zeros_like(K) # Sinal de controle

    m = len(G.den[0][0])-1 # Ordem de G
    x0 = np.zeros(m) if x0 == 0 else x0

    _n, _m = len(C.num[0][0])-1, len(C.den[0][0])-1
    _u, _e = np.zeros(_m), np.zeros(_n)

    # A primeira iteração deve ser zero para o plt.step funcionar corretamente
    for k in K[1:]:
        ek = R[k] - Y[k] # Erro atual
        uk = update_uk(C, ek, _u, _e) # Sinal de controle atual
        if uk is not None:
            # Se uk recebe ulim[0] se for menor que ulim[0]
            # Senão recebe ulim[1] se for maior que ulim[1]
            # Senão recebe uk
            uk = ulim[0] if uk < ulim[0] else ulim[1] if uk > ulim[1] else uk
        # Fim if
        _u = np.concatenate(([uk], _u[0:-1])) # Atualiza uk, uk-1,
        _e = np.concatenate(([ek], _e[0:-1])) # Atualiza ek, ek-1
        U[k] = uk # Salva uk

        _, y, x = sim_cont(G, t0, Ts, qnt, uk, x0)
        Y[k+1] = y
        x0 = x
        t0 += Ts

    return t, Y[0:-1], U

def update_uk(C, ek, _u, _e):
    """
    Função auxiliar que implementa a equação a diferença do controlador e retorna o valor do sinal de controle u[k] considerando o controlador em série com o processo e a realimentação unitária.

    Parâmetros
    ----------
    C : control.tf
        Função transferência discreta do controlador.
    ek : float
        Erro atual (instante k).
    _u : np.array
        Valores de u nos instantes anteriores.
    _e : np.array
        Valores de e nos instantes anteriores.

    Retorna
    -------
    uk : float
        Sinal de controle atual.
    """
    b, a = C.num[0][0], C.den[0][0]
    n = len(_u)
    e = np.concatenate(([ek],_e))
    U, E = sum(a[1:]*_u), sum(b*e)
    uk = (E-U)/a[0]
    return uk

# Construir controlador PID
def PID(K, Ti, Td, N=None):
    """
    Cria um controlador PID com base nos parâmetros K, Ti, Td. É possível utilizar um filtro N no derivativo.

    A dedução foi feita adotando
    * Sem filtro:
              /       1         \
        C = K | 1 + ---- + Td s |
              \     Ti s        /

    * Com filtro:
              /       1       Td s    \
        C = K | 1 + ---- + ---------- |
              \     Ti s   Td s/N + 1 /

    PID:
    Sem o filtro:
              (Ti Td s² + Ti s + 1)
        C = K ---------------------
                      Ti s

    Com filtro:
              (Ti Td (N + 1) s² + (Ti N + Td) s + N)
        C = K --------------------------------------
                         Ti s (Td s + N)

    A variável pode ser s ou z.

    Parâmetros
    ----------
    K : float
        Ganho proporcional do controlador.
    Ti : float ou np.inf
        Ganho do integrador do controlador.
    Td : float
        Ganho do derivativo do controlador.
    N : inteiro positivo
        Filtro do derivativo, normalmente entre 10 e 100.
        Padrão é None.
    """
    #assert K != 0 and Ti != 0 and Td != 0, 'Os parâmetros não podem ser todos nulos'
    assert K != 0, 'O ganho do controlador não pode ser nulo!'
    assert Ti != 0, 'O ganho do integrador não pode ser nulo'
    if N is not None:
        assert N > 0, 'Valor inválido para N!'

    if Ti is np.inf:
        if N is None:
            num = K*np.array([Td, 1])
            den = np.array([1])
        else:
            num = K*np.array([Td*(1+N), N])
            den = np.array([Td, N])
    else:
        if N is None:
            num = K*np.array([Ti*Td, Ti, 1])
            den = np.array([Ti,0])
        else:
            num = K*np.array([Ti*Td*(1+N), Ti*N+Td, N])
            den = np.convolve([Ti,0],[Td,N])

    C = ct.tf(num, den)
    return C

def PIDz(K, Ti, Td, Ts):
    """
    Gera o controlador PID discretizado com base nos parâmetros K, Ti, Td e Ts.

    Parâmetros
    ----------
    K : float
        Ganho proporcional do controlador.
    Ti : float
        Ganho da ação integral do controlador.
    Td : float
        Ganho da ação derivativa do controlador.
    Ts : float
        Taxa de amostragem do sistema discreto.

    Retorna
    -------
    Cpid : control.tf
        Função transferência do controlador PID discretizado com taxa de amostragem de Ts.
    """

    c2 = Ts/Ti + Td/Ts + 1
    c1 = -2*Td/Ts - 1
    c0 = Td/Ts

    num = K*np.array([c2, c1, c0])
    den = np.array([1, -1, 0])

    Cpid = ct.tf(num, den, Ts)
    return Cpid

def discretePID(C, Ts):
    """
    Função usada para discretizar um controlador PID C(s) com uma taxa de amostragem Ts.

    Parâmetros
    ----------
    C : control.tf
        Controlador PID contínuo.
    Ts : int, float
        Taxa de amostragem da discretização.

    Retorna
    -------
    Cz : control.tf
        Controlador PID discretizado.
    """
    b0, b1, b2 = C.num[0][0]
    K, Ti, Td = b1, b1/b2, b0/b1

    Cz = PIDz(K, Ti, Td, Ts)
    return Cz

# Projetar controlador PID
def PID_Pol(G, D, N=None):
    """
    Projeta um controlador PID utilizando o método polinomial para uma função transferência G(s) da forma:

                    a1 s + a0
         G(s) = ------------------
                 b2 s² + b1 s + b0
    para b2 != 0.

    Para tanto, um polinômio D(s) deve ser passado, sendo este da forma:
        D(s) = s³ + d2 s² + d1 s + d0

    Parâmetros
    ----------
    G : control.tf
        Função transferência do sistema a ser controlado.
    D : float
        Polinômio desejado para o denominador.
    N : int ou None
        Filtro no derivativo, a fim de tornar o derivativo implementável. Se None, não haverá filtro.
        Padrão é None.

    Notas
    -----
    O filtro é implementado depois de se projetar o controlador, então sua presença pode gerar um comportamento indesejado e deve ser avaliado e alterado conforme os requisitos dados.
    """

    num, den = G.num[0][0], G.den[0][0]

    assert len(num) <= 2 and len(den) == 3, '''
    Formato de G(s) incorreto!
    Certifique que ele seja da forma:
    \t   a1 z + a0\n\t-----------------\n\tb2 z² + b1 z + b0'''
    if len(num) == 2:
        b1, b0 = num
    else:
        b1, b0 = 0, num[0]
    a2, a1, a0 = den

    d3, d2, d1, d0 = D
    assert d3 == 1, '''
    Formato de D(s) errado! Deve ser:
    \t\t D(s) = s³ + d2 s² + d1 s + d0'''

    E = np.array([[ 0, b1, b0-b1*d2],
                  [b1, b0,   -b1*d1],
                  [b0,  0,   -b1*d0]])

    B = np.array([[d2-a1],
                  [d1-a0],
                     [d0]])

    M = inv(E).dot(B)

    b2, b1, b0 = M[0][0], M[1][0], M[2][0]

    K, Td, Ti = b1, b0/b1, b1/b2
    C = pid(K, Ti, Td, N=N)

    return C, [K, Ti, Td, N]

def PID_LGR(G, p, zPI=-0.5, Print=True, xlim=None):
    """
    Projeta um controlador PID pelo método LGR.

    A função recebe o processo a ser controlado e o polo desejado do sistema. Opcionalmente é passado o zero da ação PI do controlador (deve ser próximo de 0). Então, é retornado o controlador PID projetado pelo método LGR.

    Parâmetros
    ----------
    G : control.tf
        Função transferência do sistema a ser controlado.
    p : np.array
        Polo dominante de malha fechada desejado
    zPI : int, float
        Zero da ação PI do controlador.
        Deve ser posicionado próximo de 0.
        Padrão é -0.5
    Print : bool
        Se True, será mostrado o LGR do sistema compensado (sem o ganho).
        Padrão é True.
    xlim : None, list, tuple .. de 2 valores
        Limites do eixo real do LGR. Se for None, a função control.rlocus() decidirá os limites.
        Padrão é None.

    Retorna
    -------
    K : float
        Ganho que leva o sistema compensado C(s)G(s) para o polo desejado considerando uma realimentação unitária.
    Cpid : control.tf
        Compensador PID projetado.

    Nota
    ----
    O controlador final será dado por K Cpid(s).
    """
    zero, pole = ct.zero(G), ct.pole(G)

    # Deficiência angular
    ph = np.angle(p-zero) # Contribuição dos zeros
    th = np.angle(p-pole) # Contribuição dos polos
    be = np.pi + sum(ph) - sum(th)  # Deficiência angular

    # Distância do zero do PD ao polo desejado
    a = np.imag(p)/np.tan(-be)
    zPD = np.real(p) - a

    num = np.convolve([1, -zPI], [1, -zPD])
    den = np.array([1,0])

    Cpid = ct.tf(num, den)
    _H = Cpid*G

    K = 1/abs(ct.evalfr(_H, p))

    if Print:
        ct.rlocus(_H) if xlim is None else ct.rlocus(_H, xlim=xlim)
        plt.scatter(np.real([p, p]), np.imag([p, -p]), c='r')

        #H = ct.feedback(K*_H)
        #zz, pp = ct.zero(H), ct.pole(H)
        #plt.scatter(np.real(zz), np.imag(zz), c='k')
        #plt.scatter(np.real(pp), np.imag(pp), c='k', marker='x')

    return K, Cpid

def sintoniaPID(param, met='ZN',C='PID', modo=None):
    """
    Sintoniza um controlador PID, PI ou P com base nos parâmetros do modelo identificado e no método pedido.

    Parâmetros
    ----------
    param : tuple, list, array .. de 2 ou 3 valores
        Caso seja ZN ou ITAE então ``K, T, L = param``,
        Caso seja Kcr, ``Kcr, Pcr = param``
    met : string
        Método usado para a sintonia.
        Se for ZN, será feita a sintonia usando os valores da tabela  descobertos por Ziegler-Nichols.
        Se for ITAE, utilizará valores com base na minimização o ITAE.
        Se for Kcr, utilizará o método de Ziegler-Nichols mas com a identificação dos ganhos limites do sistema.
    C : string
        Escolha da forma do PID.
        Se for PID, PI ou P será projetado um controlador homônimo.
        Se for OS, será projetado um controlador PID para 20% de OS.
        Caso contrário, será projetado um controlador PID para 0% de OS.
    Modo : string ou None
        Modo do controlador para o método de minimização de ITAE. Se None, o modo será reação. Caso contrário, o modo servo.
        Padrão é None.
    """

    met, C = met.upper(), C.upper()
    # Tabela correspondente aos métodos de sintonia
    table = [
             [ #Método 'ZN'
              [1.2, 2, 0.5],   # PID
              [0.9, 1/0.3, 0], # PI
              [1, np.inf, 0]], # P
             [ #Método ITAE
              # Reação
              [1.357, -0.947, 0.842, -0.738, 0.842, -0.738], # PID
              [0.859, -0.977, 0.674, -0.680, 0, 1],          # PI
              # Servomotor
              [0.965, -0.850, 0.796, -0.1465,0.308,  0.929], # PID
              [6, -0.916, 1.030, -0.165, 0, 1]],             # PID
             [ # Método Kcr
              [0.6,0.5,0.125],  # PID
              [0.45, 1/1.2, 0], # PI
              [0.5, np.inf, 0], # P
              [0.33,0.5,0.125], # OS
              [0.2,0.5,1/3]]]   # Sen OS

    met, C = met.upper(), C.upper()

    # Valores de n e n correspondentes a table conforme as opções
    m = 0 if met=='ZN' else 1 if met=='ITAE' else 2
    n = 0 if C=='PID' else 1 if C=='PI' else 2 if C=='P' else 3 if C=='OS' else 4

    if m==0: # ZN
        K, T, L = param
        Kp = table[m][n][0]*T/K/L
        Ti = table[m][n][1]*L
        Td = table[m][n][2]*L
    elif m==1: # ITAE
        K, T, L = param
        if modo is None: # Reação
            Ap,Bp,Ai,Bi,Ad,Bd = table[m][n]
            Yp, Yi, Yd = Ap*(L/T)**Bp, Ai*(L/T)**Bi, Ad*(L/T)**Bd
        else: # Servo
            Ap,Bp,Ai,Bi,Ad,Bd = table[m][n+2]
            Yp, Yi, Yd = Ap*(L/T)**Bp, Ai+Bi*(L/T), Ad*(L/T)**Bd

        Kp = Yp/K
        Ti = T/Yi
        Td = Yd*T
    else: # Kcr
        Kcr, Pcr = param
        Kp = table[m][n][0]*Kcr
        Ti = table[m][n][1]*Pcr
        Td = table[m][n][2]*Pcr

    return Kp, Ti, Td

# Funções em desenvolvimento
def latexTable(t, y1, y2):
    """
    Função em desenvolvimento.
    Gerar o texto em LaTeX dos critérios de desempenho de duas respostas.
    """
    tss1, OS1, US1, tr1, *_ = stepinfo(t, y1)
    tss2, OS2, US2, tr2, *_ = stepinfo(t, y2)

    tss1, OS1, US1, tr1 = np.round([tss1, OS1, US1, tr1], 3)
    tss2, OS2, US2, tr2 = np.round([tss2, OS2, US2, tr2], 3)

    amb = 'table'
    cap = 'Critérios de desempenho dos controladores'
    lab = 'tab:CritDesempenho'
    tab = 'tabular'

    table = f'''
    \begin{amb}[ht]
    \caption{cap}
    \label{lab}
    \begin{tab}
    \toprule
    Parâmetro & Controlador 1 & Controlador 2 \\
    \midrule
    $t_s$ (s) & {tss1} & {tss1} \\
    OS (\%) & {OS1} & {OS2} \\
    US (\%) & {US1} & {US2} \\
    $t_r$ (s) & {tr1} & {tr2} \\
    \bottomrule
    \end{tab}
    \end{amb}
    '''

    return table

def find_nearest(array, value):
    """
    Função usada para achar o índice em que um valor é o mais próximo possível de um array passado.
    """
    array = np.asarray(array)
    i = (np.abs(array - value)).argmin()
    return i















