#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:16:47 2020

@author: Bernardo Bresolini

Este codigo utiliza como base os dados coletados no arquivo Tank_MA.py.
Os dados foram salvos em 2 arquivos .CSV:
    * O primeiro define os pontos de operaçao na ordem
        yop, uop, dt, max_dy, max_du
    * O segundo contem os dados da simulaçao, na ordem
        t, y, u, r

Vale ressaltar que o periodo de amostragem dt eh de 0,25 segundos.
O tempo de acomodaçao ts adotado foi de 2000 s.
"""
import matplotlib.pyplot as plt #importando a biblioteca matplotlib - para plotagem
import numpy as np #Importando a biblioteca numpy - para ambiente matemático com vetores
import csv # Usada para importar os arquivos coletados em .CSV
import pandas as pd # Usada para printar dados csv
from math import exp # Computa a exponencial de um numero
import control
from control.matlab import tf
#from sklearn.metrics import mean_squared_error # Usada para calcular o RMSE

plt.close('all') #fechando todas as janelas com figuras

# Importando os dados do ponto de operaçao
with open('pto_op.csv', 'r') as f:
    reader_op  = csv.reader(f, delimiter=',')
    headers_op = next(reader_op)
    data_op    = np.array(list(reader_op)).astype(float)

# Importando os dados dos vetores de dados coletados
with open('vetores.csv', 'r') as f:
    reader_vec  = csv.reader(f, delimiter=',')
    headers_vec = next(reader_vec)
    data_vec    = np.array(list(reader_vec)).astype(float)


# Obtendo os valores
yop, uop, ts, dt, max_dy, max_du = data_op[-1]
t = data_vec[0,:] # tempo
y = data_vec[1,:] # saida
u = data_vec[2,:] # sinal de controle
r = data_vec[3,:] # referencia


# Visualizaçao dos dadoos coletados
#plt.figure(1)
#plt.subplot(2,1,1) #Plotando a saida do sistema
#plt.plot(t,r)     #Referencia
#plt.plot(t,y)     #Saida
#plt.xlabel('Tempo (s)')
#plt.ylabel('h (cm)')
#plt.xlim([0,10*ts])
#plt.ylim((0,45))
#plt.yticks([0,10,20,30,yop,40])
#plt.grid(linestyle='--')
#
#u_inf = (uop-max_du)*np.ones(len(u))
#u_sup = (uop+max_du)*np.ones(len(u))
#plt.subplot(2,1,2) #Plotando o sinal de controle
#plt.plot(t,u_inf,'k--') #Limite inferior
#plt.plot(t,u_sup,'k--') #Limite superior
#plt.plot(t,u)    #Sinal de controle
##Configuraçoes de plotagem
#plt.xlabel('Tempo (s)')
#plt.ylabel('u (%)')
#plt.xlim([0,10*ts])
#plt.ylim((50,60))
#plt.yticks([50,u_inf[0],uop,55,u_sup[0],60])
#plt.grid(linestyle='--')
#plt.show()

# Cada step dura ts e a cada segundo existem 1/dt valores. Logo, a quantidade
# de valores em cada step eh
qnt = int(ts/dt)


# Selecionando as 3 amostras
"""Os intervalos selecionados foram
    -> 2° step: de uop a + 0,5*max_du
    -> 4° step: de uop a - 0,5*max_du
    -> 6° step: de uop a - 1,0*max_du
    sendo max_du a alteraçao maxima no sinal de controle u (valor que aumenta
    ou diminui o ponto de operaçao em 7% do fundo de escala: 4,9 cm.).

    Para alterar, deve-se alterar nos i1, i2... e na subtraçao de tt
"""
# Intervalos correspondentes
i1 = range(1*qnt,2*qnt) # Intervalo 1
i2 = range(3*qnt,4*qnt) # Intervalo 2
i3 = range(5*qnt,6*qnt) # Intervalo 3

# Pre-alocaçao
tt = np.zeros( (3,qnt) )
yy = np.zeros( (3,qnt) )
rr = np.zeros( (3,qnt) )
uu = np.zeros( (3,qnt) )

# Ao longo do codigo, se usara muito as formas de yy ou tt, logo para facilitar, tem-se
szY = yy.shape    # linhas e colunas
r_y = yy.shape[0] # Apenas linhas  (rows)
c_y = yy.shape[1] # Apenas colunas (columns)

# Amostra 1
tt[0][:] = t[i1]-1*ts  # Trazendo o tempo para 0
yy[0][:] = y[i1]-yop   # Variaçao no ponto de operaçao
rr[0][:] = r[i1]-r[0]  # Variaçao da ref no pto de operaçao
uu[0][:] = u[i1]-uop   # Variaçao do sinal de controle no pto de operaçao

# Amostra 2
tt[1][:] = t[i2]-3*ts  # Trazendo o tempo para 0
yy[1][:] = y[i2]-yop   # Variaçao no ponto de operaçao
rr[1][:] = r[i2]-r[0]  # Variaçao da ref no pto de operaçao
uu[1][:] = u[i2]-uop   # Variaçao do sinal de controle no pto de operaçao

# Amostra 3
tt[2][:] = t[i3]-5*ts  # Trazendo o tempo para 0
yy[2][:] = y[i3]-yop   # Variaçao no ponto de operaçao
rr[2][:] = r[i3]-r[0]  # Variaçao da ref no pto de operaçao
uu[2][:] = u[i3]-uop   # Variaçao do sinal de controle no pto de operaçao

# A partir daqui o codigo executa automaticamente
# A partir daqui o codigo executa automaticamente
# A partir daqui o codigo executa automaticamente
# A partir daqui o codigo executa automaticamente
# A partir daqui o codigo executa automaticamente

# Visualizaçao das amostras
#plt.figure(2)
#plt.subplot(3,1,1)
#plt.plot(tt[0],rr[0])
#plt.plot(tt[0],yy[0])
#plt.xlim([tt[0][0],tt[0][-1]])
#plt.ylabel('altura (cm)')
#plt.grid(linestyle='--')
#
#plt.subplot(3,1,2)
#plt.plot(tt[1],rr[1])
#plt.plot(tt[1],yy[1])
#plt.ylabel('altura (cm)')
#plt.xlim([tt[1][0],tt[1][-1]])
#plt.grid(linestyle='--')
#
#plt.subplot(3,1,3)
#plt.plot(tt[2],rr[2])
#plt.plot(tt[2],yy[2])
#plt.xlabel('Tempo(s)')
#plt.ylabel('altura (cm)')
#plt.xlim([tt[2][0],tt[2][-1]])
#plt.grid(linestyle='--')
#plt.show()

# Determinaçao do ganho estatico
k = np.zeros( (r_y) ) # Pre-alocaçao
for i in range(0, r_y):
    k[i] = yy[i][-1]/uu[i][-1] # k = y(ts)/u(ts)

# Calculo feito para achar B
"""
    Considere a resposta temporal
            |
   13 (yss) | ..............    ________________________
   12       | ............    -                         |
   11       | ..........    -                           |
   10       | .........   -                             |
   09       | .........  -                              |
   08       | ........  -                               |
   07   (B) | .......  -                                |
   06       | ......  -|                                |
   05       | .....  - |                                |
   04       | ....  -  |                                |
   03       | ...  -   |                                |
   02  (y0) | ----|    |                                |
   01       |     |    |                                |
    0       |_____|____|________________________________|______>
                t(0)  t(B)                             tss      t

    O ponto B equivale a
       0,63 ( yss - y0 ) + y0
"""
iB  = np.zeros( (r_y) )   # index do ponto B
pto = np.zeros( (r_y) )  # pre-alocacao do ponto onde a saida alcançou 63% da variaçao

for i in range(0, r_y):
    pto[i] = 0.63*( yy[i][-1] - yy[i][0] ) + yy[i][0]
    for j in range(0, c_y):
        if ( abs(pto[i] - yy[i][j] ) < 0.001): # nunca usar ' == 0'
            iB[i] = j
            break
        # end if
    # end for j
# end for i


dY = np.diff( yy )   # derivada de yy
iYmax = np.argmax( np.abs(dY), axis = 1 ) # Indica o ponto de maior inclinaçao


# Pre alocaçao dos coeficientes da reta e da propria reta
a1   = np.zeros( (r_y) )
a0   = np.zeros( (r_y) )
reta = np.zeros( (szY) )
"""
    A inclinaçao m da reta tangente de uma curva eh dada por
    m = dy/dt ===>  m = ( y(t) - y(t-dt) )/dt

    Para achar a inclincaçao de um ponto P, basta
    aplicar o tempo daquele ponto t(P)
    m(P) = ( y( t(P) ) - y( t(P)-dt ) )/dt
"""
for i in range(0,r_y):
    p = int(iYmax[i])
    a1[i] = dY[i][p] / dt
    a0[i] = yy[i][p+1] - a1[i]*tt[i][p+1]
    reta[i] = a1[i]*tt[i] + a0[i]

    iB = np.array(iB, int)
    # Visualizacao da reta traçada
    plt.subplot(3,1,1+i)
    plt.plot(tt[i],reta[i],'#e11d74')
    plt.plot(tt[i],yy[i],'#00bcd4')
    plt.plot(tt[i][iB[i]],yy[i][iB[i]],'r.')
    plt.rc('legend', fontsize=9,loc='best')
    plt.legend(['Reta','$y(t)$','Ponto $B$'])
    plt.xlim([ tt[i][0],tt[i][-1] ])
    plt.ylim([ min(yy[i]),max(yy[i]) ])
    plt.grid(linestyle='--')

#end for

plt.show()
plt.savefig('Imagens/retas.eps',format='eps',dpi=3000)


# Obtençao de A, B e C
"""
    -> Ja possuimos o index de B, logo B e t(iB)
    -> Para obter A, basta resolver a  equaçao da reta para y = 0, logo
            a1*A + a0 = 0   ==> A = -a0/a1
    -< Para obter C, basta resolver a equacao da reta para  y = yss, logo
            a1*C + a0 = yss ==> C = (yss - a0)/a1
"""
A = np.zeros( (r_y) )
B = np.zeros( (r_y) )
C = np.zeros( (r_y) )

for i in range( 0, r_y ):
    # A
    A[i] = -a0[i]/a1[i]

    """
    O atraso nao pode ser negativo. Deste modo, nem o valor de A pode ser
    negativo, i.e, a0 e a1 terem o mesmo sinal.
    """
    if (A[i] < 0):
        A[i] = 0
    # end if

    # B
    iB = np.array(iB,int)
    B[i] = tt[i][iB[i]]

    # C
    C[i] = (yy[i][-1] - a0[i])/a1[i]

# end for

# Obtençao da constante de tempo
L  = A     # Atraso
T1 = C-A   # Constante de tempo pelo metodo 1
T2 = B-A   # Constante de tempo pelo metodo 2

tau1 = L/(L+T1)  # Taxa normalizada de tempo morto: metodo 1
tau2 = L/(L+T2)  # Taxa normalizada de tempo morto: metodo 2

ystep1 = np.empty(szY) #pre-alocação na memória do estado do sistema
ystep1.fill(np.nan)
ystep2 = np.empty(szY) #pre-alocação na memória do estado do sistema
ystep2.fill(np.nan)

for i in range(0, r_y):
    for j in range(0, c_y): #simulação do sistema em malha aberta
        # Step para T =  AC
        ystep1[i][j] = k[i]*uu[i][j]*( 1 - exp( -(tt[i][j]-L[i])/T1[i]) ) * \
        np.heaviside(tt[i][j]-L[i], 1)

        # Step para T = AB
        ystep2[i][j] = k[i]*uu[i][j]*( 1 - exp( -(tt[i][j]-L[i])/T2[i]) ) * \
        np.heaviside(tt[i][j]-L[i], 1)

    # end for j
    ystep1[i] = ystep1[i] + yy[i][0]  # Ajustando o inicio com os dados coletados
    ystep2[i] = ystep2[i] + yy[i][0]  # Ajustando o inicio com os dados coletados
# end for i


# Visualizaçao das respostas temporais
plt.figure(4)
plt.subplot(3,1,1)
plt.plot(tt[0],rr[0],'k--')
plt.plot(tt[0],yy[0],'#00bcd4')
plt.plot(tt[0],ystep1[0],'#158467')
plt.plot(tt[0],ystep2[0],'#0f4c75')
plt.xlim([tt[0][0],tt[0][-1]])
plt.ylabel('altura (cm)')
plt.rc('legend', fontsize=7)
plt.legend(['Ref','Dados','Modelo AC','Modelo AB'])
plt.grid(linestyle='--')

plt.subplot(3,1,2)
plt.plot(tt[1],rr[1],'k--')
plt.plot(tt[1],yy[1],'#00bcd4')
plt.plot(tt[1],ystep1[1],'#158467')
plt.plot(tt[1],ystep2[1],'#0f4c75')
plt.ylabel('altura (cm)')
plt.rc('legend', fontsize=7)
plt.legend(['Ref','Dados','Modelo AC','Modelo AB'])
plt.xlim([tt[1][0],tt[1][-1]])
plt.grid(linestyle='--')

plt.subplot(3,1,3)
plt.plot(tt[2],rr[2],'k--')
plt.plot(tt[2],yy[2],'#00bcd4')
plt.plot(tt[2],ystep1[2],'#158467')
plt.plot(tt[2],ystep2[2],'#0f4c75')
plt.xlabel('Tempo(s)')
plt.ylabel('altura (cm)')
plt.rc('legend', fontsize=7)
plt.legend(['Ref','Dados','Modelo AC','Modelo AB'])
plt.xlim([tt[2][0],tt[2][-1]])
plt.grid(linestyle='--')
plt.show()
plt.savefig('Imagens/AC-AB.eps',format='eps',dpi=3000)

# Escolha do modelo com maior coeficiente de determinacao
"""
    Segundo Wikipedia
    O coeficiente de determinação, também chamado de R², é uma medida de ajuste
    de um modelo estatístico linear generalizado, como a regressão linear simples
    ou múltipla, aos valores observados de uma variável aleatória. O R² varia
    entre 0 e 1, por vezes sendo expresso em termos percentuais. Nesse caso,
    expressa a quantidade da variância dos dados que é explicada pelo modelo
    linear. Assim, quanto maior o R², mais explicativo é o modelo linear, ou
    seja, melhor ele se ajusta à amostra. Por exemplo, um R² = 0,8234 significa
    que o modelo linear explica 82,34% da variância da variável dependente a
    partir do regressores (variáveis independentes) incluídas naquele modelo linear.
    Ele eh calculado por
                       _________________________
                      |                         |
                      |              SQres      |
                      |    R² = 1 − ------      |
                      |              SQtot      |
                      |_________________________|
"""
SQtot  = np.zeros( r_y )  # Soma total dos quadrados
SQres1 = np.zeros( r_y )  # Soma dos quadrados dos residuos do modelo 1
SQres2 = np.zeros( r_y )  # Soma dos quadrados dos residuos do modelo 1
R2 = np.zeros( (2,r_y) )  # Coeficiente de determinacao

for i in range( r_y ):
    SQtot[i]  = sum( ( yy[i] - np.mean(yy[i]) )**2 )
    SQres1[i] = sum( ( yy[i] - ystep1[i] )**2 )
    SQres2[i] = sum( ( yy[i] - ystep2[i] )**2 )

    # Coeficiente de determinacao
    R2[0][i] = 1 - SQres1[i]/SQtot[i]
    R2[1][i] = 1 - SQres2[i]/SQtot[i]
# end for

# Obtendo a media dos 3 modelos
kmod = np.mean(k)
Lmod = np.mean(L)

# O modelo escolhido, sera aquele com maior media de R²
if ( np.mean(R2[0]) > np.mean(R2[1])):
    Tmod  = np.mean(T1)   # Modelo escolhido
    Tmod_ = np.mean(T2)   # Modelo rejeitado
    #end if
else:
    Tmod  = np.mean(T2)   # Modelo escolhido
    Tmod_ = np.mean(T1)   # Modelo rejeitado
# end else

taumod  = Lmod/(Lmod+Tmod)
taumod_ = Lmod/(Lmod+Tmod_)

# Intervalo para o teste
inicio  = 7
termino = 10


i_test = range(inicio*qnt,termino*qnt)
t_test = t[i_test]-inicio*ts
y_test = y[i_test]-yop
r_test = r[i_test]-r[0]  # Variaçao da ref no pto de operaçao
u_test = u[i_test]-uop

G1  = tf(kmod, [ Tmod, 1  ])     # Cria a funçao transferencia do modelo escolhido
G1_ = tf(kmod, [ Tmod_, 1 ])     # Idem, mas para o modelo rejeitado

na, da = control.pade(Lmod,10) # Aproxima o atraso pelo polinomio de Pade
atraso = tf(na,da)             # Cria funçao transferencia do atraso

G1  = control.series(G1, atraso) # Associando em serie G com o atraso
G1_ = control.series(G1_,atraso) # Rejeitado com atraso
_T, Y, _X  = control.forced_response(G1, t_test,u_test) # Resposta forçada para  u = u_test
_T, Y_, _X = control.forced_response(G1_,t_test,u_test) # Resposta forçada para  u = u_test (rejeitado)


# Obtençao de T para o modelo de 2a ordem
"""
Considere a equacao da saida
        s(t) = K*[ 1 - (1 + (t-L)/T )*exp( -(t-L)/T ) ]
    ==> s(t) - K = - K(1 + (t-L)/T )*exp( -(t-L)/T )
    Se w = -(t-L)/T - 1, segue
        1 - s(t)/K = - w exp ( w + 1 )
    Aplicando a propriedade do exponencial
        1 - s(t)/K = -w (e^w)*(e^1)
    ==> -(1 - s(t)/K)/e = w e^w

    O termo na direita eh resolvido na literatura matematica e de computaçao
    cientifica como equaçao de Lambert e eh vastamente aplicado² em problemas
    de engenharia, fisica e matematica.

    Por fim, se t = tss, entao s(t=tss) = 0,98K. Assim, segue
        -0,02/e = w e^w

    Obtendo w que satisfaza a equaçao e tss graficamente, faz-se
        w + 1 = - (tss-L)/T
    ==> T = - (tss-L)/(w+1)

    Considere a funçao
        f(w) = w e^w - z = 0
    sendo z uma constante.

    A aplicaçao do metodo de Halley¹ leva
       w_{k+1} = w_k - \frac{ f }{ e^{w_k}(w_k+1) - \frac{(w_k+2)f}{2w_k + 2} }
    cuja convergencia eh cubica.


    ¹ Veja CORLESS, R. M. et al. no artigo "On the Lambert W Function", dispo-
    nivel em <https://cs.uwaterloo.ca/research/tr/1993/03/W.pdf>. La eh mencio-
    nado as aplicaçoes da equaçao de Lambert. Alguns exemplos sao:
        * Combinatoria
        * Exponencial iterativo h^{ z^ { z^{ z^\cdots } } }
        * Soluçao de equaçoes transcendentais
        * Soluçao do problema de combustivel de jato (Jet Fuel)
        * Cinetica de enzimas
        * Soluçao de equaçoes lineares de delay com coeficientes constantes
        * Soluçao de similaridade para a equaçao de Richards

    ² O metodo de Halley eh dado por
       x_{k+1} = x_k - \frac{ f }{ f' - \frac{f \cdot f''}{ 2 f' } }
"""

# tss eh o tempo no qual y alcança 0,98 K * u, logo
iD  = np.zeros( (r_y) )   # index do ponto D
pto = np.zeros( (r_y) )  # pre-alocacao do ponto onde a saida alcançou 98% da variaçao

for i in range (0, r_y):
    # Valor da ordenada que corresponde a 98% da saida
    pto[i] = 0.98*( yy[i][-1] - yy[i][0] ) + yy[i][0]
    for j in range (0, c_y):
        if ( abs(pto[i] - yy[i][j] ) < 0.001): # nunca usar ' == 0'
            iD[i] = j
            break
        # end if
    # end for j
# end for i

iD = np.array(iD,int) # Transforma em inteiro
t_inf  = np.zeros( (r_y) ) # Pre-alocacao
for i in range( 0, r_y ):
    t_inf[i] = tt[i][iD[i]] # tempo (em segundos) no qual y = 0,98*K*u
# end for

# Escolhendo a media dos 3 modelos
tss = np.mean(t_inf)

# Metodo numerico para determinar a funçao W de Lambert para um z dado
#                  w*exp(w) = z
def lamb(w,z):
    """
        Metodo que calcula e retorna funçao W de Lambert
                    w*exp(w) -  z
        para w e z dados.

        Parameters:
            w (float): Variavel da equaçao de Lambert
            z (float): Constante da equaçao de Lambert

        Returns:
            float: Returnando o calculo de w*exp(w) - z
    """
    f = w*exp(w) - z
    return f

def halley_lamb(z, w0=0, n=5):
    """
        Metodo que resolve¹ (para w) a funçao W de Lambert
                    w*exp(w) -  z
        para um z conhecido.

        Para tanto, eh usado o metodo numerico de Halley² aplicado a funçao W
        de Lambert. Como o metodo de Halley tem convergencia cubica, bastam
        poucas iteraçoes para se alcançar elevadas precisoes numericas.

        ¹ Nota: para -1/e < z < 0 existem 2 valores de w que satisfazem
        a equaçao. Logo, verifique a resposta caso z esteja nesse caso.

        ² O metodo de Halley eh uma adaptaçao do metodo de Newton. Ele parte
        de g = f/sqrt(|f'|).

        Parameters:
            z  (float): Constante da equaçao de Lambert
            w0 (float): Valor inicial. Altere ele para encontrar a segunda
                        resposta, caso seja necessario. Menor valor eh 1.
            n    (int): Numero de iteraçoes realizadas.
        Returns:
            float: Retorna a resoluçao numerica de w*exp(w) - z = 0
    """
    w = np.empty( (n,) ) # Pre-alocaçap
    w.fill(np.nan)

    w[0] = w0 # Chute inicial
    # Primeira iteracao
    w[1] = w[0] - lamb(w[0],z)/ \
    ( exp(w[0])*(w[0]+1) - (w[0]+2)*lamb(w[0],z)/(2*w[0]+2)  )

    # for para as n iteraçoes
    for i in range(1,len(w)-1):
        w[i+1] = w[i] - lamb(w[i],z)/ \
        ( exp(w[i])*(w[i]+1) - (w[i]+2)*lamb(w[i],z)/(2*w[i]+2)  )
    # end for
    return w[-1]

# Definindo z, obtido da equaçao da resposta do sistema
z  = -0.02/np.e
# Obtendo a resoluçao da eq. de Lambert
w = halley_lamb(z,-6)
# Encontrando o T
T = -(tss-Lmod)/(w+1)
# Controlabilidade
taumod2 = Lmod/(Lmod+T)

# Denominador da tf de 2a ordem
d  = np.convolve([T, 1],[T, 1])
G2 = tf(kmod,d) # tf de 2a ordem

G2 = control.series(G2,atraso) # Associando em serie G com o atraso
_T, Y2, _X = control.forced_response(G2,t_test,u_test) # Resposta forçada para  u = u_test

# Trazendo as respostas para o ponto inicial
Y  = Y  + y_test[0]
Y_ = Y_ + y_test[0]
Y2 = Y2 + y_test[0]


# Respostas temporais dos modelos
plt.figure(5)
plt.plot(t_test,r_test,'k--')
plt.plot(t_test,y_test,'#00bcd4')
plt.plot(t_test,Y,'#f0a500')
plt.plot(t_test,Y2,'#fe7171')
plt.title('Comparaçao do modelo com os dados')
plt.rc('legend', fontsize=12)
plt.legend(['Ref','Dados','Modelo 1 ordem','Modelo 2 ordem'])
plt.xlabel('Tempo(s)')
plt.ylabel('Altura (cm)')
plt.xlim([t_test[0],t_test[-1]])
plt.grid(linestyle='--')
plt.show()
plt.savefig('Imagens/1e2-ordem.eps',format='eps',dpi=3000)

# Pre-alocaçao
_Y1 = np.zeros( (r_y,t_test.shape[0]) )
_Y2 = np.zeros( (r_y,t_test.shape[0]) )
IAE  = np.zeros( (2,r_y) )
ITAE = np.zeros( (2,r_y) )
RMSE = np.zeros( (2,r_y) )

#plt.figure(6)
for i in range( (r_y) ):
    G1 = tf(k[i], [T1[i], 1]) # T = AC
    G2 = tf(k[i], [T2[i], 1]) # T = AB
    _T, _Y1[i], _X = control.forced_response(G1, t_test, u_test) # T = AC
    _T, _Y2[i], _X = control.forced_response(G2, t_test, u_test) # T = AB

#    plt.subplot(3,1,i+1)
#    plt.plot(t_test,Y1[i])
#    plt.plot(t_test,Y2[i])
#    plt.xlim([ t_test[0],t_test[-1] ])
#    plt.grid(linestyle='--')

    # Calculo de IAE
    """
        O IAE eh calculado conforme
                IAE = \sum_{j=1}^r | y(j) - \hat{y}(j) | dt
        sendo $r$ a quantidade total de elementos, $y$ o curva de valores
        observados, $\hat{y}$ os valores estimados, dt o passo temporal dos
        valores.
    """
    IAE[0][i] = sum( abs(y_test - _Y1[i])*dt )
    IAE[1][i] = sum( abs(y_test - _Y2[i])*dt )

    # Calculo de IAE
    """
        O ITAE eh calculado conforme
                ITAE = \sum_{j=1}^r | y(j) - \hat{y}(j) | j dt^2
        sendo $r$ a quantidade total de elementos, $j$ a posiçao de um elemento
        nos vetores, $y$ o curva de valores observados, $\hat{y}$ os valores
        estimados, dt o passo temporal dos valores.
    """
    j = np.array(range(1,t_test.shape[0]+1), int)
    ITAE[0][i] = sum( abs(y_test - _Y1[i])*j*dt**2 )
    ITAE[1][i] = sum( abs(y_test - _Y2[i])*j*dt**2 )

    # Calculo de IAE
    """
        O RMSE (Root Mean SquerE) eh calculado conforme
                RMSE = \sqrt{ \overline{ \big( y(k) - \hat{y}(k) \big)^2 } }
        sendo $y$ o curva de valores observados, $\hat{y}$ os valores
        estimados, a barra significa que eh a media dos dados.
    """
    RMSE[0][i] = np.sqrt( np.mean( (y_test - _Y1[i])**2 ) )
    RMSE[1][i] = np.sqrt( np.mean( (y_test - _Y2[i])**2 ) )

##end for
#plt.show()

# IAE
IAEmod  = sum( abs(y_test - Y) *dt)   # Primeira ordem escolhido
IAEmod_ = sum( abs(y_test - Y_)*dt)   # Primeira ordem rejeitado
IAEmod2 = sum( abs(y_test - Y2)*dt)   # Segunda ordem

# ITAE
j = np.array(range(1,t_test.shape[0]+1), int)
ITAEmod  = sum( abs(y_test - Y) *j*dt**2 )    # Primeira ordem escolhido
ITAEmod_ = sum( abs(y_test - Y_)*j*dt**2 )    # Primeira ordem rejeitado
ITAEmod2 = np.sum( abs(y_test - Y2)*j*dt**2 ) # Segunda ordem

# RMSE
RMSEmod  = np.sqrt( np.mean( (y_test - Y) **2 ) )   # Primeira ordem escolhido
RMSEmod_ = np.sqrt( np.mean( (y_test - Y_)**2 ) )   # Primeira ordem rejeitado
RMSEmod2 = np.sqrt( np.mean( (y_test - Y2)**2 ) )   # Segunda ordem

SQtotmod  = sum( ( y_test - np.mean(y_test) )**2 )
SQresmod  = sum( ( y_test - Y  )**2 )
SQresmod_ = sum( ( y_test - Y_ )**2 )
SQresmod2 = sum( ( y_test - Y2 )**2 )

R2mod  = 1 - SQresmod /SQtotmod
R2mod_ = 1 - SQresmod_/SQtotmod
R2mod2 = 1 - SQresmod2/SQtotmod


# Salvar os dados dos modelos
header = np.array( ['Modelos', 'K', 'T', 'L', 'tau', 'R2', 'IAE', 'ITAE', 'RMSE'] )

# Salvando as informaçoes do ponto de operaçao e dos limites
with open('modelos.csv', mode='w') as modelo_file:
    mod_wtr = csv.writer(modelo_file, \
                        delimiter=',', \
                        quotechar='"', \
                        quoting=csv.QUOTE_MINIMAL)

    mod_wtr.writerow(header)
    for i in range (0, r_y):
        mod_wtr.writerow( ['AC: %d' %(i+1), k[i], T1[i], L[i], tau1[i], R2[0][i], IAE[0][i], ITAE[0][i], RMSE[0][1] ])
    for i in range (0, r_y):
        mod_wtr.writerow( ['AB: %d' %(i+1), k[i], T2[i], L[i], tau2[i], R2[1][i], IAE[1][i], ITAE[1][i], RMSE[1][1] ])

    mod_wtr.writerow(['1 ordem',  kmod, Tmod, Lmod, taumod, R2mod, IAEmod, ITAEmod, RMSEmod])
    mod_wtr.writerow(['Rejeitado', kmod, Tmod_, Lmod, taumod_, R2mod_, IAEmod_, ITAEmod_, RMSEmod_])
    mod_wtr.writerow(['2 ordem', kmod, T, Lmod, taumod2, R2mod2, IAEmod2, ITAEmod2, RMSEmod2])

# Exibir a tabela de dados dos modelos
df = pd.read_csv('modelos.csv', index_col='Modelos')
print(df)

# Dica: Caso queira um dado especifico dos modelos, como por exemplo R2, faça
# >>>> df.R2





