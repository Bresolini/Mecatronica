# Controle digital via alocação de polos em espaço de estados
O projeto apresentado desenvolve 3 controladores digitais por alocação de polos em espaços de estados.
A topologia utilizada foi a topologia servo tipo 1 com observador.

A implementação é feita em Python3 utilizando as bibliotecas NumPy, Matplotlib e Control nas suas versões
mais atuais (08/04/2021).

## Sistema 1
Sistema de primeira ordem com atraso. O atraso foi tratado com o uso de Preditor de Smith e a aproximação
linear de Padè de 8ª ordem.

         1
G1 = --------- exp(-2,8*s)
      s + 0,1

Requer-se: ts = 20 s e OS = 0%
Utilizou-se: Ts = 1 s.

## Sistema 2
Sistema de terceira ordem com 1 par de polos complexos e 1 polo em -2. 
                 1
G2 = -------------------------
     (s + 2)(s² + 0,2s + 0,65)

Requer-se: ts = 10 s e OS = 10%
Utilizou-se: Ts = 0,9 s.

## Sistema 3
Sistema de segunda ordem com 1 polo e 1 zero de fases não mínimas.
        5 (s - 5)
G3 = ----------------
      (s + 2)(s - 2)

Requer-se: ts = 20 s e OS = 10%
Utilizou-se: Ts = 1 s.

Autores
-------
Bernardo Bresolini

Ester Q. Alvarenga.
