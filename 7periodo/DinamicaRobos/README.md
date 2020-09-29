# Disciplina obrigáorio de Dinâmica de Robôs (Renato Dâmaso)

A disciplina tem por objetivo ensinar a modelagem cinemática direta e inversa de manipuladores robóticos industriais.

## Cinemática Direta
A cinemática direta consiste em determinar a posição do _end-effector_ com base nas variáveis de juntas do robô.

Para tanto, o método clássico para determinar a matriz de transformação _T_ é a convenção de Denavit-Hartenberg (DH):
  - DH1 O eixo _x1_ deve ser perpendicular ao eixo _z0_
  - DH2 O eixo _x1_ deve interceptar o eixo _z0_

Se a atribuição de frames do robô seguir essas duas premissas, a matriz de transfomrção homogênea _Ai_ de cada link será dada por
        Ai = Rot(z,theta) Trans(z,d) Trans(x,a) Rot(x,alpha) 

## Cinemática Inversa
Consiste em determinar os ângulos de junta do manipulador para uma dada pose.
