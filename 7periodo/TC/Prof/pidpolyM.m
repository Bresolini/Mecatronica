%% Exemplo PID polinomial
% Sistema

clear all
num = 1500;           % Numerador = B(s)
den = [1 50.1 275];  % denominador = A(s)
G = tf(num,den); 

b0 = num; b1 = 0; b2 = 0;

a0 = den(3);
a1 = den(2); 
a2 = den(1); % que � normalizado em 1!

% Especifica Din�mica da Malha Fechada

% Especific�o de malha fechada
zetaomegan = 4; % \zeta \omega_n
OS = 6
OS = OS/100; %0.1; % sobressinal de 10%
zeta = -log(OS)/sqrt(pi^2+log(OS)^2); % %OS = 10%
omegan = zetaomegan/zeta;
% monta os polos dominantes
p1 = -zetaomegan + omegan*sqrt(zeta^2-1);
p2 = conj(p1);
% define terceiro polo de maneira a nao interferir
% na dinamica especificada
p3 = -10*zetaomegan;

% Constroi solu��o para o controlador PID

% monta o polin�mio D(s)
D = conv([1 -p3],conv([1 -p2],[1 -p1]));
disp('D = ')
disp(D);

%Recupera os coeficientes (s� para facilitar!)
d2 = D(2);
d1 = D(3);
d0 = D(4);

% Monta matriz E
E = [ b0-b1*d2 b1 0; -b1*d1 b0 b1; -b1*d0 0 b0];
% Monta matriz B
B = [ d2-a1; d1-a0; d0];
% Calcula a solu��o
C = E\B

% Recupera os coeficientes de E (s� para facilitar)
c0 = C(3);
c1 = C(2);
c2 = C(1);

% Converte os coeficientes nos ganhos do PID
K = c1;
Td = c2/K;
Ti = K/c0;
% Monta controlador

% Fun��o de transfer�ncia do controlador:
Gc = tf([c2 c1 c0],[1 0])

% Monta Malha Fechada e aplica degrau unit�rio

MF = feedback(series(Gc,G),1);
step(MF)
grid;
S = stepinfo(MF)
%% Reposta em Frequ�ncia
% Sa�da - refer�ncia

figure(2)
bode(MF);
grid;
title('Rela��es sa�da-referencia')
% Sinal de controle-refer�ncia

figure(3);
bode(feedback(Gc,G));
grid;
title('Rela��es sinal de controle-referencia')