%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               ABORDAGEM POLINOMIAL em SISTEMAS CONTÍNUOS                %
%
% Autor: Bernardo Bresolini
% e-mail: berbresolini14@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
close all
clear all

%% Parâmetros de projeto
ts = 1;       % s
OS = 0.1/100;   % %/100

zw = 4/ts;
z  = -log(OS)/sqrt(log(OS)^2 + pi^2);
w  = zw/z;

%% Polos
n = 2;

p    = zeros(1,2*n-1);
% Polos dominantes complexos conjugados
p(1) = z*w + 1j*w*sqrt(1-z^2);
p(2) = conj(p(1));

% Polos não dominantes
for i = 3:(2*n-1)
    p(i) = 10*zw + (i-3);
end
%% Polinômio Desejado
D = conv([1 p(1)],[1 p(2)]);
for i = 3:length(p)
    D = conv(D,[1 p(i)]);
end
%D = fliplr(D)';   % vetor coluna

num = 13/41*5*[1, 0.5];
den = conv([1 -2],[1 -1]);

%% Sylvester

syms a0 a1 b0 b1 real
syms d1 d2 d3

E = [b0, b1,  0;
      0, b0, b1;
      0,  0, b0;];

M = [d1-a1; d2-a0; d3];

beta = inv(E)*M

beta = subs(beta,[d1 d2 d3],[D(2:end)]);
beta = subs(beta,[b1 b0],num);
beta = double(subs(beta,[a1 a0],den(2:end)));

Kp = beta(2)
Td = beta(1)/Kp
Ti = Kp/beta(3)

G = tf(num,den);
C = tf([beta(3),beta(2),beta(1)],[1 0]);

H = minreal( feedback(C*G,1) )

%% Beta0 = 0
a1 = num(1); a0 = num(2);
b1 = den(1); b0 = den(2);
d1 = D(2); d2 = D(3); d3 = D(4);
beta1 = (d1-a1)/b1
beta2 = d3/b0;
beta11 = (d2-a0 - b1*beta2)/b0



