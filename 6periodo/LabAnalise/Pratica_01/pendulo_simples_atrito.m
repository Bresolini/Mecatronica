%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Aula 1 - Lab. Análise        %
%     Pêmdulo Simples com atrito     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

syms th u

k = 1;   % Atrito viscoso
L = 0.5; % metros
m = 1;   % kg
g = 9.8; % m/s^2

A = [   0,    1;
     -g/L, -k/m; ];
B = [0; 1/m];
C = [1, 0];
D = 0;

xeq = 0; % metros
ueq = 0; % Nm

syms x
dx = A*x + B*u;

