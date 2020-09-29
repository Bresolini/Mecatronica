%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Lista 03 de Controle       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

syms lambda t
A = [   -1,    1,    2,    2.5;
       0.4, -0.5, -1.6,   -1.2;
     0.175, 0.25, -1.2, -0.275;
      -0.4,    0,  1.6,    0.7];

% A = [-1.5,    0,    0,   1.25;
%       0.4, -0.5, -1.6,   -1.2;
%       0.3,  0.5, -0.7, 0.0375;
%      -0.4,    0,  1.6,    0.7];

I = eye(4);
pol = det(A - lambda*I);

f = exp(lambda*t);
L = [lambda^3, lambda^2, lambda, 1];
L = [L; diff(L); diff(L,2); diff(L,3)];
E = [f; diff(f,lambda); diff(f,lambda,2); diff(f,lambda,3)];

H = rref([L, E]);
beta = H(:,5);
beta = subs(beta, lambda, -0.5);
h = beta(1)*A^3 + beta(2)*A^2 + beta(3)*A + beta(4)*eye(4);
Ee = expm(A*t);

