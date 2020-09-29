%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODELAGEM BALL AND BEAM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close

syms Jb JB mB mb R l g r alf dr dalf real 

%% Energia cinética
K1 = 0.5 * ( mB + JB/R^2 )*dr^2 + 0.5 * (mB*r^2)*dalf^2;
K2 = 0.5 * ( Jb + JB) * dalf^2;
K  = K1 + K2;

%% Energia potencial
P1 = 0.5*l*mb*g*sin(alf);
P2 = mB*g*r*sin(alf);
P  = P1 + P2; 

%% Lagrange

q = [r; alf];
L = K - P;

