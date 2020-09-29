%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             AVALIAÇÃO DA MODELAGEM DO SISTEMA BALL AND BEAM             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

g = 981;              %%% cm/s²
d = 10.15;            %%% cm
L = 40;               %%% cm

mB = 3e-3;            %%% kg
R  = 2;               %%% cm
JB = 2*mB*R^2/3;      %%% kg*cm^2

a = mB + JB/R^2;
b = mB*d^2/L^2;
c = mB*g*d/L;


