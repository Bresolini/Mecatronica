clc
close all
clear all

%% Critérios de desempenho
ts = 10;   % s
OS = 0.2; % porcentagem/100

%% Constantes físicas
mB = 3e-3;     % kg
d  = 10.14;    % cm
l  = 40;       % cm
g  = 981;      % cm/s^2
R  = 2;        % cm
JB = 8e-3;     % kg*cm^2

A = g*d*R^2/( l*(R^2 + JB/mB) )

%% polos
z = -log(OS)/sqrt( pi^2 + log(OS)^2 );
w = 4/(ts*z);
re   = 4/ts;
imag = w*sqrt(1-z^2); 

%% denominador desejado

den = conv( [1 10*re], conv([1 re+1j*imag],[1 re-1j*imag]) );

%% Controlador PD

p  =  den(2);
kp = -den(4)/(A*p);
kd = -den(3)/A - kp;

numC = [kp+kd,  kp*p];
denC = [1,  p];
C = tf(numC,denC);

numH = [-A*(kp+kd),  -A*kp*p];
denH = [ 1,  p,  -A*(kp+kd),  -A*kp*p];
H = tf(numH,denH);