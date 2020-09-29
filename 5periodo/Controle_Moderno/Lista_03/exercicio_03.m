%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Lista 03 de Controle       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

syms tau
A = [-2.068, -2; 2, 0];
B = [1; 0];
C = [1, -0.4];
D = 0;
uop = 30;
yop = 20;

u  = 35;
x0 = [0;0];
tt = 0:0.1:10;

for i=1:length(tt)
y(i,1) = C*expm(A*tt(i))*x0 + C*int(expm(A*(tt(i)-tau))*B*(u-uop),tau,0,tt(i)) + yop;
end
y = real(double(y))
p = plot(tt,y);
p.LineWidth = 2;
grid on;
graf = gca;
graf.GridLineStyle = '--';