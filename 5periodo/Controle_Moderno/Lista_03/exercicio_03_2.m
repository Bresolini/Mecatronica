%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Lista 03 de Controle       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

syms t lambda1 lambda2
A = [-2.068, -2; 2, 0];
B = [1; 0];
C = [1, -0.4];
D = 0;
uop = 30;
yop = 20;
v = eig(A);

f1 = exp(lambda1*t);
f2 = exp(lambda2*t);
f = [f1; f2];
L = [lambda1, 1; lambda2, 1];

H = rref([L, f]);
beta = H(:, length(A)+1);
h = beta(1)*A + beta(2)*eye(size(A));

u0a10  = 35;
u10a20 = 25;
u20a   = 30;

t0 = 0; t1 = 10; t2 = 20; t3 = 30;
dt = 0.001;
tt1 = [t0:dt:t1]';
tt2 = [t1+dt:dt:t2]';
tt3 = [t2+dt:dt:t3]';
tt = [tt1;tt2;tt3];

a = -517/500;
b = sqrt(732711)/500;
alf1 = exp(a*t)*cos(b*t);
bet1 = exp(a*t)*sin(b*t);
alf2 = exp(a*(t-10))*cos(b*(t-10));
bet2 = exp(a*(t-10))*sin(b*(t-10));
alf3 = exp(a*(t-20))*cos(b*(t-20));
bet3 = exp(a*(t-20))*sin(b*(t-20));

yr1 = (alf1*b+bet1*(5-a))/b - 1;
yr2 = yr1 - 2*(alf2*b+bet2*(5-a))/b + 2;
yr3 = yr2 + (alf3*b+bet3*(5-a))/b - 1;

yyr1 = double(subs(yr1,t,tt1));
yyr2 = double(subs(yr2,t,tt2));
yyr3 = double(subs(yr3,t,tt3));
yyr = [yyr1;yyr2;yyr3] + 20;

p = plot(tt,yyr);
p.LineWidth = 2;
xlabel('$t \:\mathrm{(s)}$','Interpreter','latex');
ylabel('$y_\mathrm{real} \:\mathrm{(psi)}$','Interpreter','latex');
grid on;
graf = gca
graf.GridLineStyle = '--';
%graf.YLim = [18.5 21.5];
graf.FontSize = 12;


% y1 = (uop-u0a10)*( C/A*B - C*expm(A*t)/A*B );
% y2 = y1 + (u0a10-u10a20)* (C/A*B - C*expm(A*(t-10))/A*B );
% y3 = y2 + (u10a20-u20a)*( C/A*B - C*expm(A*(t-20))/A*B );
% 
% y1 = double(subs(y1,t,tt1)) + yop;
% y2 = double(subs(y2,t,tt2)) + yop;
% y3 = double(subs(y3,t,tt3)) + yop;
% 
% y = [y1;y2;y3];
% 
% p = plot(tt,y);
% p.LineWidth = 2;
% xlabel('$t \:\mathrm{(s)}$','Interpreter','latex');
% ylabel('$y_\mathrm{real} \:\mathrm{(psi)}$','Interpreter','latex');
% grid on;
% graf = gca
% graf.GridLineStyle = '--';
% %graf.YLim = [18.5 21.5];
% graf.FontSize = 12;
