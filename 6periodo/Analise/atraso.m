clc; clear all; close all

%% Equaçao

syms t d k tau y(t)

y(t) = k*(1 - exp( -(t-d)/tau ));

%% Simulaçao

d_ = 1;
tau_ = 2;
k_ = 1;

ti = 0; tf = 10; dt = 0.001;
tt = (0:dt:tf)';

yy = subs(y,[d k tau],[d_ k_ tau_]);
yy = double ( subs(yy, t, tt) );
uu = ones(size(tt));
uu(1) = 0;

p = plot(tt,[yy, uu]);
a = gca;
a.XLim = [ti,tf];
a.YLim = [0,1.2];
p(1).LineWidth = 5;
p(2).LineWidth = 5;
grid on;
a.GridLineStyle = '--';
a;FontSize = 18;
xlabel('tempo (s)');

%% Curvas
clc
clear all
close all
hold on

k = 0.8;
d = 1;
num = k;
den = [1 1.96 1]; 
G = tf(k,den,'IODelay',d)

[Y,T] = step(G,10);
subplot(2,1,1);
p = plot(T,Y);
ylabel('$y(t)$', 'Interpreter','latex','FontSize',18)
title('Gráfico da resposta y(t)','FontSize',14)
hold on
a = gca;
a.XLim = [0,10]; a.XTick = [0:2:10];
a.YLim = [0,1];  a.YTick = [0:0.2:1];
p(1).LineWidth = 5;
p(1).Color = [1.00, 0.41, 0.16];
grid on;
a.GridLineStyle = '--';
a;FontSize = 20;
i_k = ceil(0.95*length(T));
p2 = plot(T(i_k), Y(i_k),'.');
p2.MarkerSize = 30;
p2.Color = [0.39 0.83 0.07];


Z = log( (k-Y)/k )
subplot(2,1,2);
q = plot(T,Z);
hold on
i1_z = ceil(0.5*length(T));
i2_z = ceil(0.6*length(T));
a = gca;
a.XLim = [0,10]; a.XTick = [0:2:10];
a.YLim = [-8,0]; a.YTick = [-8:2:0];
q(1).LineWidth = 5;
q(1).Color = [0.30, 0.75, 0.93];
grid on;
a.GridLineStyle = '--';
a;FontSize = 20;
xlabel('tempo (s)', 'FontSize', 16);
ylabel('$z(t)$', 'Interpreter','latex','FontSize',18)
title('Gráfico de z(t)','FontSize',16)
pts = plot([T(i1_z), T(i2_z)], [Z(i1_z), Z(i2_z)], '.');
pts.MarkerSize = 30;
pts.Color = [1.0 0 0];

