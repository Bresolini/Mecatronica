clc

%% Obtendo os dados
% t = ScopeData.time;
% y = ScopeData.signals(1).values;
% u = y(:,1);
% y = y(:,2);

%% Modelo do processo
G = tf(k*1.4^2, [1, 2*1.4*0.19, 1.4^2]);
rlocus(G);
