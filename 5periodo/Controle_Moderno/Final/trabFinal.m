%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Modelagem e Linearização do Servomotor    % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc
close

%% Constantes e coeficientes do motor
kf = 8;
ke = 144;
m = 5.3;
R = 1.5;
f1 = 0.0064;
f2 = 8.2826e-5;
delF = 0.381;
a = -(kf*ke)/(m*R);
b = kf/(m*R);

%% Ponto de operação
% Escolheu-se trabalhar em x = 0, dx = 0
% cuja relação física implica que ddx = 0

x0 = [0;0]; %condição inicial
vop = delF/(m*b); % 0,0714

%% Espaço de Estados

A = [ 0,           1;
      0, a - 20*f1/m];

B = [ 0;
      b];

Cp = [ 1,  0];
Cv = [ 0,  1];
D = 0;

Pos = ss(A,B,Cp,D);
Vel = ss(A,B,Cv,D);

%% Variação do Ponto de Operação
var1 = 115; % em porcentagem
var2 = -40; % em porcentagem
t1 = 1; t2 = 2; dt = 0.0001;
tt1 = [0:dt:t1]'; tt2 = [t1+dt:dt:t2]';
v1 = vop*(1 + var1/100);
v2 = vop*(1 + var2/100);

%% Resposta do Sistema não linear

options = odeset('Abstol', 1e-9, 'Reltol', 1e-9);
[tnL1, xnL1] = ode23(@(t,x) edo(t,x,v1), tt1, x0, options);
[tnL2, xnL2] = ode23(@(t,x) edo(t,x,v2), tt2, xnL1(end,:), options);
ttnL = [tnL1;tnL2];
xxnL = [xnL1;xnL2];

P = plot(ttnL,xxnL);
legend('Posição (m)', 'Velocidade (m/s)');
P(1).LineWidth=5; P(2).LineWidth=5;
grid on; aa = gca; aa.GridLineStyle = '--';
xlabel('tempo (s)');
title({'Resposta do sistema não linear à variação do ponto de operação', ''});

%% Resposta do Sistema linear
vv = [v1*ones( size(tt1) ); v2*ones( size(tt2) )] - vop;
yL1 = lsim(Pos, vv, [tt1;tt2]);
yL2 = lsim(Vel, vv, [tt1;tt2]);
yL = [yL1, yL2];
P = plot([tt1;tt2], yL);
legend('Posição (m)', 'Velocidade (m/s)');
P(1).LineWidth=5; P(2).LineWidth=5;
grid on; aa = gca; aa.GridLineStyle = '--';
xlabel('tempo (s)');
title({'Resposta do sistema linear à variação do ponto de operação', ''});

%% Comparação: NÃO LINEAR x LINEAR (POSIÇÃO)
P = plot([tt1;tt2], [xnL1(:,1);xnL2(:,1)], [tt1;tt2], yL1);
legend('Não Linear', 'Linear');
P(1).LineWidth=5; P(2).LineWidth=5;
grid on; aa = gca; aa.GridLineStyle = '--';
xlabel('tempo (s)');
ylabel('Posição (m)');
title('Comparação entre o sist. linear e o não linear');

%% Comparação: NÃO LINEAR x LINEAR (VELOCIDADE)
P = plot([tt1;tt2], [xnL1(:,2);xnL2(:,2)], [tt1;tt2], yL2);
legend('Não Linear', 'Linear');
P(1).LineWidth=5; P(2).LineWidth=5;
grid on; aa = gca; aa.GridLineStyle = '--';
xlabel('tempo (s)');
ylabel('Velocidade (m/s)');
title('Comparação entre o sist. linear e o não linear');
aa.FontSize = 18;


%% Função Transferência
gp = tf(Pos);
gv = tf(Vel);

%% Estabilidade
p = pole(gp);
neg = 0; %auxiliar
stable = true; %estabilidade
for i = 1:length(p)
    if (p < 0)
        neg = neg+1;
    else
        stable = false;
        break
    end
end
if (stable == true)
    disp('O SISTEMA É ESTÁVEL!');
else
    disp('O SISTEMA É INSTÁVEL!');
end

%% Resposta à função degrau
u = 1/10 - vop;
ydegrau = lsim(Pos,u*ones( size([tt1;tt2]) ),[tt1;tt2]);
P = plot([tt1;tt2],ydegrau);

%% Configurações do plot
aa = gca;
P(1).LineWidth = 5;
P(2).LineWidth = 5; 
grid on;
aa.GridLineStyle = '--';
aa;FontSize = 18;
xlabel('tempo (s)');
ylabel('Amplitude');

%% EDO
function dx = edo(t, x, v)
kf = 8; ke = 144;
m = 5.3;
R = 1.5;
f1 = 0.0064; f2 = 0; delF = 0.381;
a = -(kf*ke)/(m*R);
b = kf/(m*R);

dx = zeros(2,1);
dx(1) = x(2);
dx(2) = a*x(2) + b*v - (f1*sign( x(2) ) + f2*x(2) + delF)/m;
end