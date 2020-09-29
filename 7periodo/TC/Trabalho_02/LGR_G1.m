%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          CONTROLADOR POR COMPENSADOR DE ATRASO DE FASE VIA LGR              
%
% Autores: Bernardo Bresolini e Ester Alvarenga
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
close all
clear all

%% Parâmetros de projeto
ts = 1;          % s
OS = 13.7/100;   % %/100


z  = -log(OS)/sqrt(log(OS)^2 + pi^2);
%zw = -log(0.02*sqrt(1-z^2))/ts;
zw = 4/ts;  % Ábaco do OGATA, FIG. 5.11, p. 158 com folga pelo sys a ser controlador ser de 3 ordem
w  = zw/z;

%% Função Transferência G(s) do processo
numG1 = [0 5 2.5];
denG1 = [1 2 0];

G    = tf(numG1,denG1);
numG = G.num{1};
denG = G.den{1};
n = order(G);  % Ordem de G(s)

% Nota: como a planta já tem integrador, tem-se erro nulo de estado
% estacionário

%% LGR do sistema
% Aqui é analisado o comportamento da planta se ela fosse colocada em malha
% fechada com um ganho proporcional

rlocus(G);
sd = complex(-zw,w*sqrt(1-z^2)); % polo desejado
hold on
plot(sd,'*') % Posicionamento desejado do polo para atender ts

%% Cálculo da fase do sistema

zeros = zero(G);
Fnum = 0;
for i = 1:length(zeros)   % calculando fase do numerador
Fz = sd-zeros(i);
Fnum = Fnum + 180 - atan(imag(Fz(i))/real(Fz(i)))*180/pi;
end

polos = roots(denG1);
Fden = 0;
for i = 1:length(polos)   % calculando fase do denominador
Fp(i) = sd-polos(i);
Fden = Fden + 180 - atan(imag(Fp(i))/real(Fp(i)))*180/pi;
end

FG = Fnum - Fden;

%% Fase do compensador
if FG < 0
    FC = -180-FG;
else
    FC = 180-FG;
end    
%% ESCOLHA DO POLO 
% Como queremos que exista OS, devemos fazer os polos terem parte
% imaginária, portanto o zero deve ser maior que o polo


p = zw; % escolhendo o polo igual a zw

Fpp = sd + p;
FdenC = 180 - atan(imag(Fpp)/real(Fpp))*180/pi; % calcula-se a fase do neominador do compensador
% assim, a fase do numerador:
FnumC = FC - FdenC; 

% Posição do zero do compensador
%
%       . ----------|imag(sd)
%      /|           |
%     / |           |
%    /  |           |
%   /   |           |
%  /____|__________ |
% z  b  real(sd)

b = imag(sd)/tan(abs(FnumC)*pi/180);
z = p + b;
numC = [1 z];
denC = [1 p];

%% Achando K do compensador 

kc = 1/( ( abs(sd) + numC(2) )/( abs(sd) + p )*(5*( abs(sd)+0.5 )/( abs(sd)*(abs(sd) + 2) )  ) );

%% LGR do sistema compensado
figure(2)
comp = kc*tf(numC,denC); %Compensador final
rlocus(G*comp);
hold on
plot(sd,'*')
plot(conj(sd),'*')


%% Malha fechada
MFc = minreal(feedback(comp*G,1)); % sistema compensado
MF = minreal(feedback(G,1));       % sistema orignial

%% Comparação das respostas
figure(3)
t = 0:0.05:10;
c1 = step(MFc,t);
c = step(MF,t);
plot(t,c1,'-',t,c,'LineWidth',3);
grid
title('Resposta ao degrau unitário');
xlabel('t (s)');
ylabel('Saídas');
legend('Sistema compensado','Sistema sem compensação');
grid on;
a=gca;
a.GridLineStyle = '--';