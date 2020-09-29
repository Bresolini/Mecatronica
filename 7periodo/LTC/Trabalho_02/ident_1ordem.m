%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  VALIDAÇÂO DO MODELO de PRIMEIRA ORDEM                  %
%                                                                         %
% Autores: Bernardo Bresolini, Ester Q. Alvarenga                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all

% Dados de 1 ordem = 1
% Dados de 3 ordem caso contrário
escolha = 1;

% Carregando os dados da planta
if (escolha == 1)
    load('PrimeiraOrdem.mat');
else
    load('TerceiraOrdem.mat');
end


%% Identificacao de parametros

% Variáveis auxiliares
if (escolha == 1)
    t0 = 7.5;  % tempo no qual foi feita a variação
else
    t0 = 15;   % tempo no qual foi feita a variação
end
tinf = t1(end);     % tempo final
szY  = length(y1);  
D    = ceil(t0/tinf*szY);

% Definindo o intervalo válido
t = t1(D:end)-t0;
y = y1(D:end)-3;
u = u1(D:end)-3;

% Ganho do sistema
K = y(end)/u(end);
% valor final de y
yss = y(end);     

% Obtenção do ponto no qual y(t) = 0,63K
B = 2;
for i = 1:length(y)
    if( (0.63*yss - y(i)) < 0.001 )
        B = i;
        break;
    end
end

% Parâmetros da reta tangente
a1 = ( y(B) - y(B-1) )/(t(B)-t(B-1));
a0 = y(B) - a1*t(B);

% Caso o valor seja negativo, o sistema tem atraso 0.
A = -a0/a1;
if(A<0)
    A=0;
end
C = (yss-a0)/a1;

% Equação da reta
Y = a1*t + a0;

plot(t,[y Y],'LineWidth',2);

% Configurações gráficas
grid on
a=gca;
a.GridLineStyle = '--';
a.XLim = [0 tinf-t0]; a.YLim = [0, 0.6];
xlabel('$t$ (s)','Interpreter','latex');
ylabel('$v$ (V)','Interpreter','latex');

legend({'Real','Reta tangente'},'Interpreter', ...
       'latex','FontSize',14,'Location','SouthEast')

if (escolha == 1)
    print -depsc reta_tangente_1.eps;
else
    print -depsc reta_tangente_3.eps;
end

% Parâmetros restantes
L  = A;
T1 = C-A;
T2 = t(B)-A;

%% Modelos de primeira ordem obtidos
figure()

% Definição das funções transferências
G1 = tf(K,[T1 1],'IODelay',L)
G2 = tf(K,[T2 1],'IODelay',L)

% Step de 0,5
opt = stepDataOptions('StepAmplitude', 0.5);

[yy1] = step(G1,t,opt); % T = AC
[yy2] = step(G2,t,opt); % T = BC
uu  = 0.5*ones(size(yy1)); % sinal de controle

if (escolha == 1)
    % Primeira ordem
    plot(t(2:end),[uu, y(2:end),yy1,yy2],'LineWidth',2);
else
    % Terceira ordem
    plot(t,[uu, y,yy1,yy2],'LineWidth',2);
end

grid on
a=gca;
a.GridLineStyle = '--';
a.XLim = [0 tinf-t0];
xlabel('$t$ (s)','Interpreter','latex');
ylabel('$v$ (V)','Interpreter','latex');

legend({'$u(t)$', 'Real','$T = AC$','$T=AB$'}, ...
    'Interpreter','latex','FontSize',14,'Location','SouthEast')

if (escolha == 1)
    print -depsc modelo_1ordem.eps;
else
    print -depsc modelo_3ordem.eps;
end

%% Remoção de dados desnecessários
clear a A a0 a1 B C D i K L opt ScopeData1
clear szY t t0 t1 T1 T2 tinf u u1 uu y Y y1 y22 yss yy1 yy2

if (escolha == 1)
    save -mat modelos_1.mat
else
    save -mat modelos_3.mat
end
