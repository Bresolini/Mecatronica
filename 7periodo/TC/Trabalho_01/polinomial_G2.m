%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               ABORDAGEM POLINOMIAL em SISTEMAS CONTÍNUOS                %
%
% Autor: Bernardo Bresolini
% e-mail: berbresolini14@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
close all
%clear all

%% Parâmetros de projeto
ts = 1;         % s
OS = 0.1/100;   % %/100


z  = -log(OS)/sqrt(log(OS)^2 + pi^2);
%zw = -log(0.02*sqrt(1-z^2))/ts;
zw = 4.25/ts;  % Ábaco do OGATA, FIG. 5.11, p. 158
w  = zw/z;

%% Função Transferência G(s) do processo
numG2 = 5*conv([1, 6, 13],[1, 0.5]);   
denG2 = conv( [1 8 41], conv([1, -1], [1, -2]) );
G2    = tf(numG2,denG2);

% Pré-compensação de zeros e polos
numGp = [1, 8, 41];
denGp = [1, 6, 13];
Gp    = tf(numGp,denGp);
Gp    = Gp/dcgain(Gp); % Ganho unitário

G    = minreal(series(Gp,G2));
numG = G.num{1};
denG = G.den{1};
n = order(G);  % Ordem de G(s)

%% Polos

p    = zeros(1,2*n-1);
% Polos dominantes complexos conjugados
p(1) = z*w + 1j*w*sqrt(1-z^2);
p(2) = conj(p(1));

% Polos não dominantes
for i = 3:(2*n-1)
    p(i) = 10*zw + (i-3);
end
%p(end-1:end) = -roots([1, 6, 13]);       % Cortando com um zero do sistema

%% Polinômio Desejado
D = conv([1 p(1)],[1 p(2)]);
for i = 3:length(p)
    D = conv(D,[1 p(i)]);
end
D = fliplr(D)';   % vetor coluna


%% Matriz de Sylvester
E = sylv_mat(numG, denG);

%% Controlador
M = E\D;
a = flip( M(1:n) );
b = flip( M(n+1:end) );
C = tf(b',a');

%% Malha fechada H
H1 = minreal(feedback(C*G,1));      % Controlador em série
k1 = dcgain(H1);                    % Ganho dc de H1
H1 = H1/k1                          % Dividindo pelo ganho dc
H2 = minreal(feedback(G,C));        % Controlador em parelelo
k2 = dcgain(H2);                    % Ganho dc de H2
H2 = H2/k2                          % Dividindo pelo ganho dc

%% Informações
s1 = stepinfo(H1);
s2 = stepinfo(H2);

%% Compensação de zeros
figure();
t = [0:0.01:2*ts];

% Compensa os zeros de fase mínima
zH1 = zero(H1); % Armazenando os zeros
denGc1 = 1;     % Denominador inicial de Gc1
for i = 1:length(zH1)
    if (zH1(i) < 0)  % Verifica se o zero é de fase mínima
        denGc1 = conv(denGc1, [1 -zH1(i)]);
    end
end
Gc1 = tf([1 p(end)],denGc1); Gc1 = Gc1/dcgain(Gc1)
Hc1 = minreal( series(Gc1,H1) );
yc1=step(Hc1,t);

% Compensa os zeros de fase mínima
zH2 = zero(H2); % Armazenando os zeros
denGc2 = 1;     % Denominador inicial de Gc2
for i = 1:length(zH2)
    if (zH2(i) < 0)  % Verifica se o zero é de fase mínima
        denGc2 = conv(denGc2, [1 -zH2(i)]);
    end
end
Gc2 = tf([1 p(end)],denGc2); Gc2 = Gc2/dcgain(Gc2)
Hc2 = minreal( series(Gc2,H2) );
yc2=step(Hc2,t);

%% Gráfico
y1=step(H1,t); y2=step(H2,t);
plot(t,[y1, y2, yc1,yc2],'LineWidth',3);
xlabel('Tempo (s)');
ylabel('Amplitude');

grid on;
a=gca;
a.GridLineStyle = '--';
legend({'$C(s)$ em serie','$C(s)$ em paralelo', ...
        '$C(s)$ em serie compensado', '$C(s)$ em paralelo compensado'}, ...
        'Location','Southeast', 'Interpreter','latex','FontSize',14);


%% Função para fazer a matriz de Sylvester
function S = sylv_mat(A,B)
% Desenvolvida por XUE, Dingyu & CHEN, YangQuan no livro 
% Scientific Computing with MATLAB, Cap. 4. p. 164.

n = length(B)-1; m=length(A)-1; S = [];
A1 = [A(:); zeros(n-1,1)]; B1 = [B(:); zeros(m-1,1)];
for i=1:n, S=[S A1]; A1=[0; A1(1:end-1)]; end
for i=1:m, S=[S B1]; B1=[0; B1(1:end-1)]; end
S = fliplr(flip(S));
end

