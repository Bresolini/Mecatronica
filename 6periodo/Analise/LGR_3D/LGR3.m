clc
close

numG = 1;
denG = [1 3 1];
kpmax = 5;
kmax = 5;
dk = 0.25;
type = 2;

[kc,ke,kin] = LGR3(numG,denG,type,dk,kpmax,kmax);

function [kcrit,kest,kinst] = LGR3(numG,denG,tipo,dk,kpmax,Tmax)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=====================================================================
%                            TOPOLOGIA PD
%=====================================================================
%                           C(s)
%                    -----------------         ----------------
%         +         |                 |       |                |       Y(s)
% R(s)----->.------>|    Td*s + kp    |------>|      G(s)      |---.--->
%           | -     |                 |       |                |   |
%           |        -----------------         ----------------    |
%           |                                                      |
%           |                                                      |
%           |                                                      |
%           |                                                      |
%            ------------------------------------------------------
%=====================================================================
%                            TOPOLOGIA PI
%=====================================================================
%
%                           C(s)
%                    -----------------         ----------------
%         +         |                 |       |                |       Y(s)
% R(s)----->.------>|  1/(Ti*s) + kp  |------>|      G(s)      |---.--->
%           | -     |                 |       |                |   |
%           |        -----------------         ----------------    |
%           |                                                      |
%           |                                                      |
%           |                                                      |
%           |                                                      |
%            ------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% função transfercênia dada
Gp = tf(numG,denG);
% declaração inicial das variáveis (necessária para serem salvas como
% variáveis globais)

kcrit = [0, 0];
kest  = [0, 0];
kinst = [0, 0];
cor = rand(length(denG),3);
kp = (0.1:dk:kpmax)';
hold on

if (tipo == 1)
    % Variando ki = Ti*kp
    % Nota: a variação em dk em ki não fará com que Ti varie com este mesmo
    % intervalo, mas sim com dk*kp.
    for ki = 0.1: dk/20 :Tmax/kpmax/2
        C1 = tf([ki 1],[ki 0]);
        G = Gp*C1;

        r = rlocus(G,kp); p = r';
        Ti = ki./kp;
        szP = size(p);
        for ii = 1:szP(2)
            plot3(real(p(:,ii)), imag(p(:,ii)), Ti,'Color',cor(:,ii));
        end

        % Obtendo os dados de kp  e ki ou kd que resultam no sistema critico,
        % estável e instável 
        for ii = 1:szP(1)
            if (sum(p(ii,:)>0)>0)      kinst = [kinst; kp(ii), Ti(ii)];
            elseif (sum(p(ii,:)==0)>0) kcrit = [kcrit; kp(ii), Ti(ii)];
            else                       kest  = [kest;  kp(ii), Ti(ii)];
            end
        end 
    end
    zlabel('Ti');
else
    % Variando ki = Ti*kp
    % Nota: a variação em dk em ki não fará com que Ti varie com este mesmo
    % intervalo, mas sim com dk*kp.
    for kd = 0.1: dk/20 :Tmax/kpmax
        C2 = tf([kd 1],1);
        G = Gp*C2;

        r = rlocus(G,kp); p = r';
        Td = kd.*kp;
        szP = size(p);
        for ii = 1:szP(2)
            plot3(real(p(:,ii)), imag(p(:,ii)), Td,'Color',cor(:,ii));
        end

        % Obtendo os dados de kp  e ki ou kd que resultam no sistema critico,
        % estável e instável 
        for ii = 1:szP(1)
            if (sum(p(ii,:)>0)>0)      kinst = [kinst; kp(ii), Td(ii)];
            elseif (sum(p(ii,:)==0)>0) kcrit = [kcrit; kp(ii), Td(ii)];
            else                       kest  = [kest;  kp(ii), Td(ii)];
            end
        end 
    end
    zlabel('Td');
end

% Eliminando os primeiros valores setados (linha 1)
kinst = kinst(2:end,:);
kest  = kest (2:end,:);
kcrit = kcrit(2:end,:);

% configurações gráficas
title('LGR 3D')
xlabel('Real');
ylabel('Imaginário');
grid on;
a=gca;a.GridLineStyle='--';
end