clc
close

numG = 1;
denG = [1 3 1];
kpmax = 5;
kmax = 5;
dk = 0.2;
type = 1;

[kcrit,kest,kint] = LGR3(numG,denG,type,dk,kpmax,kmax);




function [kcrit, kest, kint] = LGR3(numG,denG,tipo,dk,kpmax,Tmax)

Gp = tf(numG,denG);
szGp = length(denG);
cor = rand(szGp,3);
kcrit = [ 0, 0];
kest  = [ 0, 0];
kint  = [ 0, 0];
kp = (0.01:dk:kpmax)';

if(tipo == 1)
    for Ti = 0.1: dk*kp/3 :Tmax/kpmax/2
        C1 = tf([Ti 1],[Ti 0]);
        G = Gp*C1;
        for kp = 0.1:dk:kpmax
            r = rlocus(G,kp);
            p = r';
            hold on
            size = length(p);
            ki = ones(1,size)*Ti/kp;

            if    ( real(p) < 0 ) kest  = [kest;  kp, ki(1)];
            elseif( sum(real(p) > 0) >= 1 ) kint  = [kint;  kp, ki(1)];
            else                  kcrit = [kcrit; kp, ki(1)];
            end

            for i = 1:size
                plot3(real(p(i)), imag(p(i)), ki,'*','Color',cor(i,:));
                xlabel('Real');
                ylabel('Imaginário');
                zlabel('Ti');

            end
        end
    end
    kcrit = kcrit(2:end,:);
    kest  = kest (2:end,:);
    kint  = kint (2:end,:);
else
    for Td = 0.1:dk/kp:Tmax/kpmax
        for kp = 1:dk:kpmax
            C2 = tf([Td 1],[1]);
            G = Gp*C2;

            r = rlocus(G,kp);
            p = [r'];
            hold on
            size = length(p);
            kd = ones(1,size)*Td*kp;

            if    ( real(p) < 0 ) kest  = [kest;  kp, kd(1)];
            elseif( sum(real(p) > 0) >= 1) kint  = [kint;  kp, kd(1)];
            else                  kcrit = [kcrit; kp, kd(1)];
            end

            for i = 1:1:size
                plot3(real(p(i)), imag(p(i)), kd,'*','Color',cor(i,:));
                xlabel('Real');
                ylabel('Imaginário');
                zlabel('Td');
            end
        end
    end
    kcrit = kcrit(2:end,:);
    kest  = kest (2:end,:);
    kint  = kint (2:end,:);
end


% configurações gráficas
title('LGR 3D')
grid on;
a=gca;a.GridLineStyle='--';
end