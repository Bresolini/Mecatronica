clc
hold off

%% Obter dados
% t = ScopeData.time;
% y = ScopeData.signals(1).values;
% u = y(:,1);
% y = y(:,2);

%% Identificacao de parametros
yss = y(end); uss = u(end);
[ymax, tOS] = max(y);

k = yss/uss;
OS = ( ymax-yss )/yss;

z = -log(OS)/sqrt(pi^2 + log(OS)^2);

for i = tOS+1:length(y)
    if (y(i) < y(i+1))
        break
    end
end
for j = i:length(y)
    if (y(j) > y(j+1))
        break
    end
end

tOS = t(tOS); t1 = t(i); t2 = t(j);
T = t2 - tOS;

w0 = 2*pi/(T*sqrt(1-z^2));

G2 = tf(k*w0^2,[1 2*z*w0 w0^2]);
[Ym, Tm] = step(G2,[0:0.001:30]);
clf
plot(Tm,Ym,'LineWidth',4);
hold on;
plot(t,y,'LineWidth',4);
grid on; a=gca; a.GridLineStyle = '--';
a.FontSize = 16;
xlabel('tempo (s)');
ylabel('ângulo (rad)');
legend({'Aproximado','Real'},'Location','SouthEast')

