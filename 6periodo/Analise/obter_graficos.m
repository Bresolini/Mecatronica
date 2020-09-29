clc

%% Obtendo dados
t = ScopeData.time;
y = ScopeData.signals(1).values;
r = y(:,1);
y1 = y(:,2); y2 = y(:,3);
u = ScopeData.signals(2).values;
u1 = u(:,1); u2 = u(:,2);

Gc = 4411/1231;

%% Plotando
subplot(2,1,1);
p = plot(t,[r,y1,y2],'LineWidth',4);
grid on; a=gca; a.GridLineStyle = '--';
a.FontSize = 16;
xlabel('tempo (s)');
ylabel('$\theta$ (rad)','Interpreter','latex');
legend({'r(t)', 'Não linear','Linear'},'Location','SouthEast')

subplot(2,1,2);
p = plot(t,[u1,u2],'LineWidth',4);
grid on; a=gca; a.GridLineStyle = '--';
a.FontSize = 16;
xlabel('tempo (s)');
ylabel('$u(t) = F(t)$ (N)','Interpreter','latex');
legend({'Não linear','Linear'},'Location','SouthEast')

