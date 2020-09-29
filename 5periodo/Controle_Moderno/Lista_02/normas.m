clc
clear
close

syms x1 x2 x3;

eq1 = sqrt(x1^2) + sqrt(x2^2) == 1;
eq1 = solve(eq1,x1);

xx1 = linspace(-1,1,100);
xx2 = linspace(-1,1,100);
p = plot(xx1, subs(eq1,x2,xx2) );
p(1).LineWidth = 3;
p(2).LineWidth = 3;
p(1).Color = 'b';
p(2).Color = 'b';
xlabel('$x_1$', 'Interpreter','latex');
ylabel('$x_2$', 'Interpreter','latex');
hold on;

eq2 = x1^2 + x2^2 == 1;
eq2 = solve(eq2,x1);
xx1 = linspace(-1,1,100);
xx2 = linspace(-1,1,100);
p = plot(xx1, subs(eq2,x2,xx2) );
p(1).LineWidth = 3;
p(2).LineWidth = 3;
p(1).Color = 'y';
p(2).Color = 'y';

plot([-1 1 1 -1], [-1 -1 1 1])

xlabel('$x_1$', 'Interpreter','latex');
ylabel('$x_2$', 'Interpreter','latex');
grid on
ax=gca;
ax.GridLineStyle = '--';
ax.XGrid = 2;
ax.YGrid = 2;
