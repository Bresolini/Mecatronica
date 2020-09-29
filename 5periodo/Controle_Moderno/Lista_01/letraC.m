clc
clear
close

syms u
syms h1 h2
syms g1 g2

%Valores do sistema
v_h2 = 0.5;
v_h1 = v_h2*(130/339)^2+v_h2;
v_u = sqrt( (6.78*sqrt(v_h1-v_h2) - 0.03)/(2.617*10^(-4) ) );

% v_h2 = 0.8;
% v_h1 = 0.917646;
% v_u = 93.6565;

%Equações
qi  = 2.617*10^(-4)*u^2 + 0.03;
q12 = 6.78*sqrt(h1-h2);
q0  = 2.6*sqrt(h2);
dh1 = ( qi - q12 ) / ( pi*h1^2 - 0.9*pi*h1 + 0.25*pi );
dh2 = ( q12 - q0 ) / (-(pi/9)*h2^2 - 2*pi*h2/45 + 221*pi/900 );

%Linearizaçao
A1 = jacobian([dh1,dh2],[h1,h2]); %Jacobiando
A2 = subs(A1, [u,h1,h2],[v_u,v_h1,v_h2]); %substituiçao das variaveis
A = double(A2)

B1 = jacobian([dh1,dh2],u);
B2 = subs(B1, [u,h1],[v_u,v_h1]);
B = double(B2);

C1 = [1 0]; %y = h1
C2 = [0 1]; %y = h2
D = 0;
x0 = [v_h1, v_h2];

linear1 = ss(A,B,C1,D);
linear2 = ss(A,B,C2,D);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               Modelo não linear              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opt = odeset('Abstol', 1e-6, 'Reltol', 1e-6);
%configuração da ODE (erros relativos e absolutos)
ta = 0:0.01:10;
tb = 10.01:0.01:20;
y0 = [v_h1, v_h2];
[t1,y1] = ode23(@(t1,y1) odefcn(t1,y1,v_u*1.02), ta, y0, opt);
[ly1,cy1] = size(y1); %quer-se o último ponto de y1
[t2,y2] = ode23(@(t2,y2) odefcn(t2,y2,v_u*0.95), tb, y1(ly1,:), opt);

tnl = [t1; t2];
ynl = [y1; y2];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Modelo linear                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt = 0:0.01:20;
J = tf(1);
opt1 = stepDataOptions('StepAmplitude',v_u*1.02-v_u); % + 2%
opt2 = stepDataOptions('StepAmplitude',v_u*0.95-v_u); % - 5%
u1 = step(J, opt1, ta); %configurando u1
u2 = step(J, opt2, tb); %configurando u2
u = [u1; u2];
yl =  lsim(linear1, u, tt) + v_h1;
yl2 = lsim(linear2, u, tt) + v_h2;
%p = plot(tnl, ynl(:,1), '-', tt, yl, '-'); 
%p = plot(tnl, ynl(:,2), '-', tt, yl2, '-');
p = plot(tt,yl2);
p.LineWidth = 5;

ax = gca; % current axes
ax.FontSize = 14;
%title('Respostas de h_1(t) variando u em +2% e -5% do u_{eq}');
xlabel('Tempo (s)');
ylabel('Nível de água (m)');
legend('$h_2(t)$', 'Interpreter', 'latex')
%legend('h_1(t) não linear', 'h_1(t) linear')
grid on;
ax.GridLineStyle = '--';


function dydt = odefcn(t,y,u)
dydt = zeros(2,1);
dydt(1) = ( 2.617*10^(-4)*u^2 + 0.03 - 6.78*sqrt(y(1)-y(2)) ) / ( pi*y(1)^2 - 0.9*pi*y(1) + 0.25*pi );
dydt(2) = ( 6.78*sqrt(y(1)-y(2)) - 2.6*sqrt(y(2)) ) / (-(pi/9)*y(2)^2 - 2*pi*y(2)/45 + 221*pi/900 );
end
