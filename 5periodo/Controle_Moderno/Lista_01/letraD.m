clc
clear
close

syms u
syms h1 h2
syms g1 g2

%Valores do sistema
v_h2 = 0.8;
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

linear1 = ss(A,B,C1,D); %y = h1
linear2 = ss(A,B,C2,D); %y = h2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Para 0.15-0.5                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h20 = 0.2; %Valor inicial de h2
h10 = h20*(130/339)^2+h20;


v2h2 = 0.15;
v2h1 = v2h2*(130/339)^2+v2h2;
v2u = sqrt( (6.78*sqrt(v2h1-v2h2) - 0.03)/(2.617*10^(-4) ) );
v3h2 = 0.5;
v3h1 = v3h2*(130/339)^2+v3h2;
v3u = sqrt( (6.78*sqrt(v3h1-v3h2) - 0.03)/(2.617*10^(-4) ) );

opt = odeset('Abstol', 1e-6, 'Reltol', 1e-6);
ta = 0:0.01:10;
tb = 10.01:0.01:20;
y02 = [h10, h20];

[t1,y1] = ode23(@(t1,y) odefcn(t1,y,v2u), ta, y02, opt);
[ly1,c] = size(y1); %quer-se o último valor de y1
[t2,y2] = ode23(@(t2,y) odefcn(t2,y,v3u), tb, y1(ly1,:), opt);
tnl = [t1; t2];
ynl = [y1; y2];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Modelo linear                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt = 0:0.01:20;
K = tf(1);
opt1 = stepDataOptions('StepAmplitude', v2u-v_u);
opt2 = stepDataOptions('StepAmplitude', v3u-v_u);

u1 = step(K, opt1, ta);
u2 = step(K, opt2, tb);
u = [u1; u2];
yl =  lsim(linear1,u,tt,y02-v_h1) + v_h1;
%plot(tnl, ynl(:,1), '-', tt, yl, '-');

yl2 = lsim(linear1,u,tt,[h20-v_h2 h20-v_h2]) + v_h2;
plot(tnl, ynl(:,2), '-', tt, yl2, '-');


ax = gca; % current axes
ax.FontSize = 12;
title('Resposta de h_1(t) para u_{0,15} \approx 61,01 e u_{0,5} \approx 83,13x');
xlabel('Tempo t(s)');
ylabel('Nível de água (m)');
legend('h_1(t) não linear', 'h_1(t) linear')


function dydt = odefcn(t,y,u)
dydt = zeros(2,1);
dydt(1) = ( 2.617*10^(-4)*u^2 + 0.03 - 6.78*sqrt(y(1)-y(2)) ) / ( pi*y(1)^2 - 0.9*pi*y(1) + 0.25*pi );
dydt(2) = ( 6.78*sqrt(y(1)-y(2)) - 2.6*sqrt(y(2)) ) / (-(pi/9)*y(2)^2 - 2*pi*y(2)/45 + 221*pi/900 );
end