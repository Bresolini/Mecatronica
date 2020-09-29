%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               Modelagem do Sistema Ball and Beam - Chan                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear
close

%% Constantes

mB = 0.0282;    % kg
mb = 0.334;     % kg
R  = 0.0095;    % m
l  = 0.4;       % m
d  = 0.04;      % m
JB = 1.0469e-6; % kg*m^2
Jb = 0.017813;  % kg*m^2
Kb = 0.1491;    % V/rad/s
Kt = 0.1491;    % N*m/A
Ra = 18.91;     % Ohm
g  = 9.8;       % m/s^2
n  = 4.2;       % 

syms x1
A = (1 + (mB)^(-1)*JB/R^2 )^(-1);
B = (JB+Jb+mB*x1^2)^(-1);
C = n*Kb*l/Ra/d;
D = (n*Kb*l)^2/(Ra*d^2);
E = 0.5*l*mb*g;
F = mB*g;
G = 2*mB;

%% Linearização
syms x1 x2 x3 x4 u;
% x1: Ball position       (m)
% x2: Ball velocity       (m/s)
% x3: Beam angle          (rad)
% x4: Angular vel. beam   (rad/s)
%  u: motor voltage       (V)

dxB = A*x1*x4^2 - A*g*sin(x3);
dxb = B*cos(x3)*( C*cos( asin( l/d*sin(x3) ) )*u ...
      - d/sqrt( 1 - ( l/d*sin(x3) )^2 )*x4*cos(x3)*cos( asin( l/d*sin(x3) ) ) ...
      - E - F*x1 ) - B*G*x1*x2*x4;

tf = 10; dt = 0.01;
time = (0:dt:tf)';
x0 = [0;0;0;0]';

options = odeset('Abstol', 1e-9, 'Reltol', 1e-9);
[tt, xx] = ode23(@(t,x) edo(t,x,1), time, x0, options);

% p = plot(tt,xx);
% p = plot(ttnL,xxnL);
% legend('Posição (m)', 'Velocidade (m/s)');
% p(1).LineWidth=5; p(2).LineWidth=5; p(3).LineWidth=5; p(4).LineWidth=5;
% grid on; aa = gca; aa.GridLineStyle = '--';
% xlabel('tempo (s)');
% title({'Resposta do sistema não linear'});
  
  
  
  
  
function dx = edo(t,x,u)
mB = 0.0282;    % kg
mb = 0.334;     % kg
R  = 0.0095;    % m
l  = 0.4;       % m
d  = 0.04;      % m
JB = 1.0469e-6; % kg*m^2
Jb = 0.017813;  % kg*m^2
Kb = 0.1491;    % V/rad/s
Kt = 0.1491;    % N*m/A
Ra = 18.91;     % Ohm
g  = 9.8;       % m/s^2
n  = 4.2;       % 

A = (1 + (mB)^(-1)*JB/R^2 )^(-1);
B = (JB+Jb+mB*x(1)^2)^(-1);
C = n*Kb*l/Ra/d;
D = (n*Kb*l)^2/(Ra*d^2);
E = 0.5*l*mb*g;
F = mB*g;
G = 2*mB;

dx = zeros(4,1);
dx(1) = x(2);
dx(2) = A*x(1)*x(4)^2 - A*g*sin(x(3));
dx(3) = x(4);
dx(4) = B*cos(x(3))*( C*cos( asin( l/d*sin(x(3)) ) )*u ...
      - d/sqrt( 1 - ( l/d*sin(x(3)) )^2 )*x(4)*cos(x(3))*cos( asin( l/d*sin(x(3)) ) ) ...
      - E - F*x(1) ) - B*G*x(1)*x(2)*x(4);
end 
  
  