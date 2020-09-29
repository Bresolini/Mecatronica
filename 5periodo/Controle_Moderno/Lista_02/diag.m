clc
clear
close

%variáveis
syms h1 h2 u

%Ponto de operação
vh2 = 0.5;
vh1 = 2.6*sqrt(vh2) == 6.78*sqrt(h1-vh2);
vh1 = solve(vh1,h1);
vu  = 2.6*sqrt(vh2) == 2.617e-4*u^2 + 0.03;
vu  = solve(vu,u);
vu  = vu(1);

z1 = ( 2.617e-4*u^2 + 0.03 - 6.78*sqrt(h1-h2) )/( pi*h1^2 - 0.9*pi*h1 + 0.25*pi );
z2 = ( 6.78*sqrt(h1-h2) - 2.6*sqrt(h2) )/( -pi*h1^2/9 - 2*pi*h2/45 + 221*pi/900 );
Z = [z1;z2];
H = [h1;h2];

A = double ( subs( jacobian(Z,H),[H; u],[vh1;vh2;vu] ) );
B = double ( subs( jacobian(Z,u),[H; u],[vh1;vh2;vu] ) );



