clc
clear

syms a

A = [2*a, -3, -2;
       2,  5, -3;
      -2,  1,  a];

I = eye(3);
Aest = [A I]; % A estendida

Sest = rref(Aest); % S estendida
S = Sest(:,4:6);

syms x1 x2
X = [x1; x2];

A1 = subs(A,a,a);
R1 = A1(:,1:2);
S1 = A1(:,3);

eq = R1*X == S1
[sol1,sol2] = solve(eq,[x1,x2])