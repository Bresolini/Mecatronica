clc
clear
close

syms a1 a2 a3 b1 b2;
syms p1 p2 p3 p4;
syms u1 u2 u3;
syms x y theta;

A = [-a1       a2*theta 0;
     -a2*theta -a1      0;
     0         0      -a3];
B = [-b1 -b1  b1 b1;
      b1 -b1 -b1 b1;
      b2  b2  b2 b2];
Bu = [b1  0  0;
       0 b1  0;
       0  0 b2];
X = [x;y;theta];
P = [p1;p2;p3;p4];
u = [u1;u2;u3];

Xdot = A*X + Bu*u
