clc
clear

syms s

p = s^5 + 3*s^2 + 2*s -1;

Qs = [4,  0, 0, 0, 0, 1;
      0, -1, 4, 0, 0, 0;
      0, 10, 0, 0, 0, 0];

ps = [-1; 2*s; 3*s^2; 0; 0; 1*s^5];

pqs = Qs*ps
A = [Qs ps];
rref(A)
  