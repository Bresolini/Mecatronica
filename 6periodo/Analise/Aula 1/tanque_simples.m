%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Aula 1 - Lab. Análise        %
%           Tanque Simples           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

syms u h

heq = 1; % metros
f = 2.5*u - 0.6*sqrt(h);
ueq = double(solve(subs(f==0,h,heq),u));

A = jacobian(f,h);
B = jacobian(f,u);
A = double(subs(A,[h,u],[heq,ueq]));
B = double(subs(B,[h,u],[heq,ueq]));

syms x
dx = A*x + B*u;



