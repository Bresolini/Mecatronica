%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Curvas de Torque em Motores de Indução
%
% Bernardo Bresolini
% Ester Q. Alvarenga
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all

%% Parâmetros

% Ensaio a vazio (dry-test)
Iv = 10;
Vv = 220/sqrt(3);
Pv = 950/3;
Rc = Vv^2/Pv;
Xm = 1/sqrt(Iv^2/Vv^2 - Rc^(-2));

% Ensaio com rotor bloqueado
Ib1q = 12;
Vb1q = 38/sqrt(3);
Pb1q = 400/3;
Req  = Pb1q/Ib1q^2;
Zeq  = Vb1q/Ib1q;
Xeq  = sqrt(Zeq^2 - Req^2);
X1   = abs(Xeq/2);
X2   = X1;

% Parâmetros Mecânicos
Pin    = 5100;
N      = 1735;
ws     = 1800*pi/30;
w      = N*pi/30;
s      = (ws - w)/ws;
Perdas = 1350;
Pmech  = Pin - Perdas;
Tnom   = Pmech/w/3;

% Circuito equivalente
V1  = 127;
R1  = 0.5;
Zp  = Rc*Xm*1j/(Rc + 1j*Xm);
Zth = (R1+1j*X1)*Zp/(R1+1j*X1+Zp);
Rth = real(Zth);
Xth = imag(Zth);
Vth = abs(V1*Zp/(R1+X1*1j+Zp));
s   = (ws - w)/ws;
R2  = s*Vth^2/Tnom/ws;


%% Plot

Ns = 1800;
N  = linspace(-Ns, 2.5*Ns, 3501);
Tm = 0*N;
s  = (Ns - N)/Ns;
for k = 1:length(N)
    Tm(k) = 3*R2*Vth^2/( (Rth+R2/s(k))^2 + (Xth+X2)^2)/s(k)/ws;
end

[Tmax, idx_max] = max(Tm);

hold on
plot(N, Tm, 'linewidth', 3);
scatter(N(idx_max), Tmax, 40, 'filled');
grid on;
a = gca; a.GridLineStyle='--';
xlabel('Rota\c{c}\~ao $\eta$ (rpm)', 'Interpreter', 'latex')
ylabel('Torque $T$ (N$\cdot$ m)', 'Interpreter', 'latex')









