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

% Marcações
yl = ylim;
plot([0,0],ylim,'k-.');
plot([Ns,Ns],ylim, 'k-.');
plot([Ns-500,Ns+500], [0,0],'k-.')

text(-1500,50,'Frenagem');
text(700,20,'Motor');
text(1.7*Ns,50,'Gerador');
%text(N(idx_max)-200,Tmax+5,'Pico')


%% Variando R2
figure(2)
hold on
Ns = 1800;
N  = linspace(0, Ns, 1001);
Tm = 0*N;
s  = (Ns - N)/Ns;
for Gain = 1:5
    for k = 1:length(N)
        Tm(k) = 3*Gain*R2*Vth^2/( (Rth+Gain*R2/s(k))^2 + (Xth+X2)^2)/s(k)/ws;
    end
    plot(N, Tm, 'linewidth', 3);
end

grid on;
a = gca; a.GridLineStyle='--';
xlabel('Rota\c{c}\~ao $\eta$ (rpm)', 'Interpreter', 'latex')
ylabel('Torque $T$ (N$\cdot$ m)', 'Interpreter', 'latex')

%% Variando a Freq.
fn = 60;
figure(3)
hold on
% Ganho de compensação da tensão
Gc = [4.2,2.25,1.62,1.31,1.13,1];
for f = [10,20,30,40,50,60,90,120]
    % Ensaio a vazio (dry-test)
    Iv = 10;
    Vv = 220/sqrt(3);
    Pv = 950/3;
    Rc = Vv^2/Pv;
    Xm = 1/sqrt(Iv^2/Vv^2 - Rc^(-2))*f/fn;

    % Ensaio com rotor bloqueado
    Ib1q = 12;
    Vb1q = 38/sqrt(3);
    Pb1q = 400/3;
    Req  = Pb1q/Ib1q^2;
    Zeq  = Vb1q/Ib1q;
    Xeq  = sqrt(Zeq^2 - Req^2)*f/fn;
    X1   = abs(Xeq/2);
    X2   = X1;

    % Parâmetros Mecânicos
    Pin    = 5100;
    N      = 1735;
    ws     = 1800*pi/30;
    w      = N*pi/30;
    Perdas = 1350;
    Pmech  = Pin - Perdas;
    Tnom   = Pmech/w/3;

    % Circuito equivalente
    if f >= 60
        V1 = 127;
    else
        V1  = 127*f/fn*Gc(f/10);
    end
    R1  = 0.5;
    Zp  = Rc*Xm*1j/(Rc + 1j*Xm);
    Zth = (R1+1j*X1)*Zp/(R1+1j*X1+Zp);
    Rth = real(Zth);
    Xth = imag(Zth);
    Vth = abs(V1*Zp/(R1+X1*1j+Zp));
    s   = (ws - w)/ws;
    R2  = s*Vth^2/Tnom/ws;
    
    % Varia N
    Ns = 1800*f/fn;
    N  = linspace(0*f/10, Ns, 1001);
    Tm = 0*N;
    s  = (Ns - N)/Ns;
    for k = 1:length(N)
        Tm(k) = 3*R2*Vth^2/( (Rth+R2/s(k))^2 + (Xth+X2)^2)/s(k)/ws;
    end
    plot(N,Tm,'linewidth',3)    
end

legend({'$f_1=10$ Hz','$f_2=20$ Hz','$f_3=30$ Hz','$f_4=40$ Hz',...
    '$f_5=50$ Hz','$f_6=60$ Hz','$f_7=90$ Hz','$f_8=120$ Hz'},'Interpreter', 'latex')
grid on;
a = gca; a.GridLineStyle='--';
xlabel('Rota\c{c}\~ao $\eta$ (rpm)', 'Interpreter', 'latex')
ylabel('Torque $T$ (N$\cdot$ m)', 'Interpreter', 'latex')
