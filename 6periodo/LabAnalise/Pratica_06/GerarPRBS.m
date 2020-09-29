%% Gerar PRBS
close all
clear all
clc
Range = [0, 2];
Band = [0 1];
u = idinput(2^10-1,'prbs',Band,Range);
figure(1)
um = u-mean(u);
plot(u)

%% Analisar espectro PRBS
Fs = 1/(1e-1);            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = length(um);             % Length of signal

U = fft(um);

Uabs1 = abs(U/L);
Uabs = 2*Uabs1(1:L/2+1);
Uabs(2:end-1) = Uabs(2:end-1);
f = 2*pi*Fs*(0:(L/2))/L;
% figure(2)
% stem(f,Uabs) 
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('rad/s')
% ylabel('|U(f)|')
% zoom
% figure(3)
% stem(f(1:L/20),Uabs(1:L/20)) 
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('rad/s')
% ylabel('|U(f)|')

Uphs1 = angle(U);
Uphs = Uphs1(1:L/2+1);
Uphs(2:end-1) = Uphs(2:end-1);
% Uph = mod(unwrap(Uphs),2*pi);
% figure(4)
% stem(f,Uphs) 
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('rad/s')
% ylabel('Phs(U(f))')

