%% Analisando o sinal de saída
yw = Saida.signals.values;
y = yw(101:100+2^10-1)-mean(yw(101:100+2^10-1));
figure(8)
plot(y)
Ly = length(y);             % Length of signal

Y = fft(y);

Yabs1 = abs(Y/Ly);
Yabs = 2*Yabs1(1:Ly/2+1);
Yabs(2:end-1) = Yabs(2:end-1);

% figure(5)
% stem(f,Yabs) 
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('rad/s')
% ylabel('|Y(f)|')
% % zoom
% figure(6)
% stem(f(1:Ly/20),Yabs(1:Ly/20)) 
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('rad/s')
% ylabel('|Y(f)|')

Yphs1 = angle(Y/Ly);
Yphs = Yphs1(1:Ly/2+1);
Yphs(2:end-1) = Yphs(2:end-1);

% figure(7)
% stem(log10(f),Yphs) 
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('rad/s')
% ylabel('Phs(Y(f))')

%% Resposta em frequência
gabs = Yabs./Uabs;
figure(9)
stem(log10(f),20*log10(gabs))
gphs = (Yphs-Uphs);
% figure(10)
% stem(log(f),gphs)