clc
clear
close

%% Definindo variáveis e condições dadas
G = tf( 100*[1 0.5], conv(  [1, -4], [1, -20, 125]  ) );
zpk(G)
ts = 5; OS = 17;
x0 = [0;
      0;
      0];

%% Encontrando a forma de espaço de estados
sys = ss(G);

%% Projetando os polos
zeta = -log(OS/100)/sqrt(pi^2 + log(OS/100)^2);
wn = 4/(ts*zeta);
zw = zeta*wn;
p1 = -zw + wn*sqrt(zeta^2 - 1);
p2 = -zw - wn*sqrt(zeta^2 - 1);
beta = imag(p1);

%% Projetando o controlador (subamortecido)
Aa = [sys.A, zeros(length(sys.A),1); -sys.C, 0];
Ba = [sys.B; 0];
F = [ -zw,  -beta;
     beta,    -zw];
z = zero(G);
if ( ~isempty(z) )
    for j = 1:length(z)
        F(2+j,2+j) = z(j);
    end
end

if ( length(F) == length(sys.A)+1 ) 
    kb = ones(1, length(sys.A)+1);
    if (rank(obsv(F,kb)) == length(sys.A)+1)
        disp('obsv de F e kb tem posto completo!')
        T = lyap(Aa, -F, -Ba*kb);
        kt = kb/T;
        k = kt(1:length(sys.A));
        ka = -kt(end);
    else
        disp('false')
    end
else
    for i = 3+length(z):length(sys.A)+1
    F(i,i) = -zw*(4+2*i);
    end
    kb = ones(1, length(sys.A)+1);
    if (rank(obsv(F,kb)) == length(sys.A)+1)
        disp('obsv de F e kb tem posto completo!')
        T = lyap(Aa, -F, -Ba*kb);
        kt = kb/T;
        k = kt(1:length(sys.A));
        ka = -kt(end);
    else
        disp('false')
    end
end

%% Projeto do Observador
Fobs = diag([-zw*2, -zw*1.5, -zw*3]);
if( length(Fobs) ~= length(sys.A) )
    disp('Erro!\n A matriz Fobs deve ter o mesmo tamanho de A')
else
    L = ones(length(sys.A),1);
    Tobs = lyap(-Fobs,sys.A,-L*sys.C); 
end




