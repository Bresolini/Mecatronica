clc
close all
clear all

%% Especifique G
numG = [0 1];
denG = conv([1 1+j],[1 1-j]);
G   = tf(numG, denG);

kp = linspace(0.01,5,10);
%% Controlador tipo 2
for kd = 0.1:0.2:1.1
    
    
    
    
end


