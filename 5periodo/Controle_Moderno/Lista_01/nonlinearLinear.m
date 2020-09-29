clc
clear
close

load iddata2 z2;
nlsys = nlarx(z2,[4 3 10],'tree','custom',{'sin(y1(t-2)*u1(t))+y1(t-2)*u1(t)+u1(t).*u1(t-13)','y1(t-5)*y1(t-5)*y1(t-1)'},'nlr',[1:5, 7 9]);
u0 = 1;
[X,~,r] = findop(nlsys, 'steady', 1);
y0 = r.SignalLevels.Output;
sys = linearize(nlsys,u0,X);
opt = stepDataOptions;
opt.StepAmplitude = 0.1;
t = linspace(0,10,	);
yl = step(sys, t, opt);
% plot(t, ynl, t, yl+y0)
% legend('Nonlinear', 'Linear with offset')


