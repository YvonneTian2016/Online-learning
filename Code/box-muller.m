% NORMAL SAMPLES USING BOX-MUELLER METHOD
% DRAW SAMPLES FROM PROPOSAL DISTRIBUTION
u = rand(2,100000);
r = sqrt(-2*log(u(1,:)));
theta = 2*pi*u(2,:);
x = r.*cos(theta);
y = r.*sin(theta);
 
% DISPLAY BOX-MULLER SAMPLES
figure
% X SAMPLES
subplot(121);
hist(x,100);
%colormap hot;axis square
%title(sprintf('Box-Muller Samples Y\n Mean = %1.2f\n Variance = %1.2f\n Kurtosis = %1.2f',mean(x),var(x),3-kurtosis(x)))
%xlim([-6 6])
 
% Y SAMPLES
subplot(122);
hist(y,100);
%colormap hot;axis square
%title(sprintf('Box-Muller Samples X\n Mean = %1.2f\n Variance = %1.2f\n Kurtosis = %1.2f',mean(y),var(y),3-kurtosis(y)))
%xlim([-6 6])