%%Load Picture%%
g_name='gradient-scale.jpg';
%gI = double(imread(g_name));
gI = double(rgb2gray(imread(g_name)));


u = size(gI,1);  % length of the picture
v = size(gI,2);  % width of the picture
m = u * v;  % number of pixels in the picture

u_pdf = sum(gI,2)/v;
u_cdf = cumsum(u_pdf);

v_cdf = cumsum(gI,2);


[sample,sample_point] = Sample2D_test(gI,10000,u,v,u_cdf,v_cdf);

figure(1);
hold off;
plot(sample_point(:, 1), sample_point(:, 2), 'b.');

hold on;