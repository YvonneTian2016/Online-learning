%%Load Picture%%
%g_name='gradient-scale.jpg';
g_name='LENNA.png';
%gI = double(rgb2gray(imread(g_name)));
gI = double(imread(g_name));;
figure(1);
imshow(gI/255);

u = size(gI,1);  % length of the picture
v = size(gI,2);  % width of the picture
%v=u;
m = u * v;  % number of pixels in the picture

u_pdf = sum(gI,2)/v;
u_cdf = cumsum(u_pdf);

v_cdf = cumsum(gI,2);
sum_int = sum(sum(gI));

k=1000;
%[sample,sample_point,sample_num] = Sample2D_Stratified(gI,k,u,v,u_cdf,v_cdf);
%k = sample_num;
[sample,sample_point] = Sample2D_test(gI,k,u,v,u_cdf,v_cdf);
sample_swap = fliplr(sample_point);

figure(2);
hold off;
plot(sample_swap(:, 1), sample_swap(:, 2),'b.');
set(gca,'YDir','reverse','XAxisLocation','top'); % ? Y ????X ?????
xlabel('Y');
ylabel('X');
axis([1 v 1 u]);
hold on;






for (j = 1 : k)
  pdf(j) = (gI(sample(j,1),sample(j,2))*m)/sum_int;
  r(j)=sqrt(m/(pdf(j)*k*pi));
  sig = 1/(2*pi*(pdf(j)*k/m));
  sigma(:,:,j ) = [sig 0;0 sig];
end
figure(3); % Draw Disks
viscircles(sample_swap,r);

 set(gca,'YDir','reverse','XAxisLocation','top'); % ? Y ????X ?????
 xlabel('Y');
 ylabel('X');
 % set(gcf,'color','white') % White background for the figure.
axis([1 v 1 u]);

 
 figure(4); %Draw Gaussian Contours
 hold off;
 plot(sample_swap(:, 1), sample_swap(:, 2),'b.');
 hold on;
 %set(gcf,'color','white') % White background for the figure.
 set(gca,'YDir','reverse','XAxisLocation','top'); % ? Y ????X ?????
 xlabel('Y');
 ylabel('X');
 colors = {'rx', 'go', 'k^'};
 axis([1 v 1 u]);
       
 % First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
 % the input values over the grid.
        
 gridSize = 100;
 l = linspace(1, v, gridSize);
 ll = linspace(1,u,gridSize);
 [A B] = meshgrid(l, ll);
 gridX = [A(:), B(:)];
         
 % Calculate the Gaussian response for every value in the grid.
        
 for(cluster = 1:k)
        
     z = gaussianND(gridX, sample_swap(cluster,:), sigma(:,:,cluster));
     % Reshape the responses back into a 2D grid to be plotted with contour.
     Z = reshape(z, gridSize, gridSize);
       
        
        % Plot the contour lines to show the pdf over the data.
        
        [C, h] = contour(l, ll, Z);
        hold on;
        end
        hold off;
        set(gca,'YDir','reverse','XAxisLocation','top'); % ? Y ????X ?????
        xlabel('Y');
        ylabel('X');
        axis([1 v 1 u]);
        
        title('Original Data and Estimated PDFs');
              
 