%%Load Picture%%
g_name='gradient-scale.jpg';
%g_name='LENNA.png';
gI = double(rgb2gray(imread(g_name)));
%gI = double(imread(g_name));;
figure(1);
imshow(gI/255);


u = size(gI,1);  % length of the picture
v = size(gI,2);  % width of the picture
%v=u;
m = u * v;  % number of pixels in the picture


interval1 = 10;
interval2 = 20;
n=1;
for i = 0:interval1:u-interval1
    for j = 0:interval2:v-interval2
        rand_loc1= interval1*rand(1);
        sample(n,1)=i+rand_loc1;
        rand_loc2 = interval2*rand(1);
        sample(n,2)=j+rand_loc2;
        n = n+1;
    end
end


u_pdf = sum(gI,2)/v;
u_cdf = cumsum(u_pdf);

v_cdf = cumsum(gI,2);
sum_int = sum(sum(gI));
%{
k=100000;
[sample,sample_point] = Sample2D_test(gI,k,u,v,u_cdf,v_cdf);
sample_swap = fliplr(sample_point);

%}


sample_swap = fliplr(sample);
figure(2);
hold off;
plot(sample_swap(:, 1), sample_swap(:, 2),'b.');
set(gca,'YDir','reverse','XAxisLocation','top'); % ? Y ????X ?????
xlabel('Y');
ylabel('X');
axis([1 v 1 u]);
hold on;





k = 600;
for (j = 1 : k)
  pdf(j) = (gI(ceil(sample(j,1)),ceil(sample(j,2)))*m)/sum_int;
  r(j)=sqrt(m/(pdf(j)*k*pi));
  if(pdf(j)~=0)
  sig = 1/(2*pi*(pdf(j)*k/m));
  else 
   sig = 0;
  end
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
    pdf_zeros = zeros(2,2);      
 for(cluster = 1:k)
    
    if(isequal(sigma(:,:,cluster),pdf_zeros)==0)
     
     z = gaussianND(gridX, sample_swap(cluster,:), sigma(:,:,cluster));
     % Reshape the responses back into a 2D grid to be plotted with contour.
     Z = reshape(z, gridSize, gridSize);
       
        
        % Plot the contour lines to show the pdf over the data.
        
        [C, h] = contour(l, ll, Z);
        hold on;
     end

 end
        hold off;
        set(gca,'YDir','reverse','XAxisLocation','top'); % ? Y ????X ?????
        xlabel('Y');
        ylabel('X');
        axis([1 v 1 u]);
        
        title('Original Data and Estimated PDFs');
              
