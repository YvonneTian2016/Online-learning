%%======================================================
%% STEP 1a: Generate data from IMAGE distributions.
%%Load Picture%%
tic
g_name='LENNA.png';
gI = double(imread(g_name));
figure(1);
imshow(gI/255);

u = size(gI,1);  % length of the picture
v = size(gI,2);  % width of the picture

uv = u * v;  % number of pixels in the picture

u_pdf = sum(gI,2)/v;
u_cdf = cumsum(u_pdf);

v_cdf = cumsum(gI,2);
sum_int = sum(sum(gI)); %sum of intensity




%%====================================================
%% STEP 2: Choose initial values for the parameters.

% Set 'N' to the number of data points.


N = 10000;
k = 1024; %output
c_num = k;

Accuracy1 = 0.00001;
Accuracy2 = 0.00001;
IterMax1 = 5*N;
IterMax2 = 150000;

sigma_factor = 10000000000000;% svd xx:yy/yy:xx <=100

[off_sq_sample,off_sq,sample_num] = Sample2D_Stratified(gI,k,u,v,u_cdf,v_cdf);
off_sq = off_sq';


% Randomly select k data points to serve as the initial means.

mu = off_sq(:,1:k);
muold = off_sq(:,1:k);

% Use the overal covariance of the dataset as the initial variance for each cluster.
sigma = [];


% Use pdf intensity as the initial variance for each cluster

amplify = 1;
for (j = 1 : k)
  pdf(j) = (gI(off_sq_sample(j,1),off_sq_sample(j,2))*uv)/sum_int;
  sig = 1/(2*pi*(pdf(j)*k/uv));
  sigma(:,:,j) = amplify*[sig 0;0 sig];
end
% Assign equal prior probabilities to each cluster.
phi = ones(1, k) * (1/k);


figure(2);
        hold off;
        plot(off_sq(1,1:k), off_sq(2,1:k), 'b.');
        hold on;
       
        
        set(gcf,'color','white') % White background for the figure.
        
        colors = {'rx', 'go', 'k^'};
        
        figure(3);
        for(cluster = 1:c_num)
         
        % plot(mu(1,cluster), mu(2,cluster), char(colors( mod(j, 3)+1 )), 'MarkerSize', 8, 'LineWidth', 6);
        plot(mu(1,cluster), mu(2,cluster), 'b.');
         hold on;
        end
        axis([1 u 1 v]);
        % First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
        % the input values over the grid.
        gridSize = 100;
        l = linspace(1, u, gridSize);
        [A B] = meshgrid(l, l);
        gridX = [A(:), B(:)];
         
        % Calculate the Gaussian response for every value in the grid.
        
        for(cluster = 1:c_num)
        z = gaussianND(gridX, mu(:,cluster)', sigma(:,:,cluster));
        
        % Reshape the responses back into a 2D grid to be plotted with contour.
        Z = reshape(z, gridSize, gridSize);
       
        
        % Plot the contour lines to show the pdf over the data.
        
        [C, h] = contour(l, l, Z);
        hold on;
        end
        hold off;
        axis([1 u 1 v]);
        
        title('Original Data and Estimated PDFs');
        
        pause(0.5)





m = 10000;% online update slot
alpha = 0.7;


[off_sq_sample,sq,sample_num] = Sample2D_Stratified(gI,N-k,u,v,u_cdf,v_cdf); 
      off_sq = cat(2,off_sq,sq');


%%===================================================
%% STEP 3: Run Expectation Maximization

% Matrix to hold the probability that each data point belongs to each cluster.
% One row per data point, one column per cluster.
%W = zeros(11, k);

% Loop until convergence.
converged = 0;
likhood = 0;
%%=================================================
%%offline Training Process
i = 1;
q = 1;

 pdf = zeros(q, k);
    
    % For each cluster...
    for (j = 1 : k)
        
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = gaussianND(off_sq(:,1:q)', mu(:,j)', sigma(:,:,j));
    end
    
    % Multiply each pdf value by the prior probability for cluster.
    %    pdf  [m  x  k]
    %    phi  [1  x  k]
    %  pdf_w  [m  x  k]
    pdf_w = bsxfun(@times, pdf, phi);
    
    % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
    %   sum(pdf_w, 2) -- sum over the clusters.
    %pdf_w
    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));

        
    eta = power(i,-alpha);  % i = 0->N-1 in the paper
  for j = 1:k
     U1(i,j) = eta*W(q,j)*1;
     U2(:,j,i) = eta*W(q,j)*off_sq(:,q);
     U3(:,2*j-1:2*j,i) = eta*W(q,j)*(off_sq(:,q)*off_sq(:,q)');     
  end


while(converged == 0)    
      i = i + 1;
      q = mod(i,N);
      if q == 0;
         q = N;
      end
      
      fprintf('Off-line EM Iteration %d\n', i);
    %%===============================================
    %% STEP 3a: Expectation
    %
    % Calculate the probability for each data point for each distribution.
    
    % Matrix to hold the pdf value for each every data point for every cluster.
    % One row per data point, one column per cluster.
    pdf = zeros(q, k);
    
    % For each cluster...
    for (j = 1 : k)
        
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = gaussianND(off_sq(:,1:q)', mu(:,j)', sigma(:,:,j));
    end
    
    % Multiply each pdf value by the prior probability for cluster.
    %    pdf  [m  x  k]
    %    phi  [1  x  k]
    %  pdf_w  [m  x  k]
    pdf_w = bsxfun(@times, pdf, phi);
    
    % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
    %   sum(pdf_w, 2) -- sum over the clusters.
    %pdf_w
    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));
    
    
    eta = power(i,-alpha);  % i = 0->N-1 in the paper
    

    for j = 1:k
  
       U1(i,j) = (1-eta)*U1(i-1,j)+eta*W(q,j)*1;
       U2(:,j,i) = (1-eta)*U2(:,j,i-1)+eta*W(q,j)*off_sq(:,q);
       U3(:,2*j-1:2*j,i) = (1-eta)*U3(:,2*j-1:2*j,i-1)+eta*W(q,j)*(off_sq(:,q)*off_sq(:,q)');
    end

    if(mod(i,m) == 0)
  
    %%===============================================
    %% STEP 3b: Maximization
    %%
    %% Calculate the probability for each data point for each distribution.
    a = 2.01;
    b = 5*10^(-4);
    V = 1.01;
    n = i;
    I = eye(size(off_sq,1));
       
    
    % Store the previous means.
    prevMu = mu;
    prevSigma = sigma;
    prevPhi = phi;
    prevLikhood = likhood;
    % For each of the clusters...
    
    
    
    for (j = 1 : k)
        
        phi(j) = (U1(i,j)+(V-1)/n)/(1+(k*(V-1)/n));
        mu(:,j) = U2(:,j,i)./U1(i,j);
         
        A = U2(:,j,i)*mu(:,j)'+mu(:,j)*U2(:,j,i)'; %ST
        B = mu(:,j)*mu(:,j)';
        
        sigma(:,:,j)= ((b/n).*I + U3(:,2*j-1:2*j,i)-A+U1(i,j)*B)/((a-2)/n+U1(i,j));
        
    end
    
     %%add constriant to sigma for avoiding skinny guasses
    
    for(j = 1:k)
         flag = 0;
         [svd_U,svd_S,svd_V] = svd(sigma(:,:,j));
         product_C = svd_S(2,2)*svd_S(1,1);
         
         if(svd_S(2,2)/svd_S(1,1)> sigma_factor)   
               svd_S(1,1)=sqrt(product_C/sigma_factor)
               svd_S(2,2)=sigma_factor*svd_S(1,1);
               flag = 1;
         end
         if(svd_S(1,1)/svd_S(2,2)> sigma_factor)
               svd_S(2,2)=sqrt(product_C/sigma_factor)
               svd_S(1,1)=sigma_factor*svd_S(2,2);
               flag = 1;
         end
         if(flag == 1)
               sigma(:,:,j) = svd_U*svd_S*svd_V';
         end 
         
    end
    
   
     % Check for convergence.
    likhood = 0;
    for p = 1:q
    likhood = likhood+log(phi*mvnpdf(off_sq(:,p)',mu',sigma));
    end
    
    figure(4);
    axis normal;
    hold on;
    condition = abs(likhood/prevLikhood-1)
    scatter(i,condition);
    line(i,condition,'Marker','o');
    hold off;
    if (abs(likhood/prevLikhood-1)<Accuracy1)|| i > IterMax1
           converged = 1;
    end
    
    %%animation
    %{
    if(mod(i,m) == 0)
        
        % Display a scatter plot of the two distributions.
        figure(2);
        hold off;
        plot(off_sq(1,1:q), off_sq(2,1:q), 'b.');
        hold on;
       
        
        set(gcf,'color','white') % White background for the figure.
        
        colors = {'rx', 'go', 'k^'};
        
        figure(3);
        for(cluster = 1:c_num)
         
         plot(mu(1,cluster), mu(2,cluster), char(colors( mod(i, 3)+1 )), 'MarkerSize', 8, 'LineWidth', 6);
         hold on;
        end
 
        % First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
        % the input values over the grid.
        gridSize = 100;
        l = linspace(1, u, gridSize);
        [A B] = meshgrid(l, l);
        gridX = [A(:), B(:)];
         axis([1 u 1 v]);
        % Calculate the Gaussian response for every value in the grid.
        
        for(cluster = 1:c_num)
         flag_z=0;
            z = gaussianND(gridX, mu(:,cluster)', sigma(:,:,cluster));
            
            if z == 0
                flag_z=1;
            end
            
            
            % Reshape the responses back into a 2D grid to be plotted with contour.
            Z = reshape(z, gridSize, gridSize);
            
            
            
            % Plot the contour lines to show the pdf over the data.
            if flag_z==0
                [C, h] = contour(l, l, Z);
                hold on;
            end
        end
        hold off;
        axis([1 u 1 v]);
        
        title('Original Data and Estimated PDFs');
        
        pause(0.5)
    end
    %}
      
 end  
   
end


%%online EM Stepwise 
u = size(gI,1);  % length of the picture
v = size(gI,2);  % width of the picture

uv = u * v;  % number of pixels in the picture

u_pdf = sum(gI,2)/v;
u_cdf = cumsum(u_pdf);

v_cdf = cumsum(gI,2);
sum_int = sum(sum(gI)); %sum of intensity





converged = 0;
q = 0;
onli_sq =[];

while(converged == 0)    
    
      i = i + 1;
      q = q + 1;
     % [sq,pdf_sigma] =Pic_Sam(gI,1,cdf);
      [onli_sq_sample,sq,sample_num] = Sample2D_Stratified(gI,1,u,v,u_cdf,v_cdf); 
      onli_sq = cat(2,onli_sq,sq');
      
      fprintf('On-line EM Iteration %d\n', i);
    %%===============================================
    %% STEP 3a: Expectation
    %
    % Calculate the probability for each data point for each distribution.
    
    % Matrix to hold the pdf value for each every data point for every cluster.
    % One row per data point, one column per cluster.
    pdf = zeros(q, k);
    
    % For each cluster...
    for (j = 1 : k)
        
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = gaussianND(onli_sq(:,1:q)', mu(:,j)', sigma(:,:,j));
    end
    
    % Multiply each pdf value by the prior probability for cluster.
    %    pdf  [m  x  k]
    %    phi  [1  x  k]
    %  pdf_w  [m  x  k]
    pdf_w = bsxfun(@times, pdf, phi);
    
    % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
    %   sum(pdf_w, 2) -- sum over the clusters.
    %pdf_w
    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));
    
    
    eta = power(i,-alpha);  % i = 0->N-1 in the paper
    

    for j = 1:k
  
       U1(i,j) = (1-eta)*U1(i-1,j)+eta*W(q,j)*1;
       U2(:,j,i) = (1-eta)*U2(:,j,i-1)+eta*W(q,j)*onli_sq(:,q);
       U3(:,2*j-1:2*j,i) = (1-eta)*U3(:,2*j-1:2*j,i-1)+eta*W(q,j)*(onli_sq(:,q)*onli_sq(:,q)');
    end

    if(mod(i,m) == 0)
  
    %%===============================================
    %% STEP 3b: Maximization
    %%
    %% Calculate the probability for each data point for each distribution.
    a = 2.01;
    b = 5*10^(-4);
    V = 1.01;
    n = i;
    I = eye(size(onli_sq,1));
       
    
    % Store the previous means.
    prevMu = mu;
    prevSigma = sigma;
    prevPhi = phi;
    prevLikhood = likhood;
    % For each of the clusters...
    
    
    
    for (j = 1 : k)
        
        phi(j) = (n*U1(i,j)+(V-1))/(n+(k*(V-1)));
        mu(:,j) = U2(:,j,i)./U1(i,j);
    
        
        A = U2(:,j,i)*mu(:,j)'+mu(:,j)*U2(:,j,i)'; %ST
        B = mu(:,j)*mu(:,j)';
        
       sigma(:,:,j)= ((b/n).*I + U3(:,2*j-1:2*j,i)-A+U1(i,j).*B)/((a-2)/n+U1(i,j));
   
    end
    
    %%add constriant to sigma for avoiding skinny guasses
    
    for(j = 1:k)
         flag = 0;
         [svd_U,svd_S,svd_V] = svd(sigma(:,:,j));
         product_C = svd_S(2,2)*svd_S(1,1);
         
         if(svd_S(2,2)/svd_S(1,1)> sigma_factor)   
               svd_S(1,1)=sqrt(product_C/sigma_factor)
               svd_S(2,2)=sigma_factor*svd_S(1,1);
               flag = 1;
         end
         if(svd_S(1,1)/svd_S(2,2)> sigma_factor)
               svd_S(2,2)=sqrt(product_C/sigma_factor)
               svd_S(1,1)=sigma_factor*svd_S(2,2);
               flag = 1;
         end
         if(flag == 1)
               sigma(:,:,j) = svd_U*svd_S*svd_V';
         end 
         
    end
    
  % Check for convergence.
   likhood = 0;
   for p = 1:size(onli_sq,2)
     likhood = likhood+log(phi*mvnpdf(onli_sq(:,p)',mu',sigma));
    end
    
  
    figure(4);
    axis normal;
    hold on;
    condition = abs(likhood/prevLikhood-1)
    scatter(i,condition);
    line(i,condition,'Marker','x');
    hold off;
 
 
    condition= abs(likhood/prevLikhood-1)
   
    if (abs(likhood/prevLikhood-1)<Accuracy2)|| i >=IterMax2
           converged = 1;
    end
      
    
    %%animation
  %{
    if(mod(i,m) == 0)
        
        % Display a scatter plot of the two distributions.
        figure(2);
        hold off;
        plot(onli_sq(1,1:q), onli_sq(2,1:q), 'b.');
        hold on;
       
        
        set(gcf,'color','white') % White background for the figure.
        
        colors = {'rx', 'go', 'k^'};
        
        figure(3);
        for(cluster = 1:c_num)
         
         plot(mu(1,cluster), mu(2,cluster), char(colors( mod(i, 3)+1 )), 'MarkerSize', 8, 'LineWidth', 6);
         hold on;
        end
 
        % First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
        % the input values over the grid.
        gridSize = 100;
        l = linspace(1, u, gridSize);
        [A B] = meshgrid(l, l);
        gridX = [A(:), B(:)];
        
        % Calculate the Gaussian response for every value in the grid.
        
        for(cluster = 1:c_num)
            flag_z=0;
            z = gaussianND(gridX, mu(:,cluster)', sigma(:,:,cluster));
            
            if z == 0
                flag_z=1;
            end
            
            
            % Reshape the responses back into a 2D grid to be plotted with contour.
            Z = reshape(z, gridSize, gridSize);
            
            
            
            % Plot the contour lines to show the pdf over the data.
            if flag_z==0
                [C, h] = contour(l, l, Z);
                hold on;
            end
            
            
            
        end
        hold off;
        axis([1 u 1 v]);
        
        title('Original Data and Estimated PDFs');
        
        pause(0.5)
        
        
    end
      
    %}
    
    
    end  
    
       
end

toc


  % Display a scatter plot of the two distributions.
        figure(2);
        hold off;
        plot(onli_sq(1,1:q), onli_sq(2,1:q), 'b.');
        hold on;
       
        
        set(gcf,'color','white') % White background for the figure.
        
        colors = {'rx', 'go', 'k^'};
        
         GaussFig = figure(3);
        for(cluster = 1:c_num)
         
         plot(mu(1,cluster), mu(2,cluster), char(colors( mod(i, 3)+1 )), 'MarkerSize', 8, 'LineWidth', 6);
         hold on;
        end
 
        % First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
        % the input values over the grid.
        gridSize = 100;
        l = linspace(1, u, gridSize);
        [A B] = meshgrid(l, l);
        gridX = [A(:), B(:)];
        
        % Calculate the Gaussian response for every value in the grid.
        
        for(cluster = 1:c_num)
            flag_z=0;
            z = gaussianND(gridX, mu(:,cluster)', sigma(:,:,cluster));
            
            if z == 0
                flag_z=1;
            end
            
            
            % Reshape the responses back into a 2D grid to be plotted with contour.
            Z = reshape(z, gridSize, gridSize);
            
            
            
            % Plot the contour lines to show the pdf over the data.
            if flag_z==0
                [C, h] = contour(l, l, Z);
                hold on;
            end
            
            
            
        end
        hold off;
        axis([1 u 1 v]);
        
        title('Original Data and Estimated PDFs');
        
        pause(0.5)





tic
%maxpb = 0;
sumpb = 0;
pb = 0;
tim = zeros(size(gI));
for a = 1:u
    for b = 1:v
         pb = phi*mvnpdf([a b],mu',sigma);
        
        tim(a,b) = pb;
      %  if pb > maxpb
       %     maxpb = pb;
       % end;
         sumpb = sumpb +pb;       
    end
end
toc

ResultFig=figure(5);
image = (tim/sumpb)*(sum_int/255);
imshow(image);
save('Figs/sigma_svdInfinite_imagek3m4.mat','image');

savefig(GaussFig,'Figs/sigma_svdInfinite_k3m4Gauss.fig');
savefig(ResultFig,'Figs/sigma_svdInfinite_k3m4Reuslt.fig');
savefig(imshow,'Figs/sigma_svdInfinite_k3m4image.fig');
