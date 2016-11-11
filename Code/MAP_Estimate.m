
%%======================================================
%% STEP 1a: Generate data from two 2D distributions.

mu1 = [5 8];       % Mean
sigma1 = [ 2 0;   % Covariance matrix
    0 2];

m1 = 200;          % Number of data points.

mu2 = [10 12];
sigma2 = [4 0;
    0 4];
m2 = 100;

mu3 = [15 12];
sigma3 = [16 0;
    0 16];

mu4 = [9 4];
sigma4 = [ 8 0;
    0 8];
c_num = 4;

%mu_ori = randi([-10,10],c_num,2);
%sigma_ori = randi([1,3],c_num,2);
%sigma = [];

%for(cluster = 1:c_num)
%s=zeros(2,2);
%s(1,1)=sigma_ori(cluster,1);
%s(2,2)=sigma_ori(cluster,2);    
%sigma = cat(1,sigma,s);
%end
%p_ori = ones(1,c_num)*(1/c_num);


sigma(:,:,1) = sigma1;
sigma(:,:,2) = sigma2;
sigma(:,:,3) = sigma3;
sigma(:,:,4) = sigma4;
sigma1 = [sigma1(1,1) sigma1(2,2)];
sigma2 = [sigma2(1,1) sigma2(2,2)];
sigma3 = [sigma3(1,1) sigma3(2,2)];
sigma4 = [sigma4(1,1) sigma4(2,2)];
sigma_ori =[sigma1;sigma2;sigma3;sigma4];
mu_ori = [mu1;mu2;mu3;mu4];

p_ori = ones(1,4)*(1/4);


num = 10000;
% Generate sample points with the specified means and covariance matrices.
%obj = gmdistribution(mu,sigma,p);
X = BoxMuller(mu_ori,sigma_ori,num,p_ori);


%%=====================================================
%% STEP 1b: Plot the data points and their pdfs.

figure(1);

% Display a scatter plot of the two distributions.
hold off
plot(X(1,:), X(2,:),'b.');
hold on;
for(cluster = 1:c_num)
plot(mu_ori(cluster,1), mu_ori(cluster,2), 'kx', 'MarkerSize', 8, 'LineWidth', 6);
end


set(gcf,'color','white') % White background for the figure.

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 100;
u = linspace(0, 20, gridSize);
[A B] = meshgrid(u, u);
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
for(cluster = 1:c_num)
z = gaussianND(gridX, mu_ori(cluster,:), sigma(:,:,cluster));

% Reshape the responses back into a 2D grid to be plotted with contour.
Z = reshape(z, gridSize, gridSize);

% Plot the contour lines to show the pdf over the data.
[C, h] = contour(u, u, Z);
end
axis([0 20 0 20]);
title('Original Data and PDFs');

%set(h,'ShowText','on','TextStep',get(h,'LevelStep')*2);


%%====================================================
%% STEP 2: Choose initial values for the parameters.

% Set 'N' to the number of data points.

k = c_num; %output


% Randomly select k data points to serve as the initial means.

mu = randi([0,20],2,c_num);

sigma = [];

% Assign equal prior probabilities to each cluster.
phi = ones(1, k) * (1/k);

alpha = 0.7;
%%===================================================
%% STEP 3: Run Expectation Maximization

% Matrix to hold the probability that each data point belongs to each cluster.
% One row per data point, one column per cluster.
%W = zeros(11, k);

% Loop until convergence.

i = 0;

m = 10;
N = 100;

off_sq = BoxMuller(mu_ori,sigma_ori,N,p_ori);

converged = 0;
% Use the overal covariance of the dataset as the initial variance for each cluster.
for (j = 1 : k)
   % sigma(:,:,j) = cov(off_sq');
    sigma(:,:,j) = [10 0;0 10];
end

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
    v = 1.01;
    n = i;
    I = eye(size(off_sq,1));
       
    
    % Store the previous means.
    prevMu = mu;
    prevSigma = sigma;
    prevPhi = phi;
    prevLikhood = likhood;
    % For each of the clusters...
    
    
    
    for (j = 1 : k)
        
        phi(j) = (U1(i,j)+(v-1)/n)/(1+(k*(v-1)/n));
        mu(:,j) = U2(:,j,i)./U1(i,j);
         
        A = U2(:,j,i)*mu(:,j)'+mu(:,j)*U2(:,j,i)'; %ST
        B = mu(:,j)*mu(:,j)';
        
        
      %  sigma(:,:,j)= (b*I + n*(U3(:,2*j-1:2*j,i)-2*U2(:,j,i)*mu(:,j)'+U2(:,j,i)*mu(:,j)'))/((a-2)+n*U1(i,j));
        sigma(:,:,j)= ((b/n).*I + U3(:,2*j-1:2*j,i)-A+U1(i,j)*B)/((a-2)/n+U1(i,j));
        
    end
    
   
     % Check for convergence.
    likhood = 0;
    for p = 1:N
    likhood = likhood+log(phi*mvnpdf(off_sq(:,p)',mu',sigma));
    end
    
    figure(4);
    axis normal;
    hold on;
    condition = abs(likhood/prevLikhood-1)
    scatter(i,condition);
    line(i,condition,'Marker','o');
    hold off;
    if abs(likhood/prevLikhood-1)<0.001
           converged = 1;
    end
    
    %%animation
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
        u = linspace(0, 20, gridSize);
        [A B] = meshgrid(u, u);
        gridX = [A(:), B(:)];
         axis([0 20 0 20]);
        % Calculate the Gaussian response for every value in the grid.
        
        for(cluster = 1:c_num)
        z = gaussianND(gridX, mu(:,cluster)', sigma(:,:,cluster));
        
        % Reshape the responses back into a 2D grid to be plotted with contour.
        Z = reshape(z, gridSize, gridSize);
       
        
        % Plot the contour lines to show the pdf over the data.
        
        [C, h] = contour(u, u, Z);
        hold on;
        end
        hold off;
        axis([0 20 0 20]);
        
        title('Original Data and Estimated PDFs');
        
        pause(0.5)
    end
      
 end  
   
end


%%online EM Stepwise 
converged = 0;
q = 0;
onli_sq =[];

while(converged == 0)    
      i = i + 1;
      q = q + 1;
      sq =BoxMuller(mu_ori,sigma_ori,1,p_ori);
      onli_sq = cat(2,onli_sq,sq);
      
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
    v = 1.01;
    n = i;
    I = eye(size(onli_sq,1));
       
    
    % Store the previous means.
    prevMu = mu;
    prevSigma = sigma;
    prevPhi = phi;
    prevLikhood = likhood;
    % For each of the clusters...
    
    
    
    for (j = 1 : k)
        
        phi(j) = (n*U1(i,j)+(v-1))/(n+(k*(v-1)));
        mu(:,j) = U2(:,j,i)./U1(i,j);
        
        
        
       % U3_S(1,1) = sqrt(U3(1,2*j-1,i));
        %U3_S(2,1) = sqrt(U3(2,2*j,i));
        
        A = U2(:,j,i)*mu(:,j)'+mu(:,j)*U2(:,j,i)'; %ST
        B = mu(:,j)*mu(:,j)';
        
        
        
       sigma(:,:,j)= ((b/n).*I + U3(:,2*j-1:2*j,i)-A+U1(i,j).*B)/((a-2)/n+U1(i,j));
     %   sigma(:,:,j)= (b.*I + n*(U2(:,j,i)*U2(:,j,i)'-2*U2(:,j,i)*mu(:,j)'+U2(:,j,i)*mu(:,j)'))/(a-2+n*U1(i,j));
    %   sigma(:,:,j)= (b.*I + n*(U2(:,j,i)*U2(:,j,i)'-U2(:,j,i)*mu(:,j)'))/(a-2+n*U1(i,j))
       %  sigma(:,:,j)= (b.*I + n*(U2(:,j,i)*U2(:,j,i)'*(1-1/U1(i,j))))/(a-2+n*U1(i,j));
    end
    
   
     % Check for convergence.
    likhood = 0;
    for p = 1:q
    likhood = likhood+phi*mvnpdf(onli_sq(:,p)',mu',sigma);
    end
    
    if(i>150000)
       alpha =0.6;
    end
    
    
    figure(4);
    axis normal;
    hold on;
    condition = abs(likhood/prevLikhood-1)
    scatter(i,condition);
    line(i,condition,'Marker','x');
    hold off;
 
    if abs(likhood/prevLikhood-1)<0.0001
           converged = 1;
    end
    
    %%animation
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
        u = linspace(0, 20, gridSize);
        [A B] = meshgrid(u, u);
        gridX = [A(:), B(:)];
        
        % Calculate the Gaussian response for every value in the grid.
        
        for(cluster = 1:c_num)
        z = gaussianND(gridX, mu(:,cluster)', sigma(:,:,cluster));
        
        % Reshape the responses back into a 2D grid to be plotted with contour.
        Z = reshape(z, gridSize, gridSize);
       
        
        % Plot the contour lines to show the pdf over the data.
        
        [C, h] = contour(u, u, Z);
        hold on;
        end
        hold off;
        axis([0 20 0 20]);
        
        title('Original Data and Estimated PDFs');
        
        pause(0.5)
    end
      
 end  

end



