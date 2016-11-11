
%%======================================================
%% STEP 1a: Generate data from two 2D distributions.
%{
mu1 = [1 2];       % Mean
sigma1 = [ 3 0;   % Covariance matrix
    0 2];

m1 = 200;          % Number of data points.

mu2 = [-1 -2];
sigma2 = [2 0;
    0 1];
m2 = 100;

mu3 = [-5 -4];
sigma3 = [2 0;
    0 1];

mu4 = [5 4];
sigma4 = [3 0;
    0 2];
%}
c_num = 6;

mu_ori = randi([-10,10],c_num,2);
sigma_ori = randi([1,3],c_num,2);
%sigma_ori = ones(10,2);
sigma = [];

for(cluster = 1:c_num)
s=zeros(2,2);
s(1,1)=sigma_ori(cluster,1);
s(2,2)=sigma_ori(cluster,2);    
sigma = cat(1,sigma,s);
%sigma = [sigma1;sigma2;sigma3;sigma4];
end
%mu_ori = [mu1;mu2;mu3;mu4];

p_ori = ones(1,c_num)*(1/c_num);

num = 10000;
% Generate sample points with the specified means and covariance matrices.
%obj = gmdistribution(mu,sigma,p);
X = BoxMuller(mu_ori,sigma_ori,num,p_ori);


%%=====================================================
%% STEP 1b: Plot the data points and their pdfs.

figure(1);

% Display a scatter plot of the two distributions.
hold off
plot(X(:, 1), X(:, 2),'b.');
hold on;
for(cluster = 1:c_num)
plot(mu_ori(cluster,1), mu_ori(cluster,2), 'kx', 'MarkerSize', 8, 'LineWidth', 6);
end
%plot(X2(:, 1), X2(:, 2), 'ro');

set(gcf,'color','white') % White background for the figure.

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 100;
u = linspace(-20, 20, gridSize);
[A B] = meshgrid(u, u);
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
for(cluster = 1:c_num)
z = gaussianND(gridX, mu_ori(cluster,:), sigma(cluster*2-1:cluster*2,:));

% Reshape the responses back into a 2D grid to be plotted with contour.
Z = reshape(z, gridSize, gridSize);

% Plot the contour lines to show the pdf over the data.
[C, h] = contour(u, u, Z);
end
axis([-20 20 -20 20])
title('Original Data and PDFs');

%set(h,'ShowText','on','TextStep',get(h,'LevelStep')*2);


%%====================================================
%% STEP 2: Choose initial values for the parameters.

% Set 'N' to the number of data points.
N = size(X, 1);

k = size(mu_ori,1);  % The number of clusters.

k  %output


n = 2;  % The vector lengths.

% Randomly select k data points to serve as the initial means.
indeces = randperm(N);
mu = X(indeces(1:k), :);
muold = X(indeces(1:k),:);

sigma = [];

alpha = 0.7;

% Use the overal covariance of the dataset as the initial variance for each cluster.
for (j = 1 : k)
    sigma{j} = cov(X);
end

% Assign equal prior probabilities to each cluster.
phi = ones(1, k) * (1/k);

%%===================================================
%% STEP 3: Run Expectation Maximization

% Matrix to hold the probability that each data point belongs to each cluster.
% One row per data point, one column per cluster.
W = zeros(11, k);

% Loop until convergence.

i = 10;
m = 100;
Y = [];

Y = BoxMuller(mu_ori,sigma_ori,10,p_ori);
converged = 0;


while(converged == 0)
%for (q = 1 : N)
    
    Z = BoxMuller(mu_ori,sigma_ori,1,p_ori);
    i = i + 1;
    
    Y = cat(1,Y,Z);
   
  
    
    fprintf('  EM Iteration %d\n', i);
    
    if(mod(i,m) == 0)
    
    
    %%===============================================
    %% STEP 3a: Expectation
    %
    % Calculate the probability for each data point for each distribution.
    
    % Matrix to hold the pdf value for each every data point for every cluster.
    % One row per data point, one column per cluster.
    pdf = zeros(i, k);
    
    % For each cluster...
    for (j = 1 : k)
        
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = gaussianND(Y, mu(j, :), sigma{j});
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
    
    %%===============================================
    %% STEP 3b: Maximization
    %%
    %% Calculate the probability for each data point for each distribution.
    
    % Store the previous means.
    prevMu = mu;
    prevSigma = sigma;
    prevPhi = phi;
    % For each of the clusters...
    for (j = 1 : k)
        
        % Calculate the prior probability for cluster 'j'.
        phi(j) = mean(W(:, j), 1);
        
        % Calculate the new mean for cluster 'j' by taking the weighted
        % average of all data points.
        
        eta = power(i,-alpha);
        
        %muold(j, :) = weightedAverage(W(1:i-1, j), Y(1:i-1, :));
        mu(j,:) =plus((1-eta)*prevMu(j,:),eta * W(i,j) * Y(i,:));
        
        
        % Calculate the covariance matrix for cluster 'j' by taking the
        % weighted average of the covariance for each training example.
        
        sigma_k = zeros(n, n);
        
        % Subtract the cluster mean from all data points.
        Xm = bsxfun(@minus, Y, mu(j, :));
        
        % Calculate the contribution of each training example to the covariance matrix.
        for (o = 1 : i)
            sigma_k = sigma_k + (W(o, j) .* (Xm(o, :)' * Xm(o, :)));
        end
        
        %W
        %sigma_k
       
        % Divide by the sum of weights.
        sigma{j} = sigma_k ./ sum(W(:, j));
        
        
        % sigma{j} = sqrt(sigma{j};
        phi_output = phi(j)
        mu_output = mu(j, :)
        sigma_output = sigma{j}
        
    end
    
     % Check for convergence.
    if (norm(mu - prevMu) <0.01)
        converged = 1;
        break;
    end
    
    
    
    
    %%animation
    if(mod(i,5) == 0)
        
        % Display a scatter plot of the two distributions.
        figure(2);
        hold off;
        plot(Y(:, 1), Y(:, 2), 'b.');
        hold on;
        %plot(X2(:, 1), X2(:, 2), 'ro');
        
        set(gcf,'color','white') % White background for the figure.
        
        colors = {'rx', 'go', 'k^'};
        
        figure(3);
        for(cluster = 1:c_num)
         
         plot(mu(cluster,1), mu(cluster,2), char(colors( mod(i, 3)+1 )), 'MarkerSize', 8, 'LineWidth', 6);
         hold on;
        end
 
        % First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
        % the input values over the grid.
        gridSize = 100;
        u = linspace(-20, 20, gridSize);
        [A B] = meshgrid(u, u);
        gridX = [A(:), B(:)];
        
        % Calculate the Gaussian response for every value in the grid.
        
        for(cluster = 1:c_num)
        z = gaussianND(gridX, mu(cluster, :), sigma{cluster});
        
        % Reshape the responses back into a 2D grid to be plotted with contour.
        Z = reshape(z, gridSize, gridSize);
       
        
        % Plot the contour lines to show the pdf over the data.
        
        [C, h] = contour(u, u, Z);
        hold on;
        end
        hold off;
        axis([-20 20 -20 20])
        
        title('Original Data and Estimated PDFs');
        
        pause(0.5)
    end
      
 end  
 
    
    
    
    % End of Expectation Maximization
%{
%%=====================================================
%% STEP 4: Plot the data points and their estimated pdfs.

% Display a scatter plot of the two distributions.
figure(2);
hold off;
plot(X(:, 1), X(:, 2), 'b.');
hold on;
%plot(X2(:, 1), X2(:, 2), 'ro');

set(gcf,'color','white') % White background for the figure.

plot(mu1(1), mu1(2), 'kx');
plot(mu2(1), mu2(2), 'kx');

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 100;
u = linspace(-6, 6, gridSize);
[A B] = meshgrid(u, u);
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.

z1 = gaussianND(gridX, mu(1, :), sigma{1});
z2 = gaussianND(gridX, mu(2, :), sigma{2});

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1 = reshape(z1, gridSize, gridSize);
Z2 = reshape(z2, gridSize, gridSize);

% Plot the contour lines to show the pdf over the data.
[C, h] = contour(u, u, Z1);
[C, h] = contour(u, u, Z2);
axis([-6 6 -6 6])

title('Original Data and Estimated PDFs');
%}
end