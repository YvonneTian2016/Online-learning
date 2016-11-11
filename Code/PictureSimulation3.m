%%Load Picture%%
g_name='LENNA.png';
gI = double(imread(g_name));
%gI = imread(g_name);
%gI = double(rgb2gray(gI));

figure(1);
imshow(gI/255);
w = size(gI,1);  % width of the picture
l = size(gI,2);  % length of the picture
m = w * l;  % number of pixels in the picture


sum_pdf = 0;
for i = 1:m
    sum_pdf = sum_pdf + gI(i);
    cdf(i) = sum_pdf;
end
%%====================================================
%% STEP 1: Choose initial values for the parameters.

k=500;  %Number of Guassians in the GMM
c_num = k;

alpha = 0.7;

iter = 0;  % iteration times
m = 100;    % online update slot
i = k;     % index of all the data set
Y = [];     % data set


Y = Pic_Sam(gI,k,cdf);
N = size(Y, 1);


n = 2;  % The vector lengths.

% Randomly select k data points to serve as the initial means.%
%indeces = randperm(N);
mu = Y(1:k, :);
muold = Y(1:k,:);

sigma = [];

% Use the overal covariance of the dataset as the initial variance for each cluster.
div = gcd(w,l);
w_div = w/div;
l_div = l/div;

sigma_w = round(w/(w_div*(sqrt(k/(w_div*l_div)))))/2;
sigma_l = round(l/(l_div*(sqrt(k/(w_div*l_div)))))/2;


for (j = 1 : k)
     sigma(:,:,j) = [5000 -0.2;-0.2 5000];
    % sigma(:,:,j) = cov(Y);
end

%{
for (j = 1 : k)
    sigma(:,:,j) = cov(Y);
end
%}

% Assign equal prior probabilities to each cluster.
phi = ones(1, k) * (1/k);

%%===================================================
%% STEP 2: Run Expectation Maximization

% Matrix to hold the probability that each data point belongs to each cluster.
% One row per data point, one column per cluster.
%W = zeros(i+1, k);
W = [];

% Loop until convergence.


converged = 0;

prevMu = mu;
prevSigma = sigma;
prevPhi = phi;


tic
while(converged == 0)
    %for (q = 1 : N)
    
    Z = Pic_Sam(gI,m,cdf);
    i = i + m;
    
    Y = cat(1,Y,Z);
    
    
    if(mod(i,m) == 0)
        
        iter = iter+1;
        fprintf('  EM Iteration %d\n', iter);
        %%===============================================
        %% STEP 2a: Expectation
        %
        % Calculate the probability for each data point for each distribution.
        
        % Matrix to hold the pdf value for each every data point for every cluster.
        % One row per data point, one column per cluster.
        pdf = zeros(i, k);
        
        % For each cluster...
        for (j = 1 : k)
          %  if all(eig(sigma(:,:,j)) <= 0)
           %      sigma(:,:,j)=PrevSigma(:,:,j);
           % end
            % Evaluate the Gaussian for all data points for cluster 'j'.
            pdf(:, j) = gaussianND(Y, mu(j, :), sigma(:,:,j));
          %  if pdf(:, j) < 1e-10
           %     pdf(:, j) = 1e-10;
            %end
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
        %% STEP 2b: Maximization
        %%
        %% Calculate the probability for each data point for each distribution.
        
        % Store the previous means, sigmas, phis.
       preprevMu = prevMu;
       preprevSigma = prevSigma;
       preprevPhi = prevPhi;
        
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
            
            mu(j, :) = weightedAverage(W(:, j), Y(:, :));
            %mu(j,:) =plus((1-eta)*muold(j,:),eta * W(i,j) * Y(i,:));
            
            
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
            sigma(:,:,j) = sigma_k ./ sum(W(:, j));
           
            
            % sigma{j} = sqrt(sigma{j};
            % phi_output = phi(j)
            %mu_output = mu(j, :)
            %sigma_output = sigma{j}
            
        end
         
    % Check for convergence.
    % mu
    % prevMu

 if  iter >= 350 || norm(mu - prevMu) < 2
        converged = 1;
        %break;
    end
 
   %{
 if iter > 500 ||(abs((norm(mu-prevMu)-norm(prevMu-preprevMu)))/norm(mu-prevMu)<0.01 && abs((norm(sigma-prevSigma)-norm(prevSigma-preprevSigma)))/norm(sigma-prevSigma)<0.01 && abs((norm(phi-prevPhi)-norm(prevPhi-preprevPhi)))/norm(phi-prevPhi)<0.01)
         converged = 1;
    end
    %}
end
% End of Expectation Maximization
end
toc

% Display a scatter plot of the two distributions.
figure(2);
hold off;
plot(Y(:, 1), Y(:, 2), 'b.');

hold on;

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
u = linspace(1, w, gridSize);
[A B] = meshgrid(u, u);
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
for(cluster = 1:c_num)
    z = gaussianND(gridX, mu(cluster, :), sigma(:,:,cluster));
    
    % Reshape the responses back into a 2D grid to be plotted with contour.
    Z = reshape(z, gridSize, gridSize);
    
    % Plot the contour lines to show the pdf over the data.
    [C, h] = contour(u, u, Z);
    hold on;
end
hold off;
axis([1 w 1 l])

title('Original Data');

%% Draw the picture using current GMM
tic
%maxpb = 0;
sumpb = 0;
pb = 0;
tim = zeros(size(gI));
for a = 1:w
    for b = 1:l
         pb = phi*mvnpdf([a b],mu,sigma);
        
        tim(a,b) = pb;
      %  if pb > maxpb
       %     maxpb = pb;
       % end;
         sumpb = sumpb +pb;       
    end
end
toc

figure(4);
imshow((tim/sumpb)*(sum_pdf/255));
