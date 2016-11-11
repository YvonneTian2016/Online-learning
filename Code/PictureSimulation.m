%{
g_name='pixel100.jpg'; 
gI = double(imread(g_name));
gI = rgb2gray(gI);
imshow(gI);
Im = gI;
Im(:,:) = 0;
%}

g_name='stripe.jpg';
gI = imread(g_name);
gI = double(rgb2gray(gI));

sum_pdf = 0;
w = size(gI,1);
l = size(gI,2);
m = w * l;
for i = 1:m
    sum_pdf = sum_pdf + gI(i);
    cdf(i) = sum_pdf;
end

sample_num = 5000;

X1 = LENNA(gI,sample_num,cdf);
X = X1(:,1:2);

Im = gI;

Im(:,:) = 0;

for i = 1:sample_num
    Im(X1(i,1),X1(i,2)) = X1(i,3);
end

%figure(1);
%imshow(Im/255);


figure(1);
plot(X1(:, 1), X1(:, 2), 'b.');
%%====================================================
%% STEP 2: Choose initial values for the parameters.

% Set 'N' to the number of data points.
N = size(X, 1);

%k = size(mu_ori,1);  % The number of clusters.

k=20;  %output
c_num = k;

n = 2;  % The vector lengths.

% Randomly select k data points to serve as the initial means.%
indeces = randperm(N);
mu = X(indeces(1:k), :);
%mu = X(1:k,:);
muold = X(indeces(1:k),:);
%muold = X(1:k,:);

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

%Y = BoxMuller(mu_ori,sigma_ori,10,p_ori);

Y1 = LENNA(gI,i,cdf);
Y = Y1(:,1:2);
converged = 0;


while(converged == 0)
    %for (q = 1 : N)
    
    Z1 = LENNA(gI,1,cdf);
    Z = Z1(:,1:2);
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
        
        % For each of the clusters...
        for (j = 1 : k)
            
            % Calculate the prior probability for cluster 'j'.
            phi(j) = mean(W(:, j), 1);
            
            % Calculate the new mean for cluster 'j' by taking the weighted
            % average of all data points.
            
            eta = power(i,-alpha);
            
            muold(j, :) = weightedAverage(W(1:i-1, j), Y(1:i-1, :));
            mu(j,:) =plus((1-eta)*muold(j,:),eta * W(i,j) * Y(i,:));
            
            
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
        
        
        %{
    y = size(Y,1);
    for a = 1:y
       point_result = 0;
       for b = 1:k
           point_result = point_result + phi(k)*mvnpdf(Y(a,1:2),mu(b,1:2),sigma{b});
       end
       Im(floor(Y(a,1)),floor(Y(a,2)))=point_result*255;
    end
    
    
    figure(3);
    imshow(Im);
    pause(0.5)
        %}
        
        %%animation
        if(mod(i,5) == 0)
            
            % Display a scatter plot of the two distributions.
            figure(2);
            hold off;
            plot(Y(:, 1), Y(:, 2), 'b.');
            %tY = zeros(size(gI));
           % for i = 1:size(Y,1)
            %    tY(Y(i,1), Y(i,2)) = tY(Y(i,1), Y(i,2))  + 10;
            %end
            %imshow(tY);
            hold on;
            %plot(X2(:, 1), X2(:, 2), 'ro');
            
            set(gcf,'color','white') % White background for the figure.
            
            %{
            colors = {'rx', 'go', 'k^'};
            
            
            for(cluster = 1:c_num)
                plot(mu(cluster,1), mu(cluster,2), char(colors( mod(i, 3)+1 )), 'MarkerSize', 8, 'LineWidth', 6);
            end
            
            % First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
            % the input values over the grid.
            gridSize = 100;
            u = linspace(1, 512, gridSize);
            [A B] = meshgrid(u, u);
            gridX = [A(:), B(:)];
            
            % Calculate the Gaussian response for every value in the grid.
            for(cluster = 1:c_num)
                z = gaussianND(gridX, mu(cluster, :), sigma{cluster});
                
                % Reshape the responses back into a 2D grid to be plotted with contour.
                Z = reshape(z, gridSize, gridSize);
                
                % Plot the contour lines to show the pdf over the data.
                [C, h] = contour(u, u, Z);
            end
            axis([1 512 1 512])
            %}
            %title('Original Data and Estimated PDFs');
            title('Original Data');
            pause(0.5)
            
            
            tic
sI = zeros(w, l);
for iii = 1:10000
    p = unifrnd(0, 1);
    for lll = 1:k
        if p <= sum(phi(1:lll))
            p = lll; break;
        end
    end
    sX = mvnrnd(mu(lll, :), sigma{lll}, 10);
    sX = round(sX);
    c1 = find(sX(:,1) > 0 & sX(:,1) <= w);
    c2 = find(sX(:,2) > 0 & sX(:,2) <= l);
    C = intersect(c1,c2);
    sI(sX(C, 1), sX(C, 2)) =  sI(sX(C, 1), sX(C, 2)) + 1;
end

maxv = max( max(sI) ); minv = min( min(sI) );
sI = (sI - minv)/(maxv - minv);
tim = sI;


toc
            tim = (tim * 255);
            figure(3);
            imshow(tim/255);
            pause(0.5)
        end
        
        % Check for convergence.
        % mu
        % prevMu
        if (norm(mu - prevMu) < 0.5)
            converged = 1;
            %break;
        end
    end
    % End of Expectation Maximization
    
end
%{
     tic
            tim = zeros(size(Im));
            for a = 1:w
                for b = 1:l
                    maxv = 0; idx = 0;
                    for ll = 1:k
                        pb =  phi(k)*mvnpdf([a b],mu(ll,:),sigma{ll});
                        if maxv < pb
                            maxv = pb; idx = ll;
                        end
                    end
                    tim(a, b) = mvnpdf([a b],mu(idx,:),sigma{idx}) / mvnpdf(mu(idx,:),mu(idx,:),sigma{idx});
                end
            end
            toc
           %}
%{
tic
sI = zeros(w, l);
for i = 1:1
    p = unifrnd(0, 1);
    for l = 1:k
        if p <= sum(phi(1:l))
            p = l; break;
        end
    end
    sX = mvnrnd(mu(l, :), sigma{l}, 10);
    sX = round(sX);
    c1 = find(sX(:,1) > 0 & sX(:,1) <= w);
    c2 = find(sX(:,2) > 0 & sX(:,2) <= l);
    C = intersect(c1,c2);
    sI(sX(C, 1), sX(C, 2)) =  sI(sX(C, 1), sX(C, 2)) + 1;
end

maxv = max( max(sI) ); minv = min( min(sI) );
sI = (sI - minv)/(maxV - minV);
tim = sI;


toc
            tim = (tim * 255);
            figure(3);
            imshow(tim/255);
            pause(0.5)
%}
