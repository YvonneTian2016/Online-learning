
%%======================================================
%% STEP 1a: Generate data from IMAGE distributions.
g_name='LENNA.png';
gI = double(imread(g_name));
max_gI= max(max(gI));
min_gI= min(min(gI));

g2_name = 'gradient-scale.jpg';
gI2 = double(rgb2gray(imread(g2_name)));
max_gI2 = max(max(max(gI2)));
min_gI2 = min(min(min(gI2)));


%figure(1);
%imshow(gI/255);
l = size(gI2,1);  % length of the picture
w = size(gI2,2);  % width of the picture
m = w * l;  % number of pixels in the picture



sum_pdf = 0;
for i = 1:m
    sum_pdf = sum_pdf + gI2(i);
    cdf(i) = sum_pdf;
end

%%====================================================
%% STEP 2: Choose initial values for the parameters.

% Set 'N' to the number of data points.


N = 1000;
k = 1000; %output
c_num = k;

%[off_sq,pdf_sigma] = Pic_Sam(gI,N,cdf);
%off_sq = off_sq';


% Randomly select k data points to serve as the initial means.
%indeces = randperm(k);
%mu = off_sq(:, indeces(1:k));
%muold = off_sq(:,indeces(1:k));
 
%mu = off_sq(:,1:k);
%muold = off_sq(:,1:k);


%w = 50;
%l = 50;
%m= w*l;
%pdf_grey = [32 16 8 4 2];


%{
%mu = randi([1,50],k,2);
pdf=ones(50,50);
pdf(1:10,:)=pdf_grey(1)*pdf(1:10,:);
pdf(11:20,:)=pdf_grey(2)*pdf(11:20,:);
pdf(21:30,:)=pdf_grey(3)*pdf(21:30,:);
pdf(31:40,:)=pdf_grey(4)*pdf(31:40,:);
pdf(41:50,:)=pdf_grey(5)*pdf(41:50,:);

sum_pdf = 0;
for i = 1:m
    sum_pdf = sum_pdf + pdf(i);
    cdf(i) = sum_pdf;
end
[mu_slope,mu] = Pic_Sam(pdf,k,cdf);

%}
 
[mu_slope,mu] = Pic_Sam(gI2,k,cdf);

figure(1);
        hold off;
       % plot(off_sq(1,1:k), off_sq(2,1:k), 'b.');
       plot(mu_slope(:,1), mu_slope(:,2), 'b.');
        hold on;
       


%{
count = zeros(max_gI - min_gI + 1,1);
for i = 1:m
    idx = gI(i) - min_gI;
    count(idx+1) = count(idx+1) + 1;
end

unique_pdf=unique(gI);
num_pdf = size(unique_pdf,1);



pdf1 = (gI(mu(1,1),mu(1,2))*count(gI(mu(1,1),mu(1,2))+1))/sum_pdf;
r1= sqrt((w*l)/(4*k*num_pdf*pdf1));
r(1)=r1;
%}

%{
pdf1=pdf(mu(1,1),mu(1,2))/sum(pdf_grey);
r1= sqrt((w*l)/(4*k*5*pdf1));
r(1)=r1;
%}
% Use the overal covariance of the dataset as the initial variance for each cluster.
sigma = [];

for (j = 1 : k)
  %sigma(:,:,j) = cov(off_sq');
  %sigma(:,:,j) = [500 0;0 500];
   %pdfn=(gI(mu(j,1),mu(j,2))*count(gI(mu(j,1),mu(j,2))+1))/sum_pdf;
   %ratio_n=pdfn/pdf1;
   %rn=sqrt(1/ratio_n)*r1;
   %r(j)=rn;
 
   pdf = (gI(mu(j,1),mu(j,2))*m)/sum_pdf;
   r(j)=sqrt(m/(pdf*k*pi));
  sig = r(j);
  sigma(:,:,j) = [sig 0;0 sig];
end
figure(2);
viscircles(mu_slope,r);
 axis([1 w 1 l]);







% Assign equal prior probabilities to each cluster.
phi = ones(1, k) * (1/k);


figure(3);
        hold off;
       % plot(off_sq(1,1:k), off_sq(2,1:k), 'b.');
       plot(mu_slope(:,1), mu_slope(:,2), 'b.');
        hold on;
       
        
        set(gcf,'color','white') % White background for the figure.
        
        colors = {'rx', 'go', 'k^'};
        
       % figure(3);
       % for(cluster = 1:c_num)
         
        % plot(mu(1,cluster), mu(2,cluster), char(colors( mod(i, 3)+1 )), 'MarkerSize', 8, 'LineWidth', 6);
        % hold on;
        %end
        axis([1 w 1 l]);
        % First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
        % the input values over the grid.
        gridSize = 100;
        u = linspace(1, w, gridSize);
        [A B] = meshgrid(u, u);
        gridX = [A(:), B(:)];
         
        % Calculate the Gaussian response for every value in the grid.
        
        for(cluster = 1:c_num)
        z = gaussianND(gridX, mu(cluster,:), sigma(:,:,cluster));
        
        % Reshape the responses back into a 2D grid to be plotted with contour.
        Z = reshape(z, gridSize, gridSize);
       
        
        % Plot the contour lines to show the pdf over the data.
        
        [C, h] = contour(u, u, Z);
        hold on;
        end
        hold off;
        axis([1 w 1 l]);
        
        title('Original Data and Estimated PDFs');
        
        pause(0.5)












m = 10;% online update slot
alpha = 0.7;
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
    if (abs(likhood/prevLikhood-1)<0.00001)|| i >= N
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
        u = linspace(1, w, gridSize);
        [A B] = meshgrid(u, u);
        gridX = [A(:), B(:)];
         axis([1 w 1 l]);
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
                [C, h] = contour(u, u, Z);
                hold on;
            end
        end
        hold off;
        axis([1 w 1 l]);
        
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
      [sq,pdf_sigma] =Pic_Sam(gI,1,cdf);
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
    
        
        A = U2(:,j,i)*mu(:,j)'+mu(:,j)*U2(:,j,i)'; %ST
        B = mu(:,j)*mu(:,j)';
        
       sigma(:,:,j)= ((b/n).*I + U3(:,2*j-1:2*j,i)-A+U1(i,j).*B)/((a-2)/n+U1(i,j));
   
    end
    
   
     % Check for convergence.
    likhood = 0;
    for p = 1:q
    likhood = likhood+phi*mvnpdf(onli_sq(:,p)',mu',sigma);
    end
    
    %{
    if(i>150000)
       alpha =0.6;
    end
    %}
    
    
  
    figure(4);
    axis normal;
    hold on;
    condition = abs(likhood/prevLikhood-1)
    scatter(i,condition);
    line(i,condition,'Marker','x');
    hold off;
 
 
    condition= abs(likhood/prevLikhood-1)
    %{
    if (abs(likhood/prevLikhood-1)<0.00001)||i >=100000
           converged = 1;
    end
    %}
    
    if i >=15000
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
        u = linspace(1, w, gridSize);
        [A B] = meshgrid(u, u);
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
                [C, h] = contour(u, u, Z);
                hold on;
            end
            
            
            
        end
        hold off;
        axis([1 w 1 l]);
        
        title('Original Data and Estimated PDFs');
        
        pause(0.5)
        
        
    end
      
    
    
    
    end  
    
       
end
tic
%maxpb = 0;
sumpb = 0;
pb = 0;
tim = zeros(size(gI));
for a = 1:w
    for b = 1:l
         pb = phi*mvnpdf([a b],mu',sigma);
        
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


