function [ pdf ] = gaussianND(X, mu, Sigma)
%GAUSSIANND 
%      X - Matrix of data points, one per row.
%     mu - Row vector for the mean.
%  Sigma - Covariance matrix.

% Get the vector length.
n = size(X, 2);

% Subtract the mean from every data point.
meanDiff = bsxfun(@minus, X, mu);

% Calculate the multivariate gaussian.
%Sigma
%Sigma = Sigma - diag(diag(Sigma)); Sigma = Sigma+diag(sum(abs(Sigma)))+.1*eye(size(Sigma));

pdf = 1 / sqrt((2*pi)^n * det(Sigma)) * exp(-1/2 * sum((meanDiff * inv(Sigma) .* meanDiff), 2));
%{
if pdf < 1e-10
    pdf = 1e-10;
end
%}
end

