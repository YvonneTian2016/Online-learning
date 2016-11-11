function [BiR] = inversion_Pdf(p)

m = size(p,2);

pdf = zeros(1,m);
sum = 0;
%k=1;


for i = 1:m
    sum = sum + p(i);
    pdf(i) = sum;
end

%pdf

r = rand;

BiR = BinarySearch(pdf,r,m);

%{
i = 1;
while(r > pdf(i))
    k = k + 1;
    i = i + 1;
end
%}
end