function [ sample ] = LENNA(gI,sample_num,cdf)

sample = zeros(sample_num,3);

l = size(gI,1);
w = size(gI,2);

m = w * l;



r = randi([0,cdf(m)],sample_num,1);

for i = 1 : sample_num

   
    j = BinarySearch(cdf,r(i),m);
    
    sample(i,1) = mod(j,l);
    sample(i,2) = floor(j/l)+1;
    
    if sample(i,1) == 0
     sample(i,1) = l;
     sample(i,2) = floor(j/l);
    end
    sample(i,3) = gI(j);   
end


