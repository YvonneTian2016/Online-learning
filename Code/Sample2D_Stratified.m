function [ sample, sample_point, sample_num ] = Sample2D_Stratified(gI,sample_num,u,v,u_cdf,v_cdf)

sample = zeros(sample_num,2);

sqrtVal = floor(sqrt(sample_num)+0.5);


sample_num = sqrtVal*sqrtVal;
invSqrtVal = 1 / sqrtVal;

sample = zeros(sample_num,2);
rand_sample = zeros(sample_num,2);

for i=0:sample_num-1
     y = floor(i / sqrtVal);
     x = mod(i,sqrtVal);

     rand_sample(i+1,1)= (x + rand())*invSqrtVal;
     rand_sample(i+1,2)= (y + rand())*invSqrtVal;
   
end

ru = 0+u_cdf(u)*rand_sample(1:sample_num,1);



for i = 1 : sample_num
    
    j = BinarySearch(u_cdf,ru(i),u);
    sample(i,1)=j;
    
    
    
    if j == 1
        slope = ru(i)/u_cdf(j);
    else 
        slope = (ru(i)-u_cdf(j-1))/(u_cdf(j)-u_cdf(j-1));
    end
    sample_point(i,1)= j-1+slope;
    
    
    
    rv = 0+v_cdf(j,v)*rand_sample(i,2);
    k = BinarySearch(v_cdf(j,:),rv,v);
    sample(i,2)=k;
    
    if k == 1
        slope = rv/v_cdf(j,k);
    else
        slope = (rv-v_cdf(j,k-1))/(v_cdf(j,k)-v_cdf(j,k-1));
    end
    sample_point(i,2)= k-1+slope;   
end

end




