function [ sample, sample_point ] = Sample2D_test(gI,sample_num,u,v,u_cdf,v_cdf)

sample = zeros(sample_num,2);

ru = 0+u_cdf(u)*rand(sample_num,1);




for i = 1 : sample_num
    
    j = BinarySearch(u_cdf,ru(i),u);
    sample(i,1)=j;
    
    
    
    if j == 1
        slope = ru(i)/u_cdf(j);
    else 
        slope = (ru(i)-u_cdf(j-1))/(u_cdf(j)-u_cdf(j-1));
    end
    sample_point(i,1)= j-1+slope;
    
    
    
    rv = 0+v_cdf(j,v)*rand(1,1);
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




