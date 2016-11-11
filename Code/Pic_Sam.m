function [sample,sample_ori] = Pic_Sam(gI,sample_num,cdf)

sample = zeros(sample_num,2);
sample_ori = zeros(sample_num,2);
slope = 0;
l = size(gI,1);
w = size(gI,2);
w_pixel = 10;
l_pixel = 10;

m = w * l;



r = randi([0,cdf(m)],sample_num,1);

for i = 1 : sample_num

   
    j = BinarySearch(cdf,r(i),m);
    %pdf(i) = cdf(j)-cdf(j-1);
     if j == 1;
        slope = (r(i)-0)/(cdf(1)-0);
     else
        slope =(r(i)-cdf(j-1))/(cdf(j)-cdf(j-1));
     end
     
    slope = roundn(slope,-2);
    
    w_slope = mod(slope*100,w_pixel)/w_pixel;
    if(w_slope == 0)
        w_slope = 1;
    end
    l_slope =ceil(slope*100/l_pixel)/l_pixel;
   
    sample(i,1) = mod(j,l);
    sample(i,2) = floor(j/l)+1;
    
    
    
    
    if sample(i,1) == 0
     sample(i,1) = l-1+w_slope;
     sample(i,2) = floor(j/l)-1+l_slope;
     sample_ori(i,1) = l;
     sample_ori(i,2) =floor(j/l);
    else
     sample(i,1) = mod(j,l)-1+w_slope;
     sample(i,2) = floor(j/l)+1-1+l_slope;
     sample_ori(i,1) =mod(j,l);
     sample_ori(i,2) =floor(j/l)+1;
    end
   % sample(i,3) = gI(j);   
end
