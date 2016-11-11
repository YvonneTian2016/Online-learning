function [ i ] = BinarySearch(pdf, rand_miu, miu_size) %

   flag = 0;
   left = 1;
   right = miu_size;
 
    while (flag == 0 && left <= right)
   
        middle = floor((left+right)/2);
       
         if middle == 1 && rand_miu < pdf(middle)
            i = 1;
            flag = 1;
         end
       
        if rand_miu >= pdf(middle) && rand_miu < pdf(middle + 1)
            i = middle+1;
            flag = 1;
        else
            if rand_miu > pdf(middle)
                left = middle + 1;
            else
                right = middle - 1;
            end
        end
     % i = middle;
    end
end
