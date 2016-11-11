#########********Where is the main function?********##########
There are multiple codes could be regarded as the main function,styled as “MAP_PictureSimulation***.m”.

#########********How to run the code?************##########
The most recent one which covered all the optimized things and also the most stable one is “MAP_PictureSimulation0428Stratified.m”.

If you want to edit the initial parameters run the code, you can do as following steps:

1.Set the initial parameters below the code notation:

“%% STEP 2: Choose initial values for the parameters.”

N = 10000; % Set 'N' to the number of offline data points.
k = 1024;  % Set 'k' to the number of Gausses.
c_num = k;

m = 10000;% online update slot
alpha = 0.7;

%Convergence Condition
Accuracy1 = 0.00001;
Accuracy2 = 0.00001;
IterMax1 = 5*N;
IterMax2 = 200000;

2. Now the code is simulating the picture “LENNA.png”, if you want to change 
to another picture, you can change the picture path below the notation :

%%Load Picture%%

g_name='LENNA.png';


After all those settings, you can run the code directly.

###########****Expression of the Generated Figures*****################
After you run the code, it will firstly generate the original image, initial gausses and random sample points Figures before it goes into the offline iterations.It will take a while to generate the Gaussian Contours. 
To reduce the time for drawing these figures, I commented out the animation code. After the GMM converged, the final image will show on the figure directly. And it will also take some time to draw the image out.

