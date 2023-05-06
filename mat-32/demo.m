clear all
close all

X=sort(rand(1,100000)*100);  %100000 data points
theta=1;
Y=sin(pi*X);                 %value of sin on X
x=rand*100;                  %a random input point

tic
[A,Phi]=compute_APhi(X,theta);
phi=compute_phi(x,X,A,theta);
y_hat=phi*(Phi\Y');
time=toc;

fprintf('Abbsolute err of predictor: %f, time taken for computing %f\n',abs(y_hat-sin(pi*x)),time);