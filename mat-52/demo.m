clear all
close all

X=sort(rand(1,100000)*1000);  %100000 data points
theta=.2;
Y=sin(pi*X);                 %value of sin on X
x=rand*1000;                  %a random input point

tic
[A,Phi]=compute_APhi(X,theta);
phi=compute_phi(x,X,A,theta);
y_hat=phi*(Phi\Y');
time=toc;

fprintf('Abbsolute err of predictor: %f, time taken for computing %f\n',abs(y_hat-sin(pi*x)),time);