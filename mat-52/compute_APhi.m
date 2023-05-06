function [A,Phi] = compute_APhi(X,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function computes the kernel matrix factorization KA=Phi where K is %
%is the kernel covariance matrix induced by the following kernel function:%
%                                                                         %
%   k(x,y)= [1+theta|x-y|+(theta|x-y|)^2/3]exp(-theta |x-y|).             %
%                                                                         %
%A is a three-banded matrix and Phi is a two-banded matrix                %
%                                                                         %
%input: X: 1D data points sorted in increasing order                      %
%       theta: scale parameter of kernel function                         %
%                                                                         %
%output: A: a three-banded matrix where each of its i-th column solve the %
%        following system of equations                                    %
%          Σ_{j=i-3}^{i+3} A_[j,i]exp(-theta X_[j])        =0             %
%          Σ_{j=i-3}^{i+3} A_[j,i]exp( theta X_[j])        =0             %
%          Σ_{j=i-3}^{i+3} A_[j,i]X_[j]exp(-theta X_[j])   =0             %
%          Σ_{j=i-3}^{i+3} A_[j,i]X_[j]exp( theta X_[j])   =0             %
%          Σ_{j=i-3}^{i+3} A_[j,i]X_[j]^2exp(-theta X_[j]) =0             %
%          Σ_{j=i-3}^{i+3} A_[j,i]X_[j]^2exp( theta X_[j]) =0             %
%        Phi: Phi=KA, it must be a two-banded matrix for any X            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,n]=size(X);
h=n^-3;
X=theta*X;


S=[exp(X(1:4));X(1:4).*exp(X(1:4));X(1:4).^2.*exp(X(1:4)) ]';
[Q,~]=qr(S);
A_l_1=Q(:,end)/h;
K=[(1+abs(X(1)-X(1))+abs(X(1)-X(1))^2/3)*exp(-abs(X(1)-X(1)))  (1+abs(X(1)-X(2))+abs(X(1)-X(2))^2/3)*exp(-abs(X(1)-X(2)))  (1+abs(X(1)-X(3))+abs(X(1)-X(3))^2/3)*exp(-abs(X(1)-X(3))) (1+abs(X(1)-X(4))+abs(X(1)-X(4))^2/3)*exp(-abs(X(1)-X(4))) ;...
   (1+abs(X(2)-X(1))+abs(X(2)-X(1))^2/3)*exp(-abs(X(2)-X(1)))  (1+abs(X(2)-X(2))+abs(X(2)-X(2))^2/3)*exp(-abs(X(2)-X(2)))  (1+abs(X(2)-X(3))+abs(X(2)-X(3))^2/3)*exp(-abs(X(2)-X(3))) (1+abs(X(2)-X(4))+abs(X(2)-X(4))^2/3)*exp(-abs(X(2)-X(4))) ;...
   (1+abs(X(3)-X(1))+abs(X(3)-X(1))^2/3)*exp(-abs(X(3)-X(1)))  (1+abs(X(3)-X(2))+abs(X(3)-X(2))^2/3)*exp(-abs(X(3)-X(2)))  (1+abs(X(3)-X(3))+abs(X(3)-X(3))^2/3)*exp(-abs(X(3)-X(3))) (1+abs(X(3)-X(4))+abs(X(3)-X(4))^2/3)*exp(-abs(X(3)-X(4))) ];


Phi_l_1=K*A_l_1;




S=[exp(X(1:5));X(1:5).*exp(X(1:5));X(1:5).^2.*exp(X(1:5));exp(-X(1:5))]';
[Q,~]=qr(S);
A_l_2=Q(:,end)/h;

K=[(1+abs(X(1)-X(1))+abs(X(1)-X(1))^2/3)*exp(-abs(X(1)-X(1)))  (1+abs(X(1)-X(2))+abs(X(1)-X(2))^2/3)*exp(-abs(X(1)-X(2)))  (1+abs(X(1)-X(3))+abs(X(1)-X(3))^2/3)*exp(-abs(X(1)-X(3))) (1+abs(X(1)-X(4))+abs(X(1)-X(4))^2/3)*exp(-abs(X(1)-X(4))) (1+abs(X(1)-X(5))+abs(X(1)-X(5))^2/3)*exp(-abs(X(1)-X(5))) ;...
   (1+abs(X(2)-X(1))+abs(X(2)-X(1))^2/3)*exp(-abs(X(2)-X(1)))  (1+abs(X(2)-X(2))+abs(X(2)-X(2))^2/3)*exp(-abs(X(2)-X(2)))  (1+abs(X(2)-X(3))+abs(X(2)-X(3))^2/3)*exp(-abs(X(2)-X(3))) (1+abs(X(2)-X(4))+abs(X(2)-X(4))^2/3)*exp(-abs(X(2)-X(4))) (1+abs(X(2)-X(5))+abs(X(2)-X(5))^2/3)*exp(-abs(X(2)-X(5)));...
   (1+abs(X(3)-X(1))+abs(X(3)-X(1))^2/3)*exp(-abs(X(3)-X(1)))  (1+abs(X(3)-X(2))+abs(X(3)-X(2))^2/3)*exp(-abs(X(3)-X(2)))  (1+abs(X(3)-X(3))+abs(X(3)-X(3))^2/3)*exp(-abs(X(3)-X(3))) (1+abs(X(3)-X(4))+abs(X(3)-X(4))^2/3)*exp(-abs(X(3)-X(4))) (1+abs(X(3)-X(5))+abs(X(3)-X(5))^2/3)*exp(-abs(X(3)-X(5)));...
   (1+abs(X(4)-X(1))+abs(X(4)-X(1))^2/3)*exp(-abs(X(4)-X(1)))  (1+abs(X(4)-X(2))+abs(X(4)-X(2))^2/3)*exp(-abs(X(4)-X(2)))  (1+abs(X(4)-X(3))+abs(X(4)-X(3))^2/3)*exp(-abs(X(4)-X(3))) (1+abs(X(4)-X(4))+abs(X(4)-X(4))^2/3)*exp(-abs(X(4)-X(4))) (1+abs(X(4)-X(5))+abs(X(4)-X(5))^2/3)*exp(-abs(X(4)-X(5)))];


Phi_l_2=K*A_l_2;


S=[exp(X(1:6));X(1:6).*exp(X(1:6));X(1:6).^2.*exp(X(1:6));exp(-X(1:6));X(1:6).*exp(-X(1:6))]';
[Q,~]=qr(S);
A_l_3=Q(:,end)/h;

K=[(1+abs(X(1)-X(1))+abs(X(1)-X(1))^2/3)*exp(-abs(X(1)-X(1)))  (1+abs(X(1)-X(2))+abs(X(1)-X(2))^2/3)*exp(-abs(X(1)-X(2)))  (1+abs(X(1)-X(3))+abs(X(1)-X(3))^2/3)*exp(-abs(X(1)-X(3))) (1+abs(X(1)-X(4))+abs(X(1)-X(4))^2/3)*exp(-abs(X(1)-X(4))) (1+abs(X(1)-X(5))+abs(X(1)-X(5))^2/3)*exp(-abs(X(1)-X(5)))  (1+abs(X(1)-X(6))+abs(X(1)-X(6))^2/3)*exp(-abs(X(1)-X(6)));...
   (1+abs(X(2)-X(1))+abs(X(2)-X(1))^2/3)*exp(-abs(X(2)-X(1)))  (1+abs(X(2)-X(2))+abs(X(2)-X(2))^2/3)*exp(-abs(X(2)-X(2)))  (1+abs(X(2)-X(3))+abs(X(2)-X(3))^2/3)*exp(-abs(X(2)-X(3))) (1+abs(X(2)-X(4))+abs(X(2)-X(4))^2/3)*exp(-abs(X(2)-X(4))) (1+abs(X(2)-X(5))+abs(X(2)-X(5))^2/3)*exp(-abs(X(2)-X(5)))  (1+abs(X(2)-X(6))+abs(X(2)-X(6))^2/3)*exp(-abs(X(2)-X(6)));...
   (1+abs(X(3)-X(1))+abs(X(3)-X(1))^2/3)*exp(-abs(X(3)-X(1)))  (1+abs(X(3)-X(2))+abs(X(3)-X(2))^2/3)*exp(-abs(X(3)-X(2)))  (1+abs(X(3)-X(3))+abs(X(3)-X(3))^2/3)*exp(-abs(X(3)-X(3))) (1+abs(X(3)-X(4))+abs(X(3)-X(4))^2/3)*exp(-abs(X(3)-X(4))) (1+abs(X(3)-X(5))+abs(X(3)-X(5))^2/3)*exp(-abs(X(3)-X(5)))  (1+abs(X(3)-X(6))+abs(X(3)-X(6))^2/3)*exp(-abs(X(3)-X(6)));...
   (1+abs(X(4)-X(1))+abs(X(4)-X(1))^2/3)*exp(-abs(X(4)-X(1)))  (1+abs(X(4)-X(2))+abs(X(4)-X(2))^2/3)*exp(-abs(X(4)-X(2)))  (1+abs(X(4)-X(3))+abs(X(4)-X(3))^2/3)*exp(-abs(X(4)-X(3))) (1+abs(X(4)-X(4))+abs(X(4)-X(4))^2/3)*exp(-abs(X(4)-X(4))) (1+abs(X(4)-X(5))+abs(X(4)-X(5))^2/3)*exp(-abs(X(4)-X(5)))  (1+abs(X(4)-X(6))+abs(X(4)-X(6))^2/3)*exp(-abs(X(4)-X(6)));...
   (1+abs(X(5)-X(1))+abs(X(5)-X(1))^2/3)*exp(-abs(X(5)-X(1)))  (1+abs(X(5)-X(2))+abs(X(5)-X(2))^2/3)*exp(-abs(X(5)-X(2)))  (1+abs(X(5)-X(3))+abs(X(5)-X(3))^2/3)*exp(-abs(X(5)-X(3))) (1+abs(X(5)-X(4))+abs(X(5)-X(4))^2/3)*exp(-abs(X(5)-X(4))) (1+abs(X(5)-X(5))+abs(X(5)-X(5))^2/3)*exp(-abs(X(5)-X(5)))  (1+abs(X(5)-X(6))+abs(X(5)-X(6))^2/3)*exp(-abs(X(5)-X(6)))];


Phi_l_3=K*A_l_3;





A_central=zeros(7,n-6);
Phi_central=zeros(5,n-6);
parfor i=4:n-3
   S=[exp(X(i-3:i+3));...
       exp(-X(i-3:i+3));...
       X(i-3:i+3).*exp(X(i-3:i+3));...
       X(i-3:i+3).*exp(-X(i-3:i+3));...
       X(i-3:i+3).^2.*exp(-X(i-3:i+3));...
       X(i-3:i+3).^2.*exp(X(i-3:i+3))]';
   [Q,~]=qr(S);
   A_central(:,i-3)=-Q(:,end)/h;
   K=[(1+abs(X(i-2)-X(i-3))+abs(X(i-2)-X(i-3))^2/3)*exp(-abs(X(i-2)-X(i-3)))  (1+abs(X(i-2)-X(i-2))+abs(X(i-2)-X(i-2))^2/3)*exp(-abs(X(i-2)-X(i-2)))  (1+abs(X(i-2)-X(i-1))+abs(X(i-2)-X(i-1))^2/3)*exp(-abs(X(i-2)-X(i-1)))  (1+abs(X(i-2)-X(i))+abs(X(i-2)-X(i))^2/3)*exp(-abs(X(i-2)-X(i))) (1+abs(X(i-2)-X(i+1))+abs(X(i-2)-X(i+1))^2/3)*exp(-abs(X(i-2)-X(i+1))) (1+abs(X(i-2)-X(i+2))+abs(X(i-2)-X(i+2))^2/3)*exp(-abs(X(i-2)-X(i+2)))  (1+abs(X(i-2)-X(i+3))+abs(X(i-2)-X(i+3))^2/3)*exp(-abs(X(i-2)-X(i+3)));... 
      (1+abs(X(i-1)-X(i-3))+abs(X(i-1)-X(i-3))^2/3)*exp(-abs(X(i-1)-X(i-3)))  (1+abs(X(i-1)-X(i-2))+abs(X(i-1)-X(i-2))^2/3)*exp(-abs(X(i-1)-X(i-2)))  (1+abs(X(i-1)-X(i-1))+abs(X(i-1)-X(i-1))^2/3)*exp(-abs(X(i-1)-X(i-1)))  (1+abs(X(i-1)-X(i))+abs(X(i-1)-X(i))^2/3)*exp(-abs(X(i-1)-X(i))) (1+abs(X(i-1)-X(i+1))+abs(X(i-1)-X(i+1))^2/3)*exp(-abs(X(i-1)-X(i+1))) (1+abs(X(i-1)-X(i+2))+abs(X(i-1)-X(i+2))^2/3)*exp(-abs(X(i-1)-X(i+2)))  (1+abs(X(i-1)-X(i+3))+abs(X(i-1)-X(i+3))^2/3)*exp(-abs(X(i-1)-X(i+3)));... 
      (1+abs(X( i )-X(i-3))+abs(X( i )-X(i-3))^2/3)*exp(-abs(X( i )-X(i-3)))  (1+abs(X( i )-X(i-2))+abs(X( i )-X(i-2))^2/3)*exp(-abs(X( i )-X(i-2)))  (1+abs(X( i )-X(i-1))+abs(X( i )-X(i-1))^2/3)*exp(-abs(X( i )-X(i-1)))  (1+abs(X( i )-X(i))+abs(X( i )-X(i))^2/3)*exp(-abs(X( i )-X(i))) (1+abs(X( i )-X(i+1))+abs(X( i )-X(i+1))^2/3)*exp(-abs(X( i )-X(i+1))) (1+abs(X( i )-X(i+2))+abs(X( i )-X(i+2))^2/3)*exp(-abs(X( i )-X(i+2)))  (1+abs(X( i )-X(i+3))+abs(X( i )-X(i+3))^2/3)*exp(-abs(X( i )-X(i+3)));...
      (1+abs(X(i+1)-X(i-3))+abs(X(i+1)-X(i-3))^2/3)*exp(-abs(X(i+1)-X(i-3)))  (1+abs(X(i+1)-X(i-2))+abs(X(i+1)-X(i-2))^2/3)*exp(-abs(X(i+1)-X(i-2)))  (1+abs(X(i+1)-X(i-1))+abs(X(i+1)-X(i-1))^2/3)*exp(-abs(X(i+1)-X(i-1)))  (1+abs(X(i+1)-X(i))+abs(X(i+1)-X(i))^2/3)*exp(-abs(X(i+1)-X(i))) (1+abs(X(i+1)-X(i+1))+abs(X(i+1)-X(i+1))^2/3)*exp(-abs(X(i+1)-X(i+1))) (1+abs(X(i+1)-X(i+2))+abs(X(i+1)-X(i+2))^2/3)*exp(-abs(X(i+1)-X(i+2)))  (1+abs(X(i+1)-X(i+3))+abs(X(i+1)-X(i+3))^2/3)*exp(-abs(X(i+1)-X(i+3)));...
      (1+abs(X(i+2)-X(i-3))+abs(X(i+2)-X(i-3))^2/3)*exp(-abs(X(i+2)-X(i-3)))  (1+abs(X(i+2)-X(i-2))+abs(X(i+2)-X(i-2))^2/3)*exp(-abs(X(i+2)-X(i-2)))  (1+abs(X(i+2)-X(i-1))+abs(X(i+2)-X(i-1))^2/3)*exp(-abs(X(i+2)-X(i-1)))  (1+abs(X(i+2)-X(i))+abs(X(i+2)-X(i))^2/3)*exp(-abs(X(i+2)-X(i))) (1+abs(X(i+2)-X(i+1))+abs(X(i+2)-X(i+1))^2/3)*exp(-abs(X(i+2)-X(i+1))) (1+abs(X(i+2)-X(i+2))+abs(X(i+2)-X(i+2))^2/3)*exp(-abs(X(i+2)-X(i+2)))  (1+abs(X(i+2)-X(i+3))+abs(X(i+2)-X(i+3))^2/3)*exp(-abs(X(i+2)-X(i+3)))]; 



   Phi_central(:,i-3)=K*A_central(:,i-3);
end

S=[exp(-X(n-5:n));X(n-5:n).*exp(-X(n-5:n));X(n-5:n).^2.*exp(-X(n-5:n));exp(X(n-5:n));X(n-5:n).*exp(X(n-5:n))]';
[Q,~]=qr(S);
A_r_3=Q(:,end)/h;
K=[(1+abs(X(n-4)-X(n-5))+abs(X(n-4)-X(n-5))^2/3)*exp(-abs(X(n-4)-X(n-5)))  (1+abs(X(n-4)-X(n-4))+abs(X(n-4)-X(n-4))^2/3)*exp(-abs(X(n-4)-X(n-4)))  (1+abs(X(n-4)-X(n-3))+abs(X(n-4)-X(n-3))^2/3)*exp(-abs(X(n-4)-X(n-3))) (1+abs(X(n-4)-X(n-2))+abs(X(n-4)-X(n-2))^2/3)*exp(-abs(X(n-4)-X(n-2))) (1+abs(X(n-4)-X(n-1))+abs(X(n-4)-X(n-1))^2/3)*exp(-abs(X(n-4)-X(n-1)))  (1+abs(X(n-4)-X(n))+abs(X(n-4)-X(n))^2/3)*exp(-abs(X(n-4)-X(n)));...
   (1+abs(X(n-3)-X(n-5))+abs(X(n-3)-X(n-5))^2/3)*exp(-abs(X(n-3)-X(n-5)))  (1+abs(X(n-3)-X(n-4))+abs(X(n-3)-X(n-4))^2/3)*exp(-abs(X(n-3)-X(n-4)))  (1+abs(X(n-3)-X(n-3))+abs(X(n-3)-X(n-3))^2/3)*exp(-abs(X(n-3)-X(n-3))) (1+abs(X(n-3)-X(n-2))+abs(X(n-3)-X(n-2))^2/3)*exp(-abs(X(n-3)-X(n-2))) (1+abs(X(n-3)-X(n-1))+abs(X(n-3)-X(n-1))^2/3)*exp(-abs(X(n-3)-X(n-1)))  (1+abs(X(n-3)-X(n))+abs(X(n-3)-X(n))^2/3)*exp(-abs(X(n-3)-X(n)));...
   (1+abs(X(n-2)-X(n-5))+abs(X(n-2)-X(n-5))^2/3)*exp(-abs(X(n-2)-X(n-5)))  (1+abs(X(n-2)-X(n-4))+abs(X(n-2)-X(n-4))^2/3)*exp(-abs(X(n-2)-X(n-4)))  (1+abs(X(n-2)-X(n-3))+abs(X(n-2)-X(n-3))^2/3)*exp(-abs(X(n-2)-X(n-3))) (1+abs(X(n-2)-X(n-2))+abs(X(n-2)-X(n-2))^2/3)*exp(-abs(X(n-2)-X(n-2))) (1+abs(X(n-2)-X(n-1))+abs(X(n-2)-X(n-1))^2/3)*exp(-abs(X(n-2)-X(n-1)))  (1+abs(X(n-2)-X(n))+abs(X(n-2)-X(n))^2/3)*exp(-abs(X(n-2)-X(n)));...
   (1+abs(X(n-1)-X(n-5))+abs(X(n-1)-X(n-5))^2/3)*exp(-abs(X(n-1)-X(n-5)))  (1+abs(X(n-1)-X(n-4))+abs(X(n-1)-X(n-4))^2/3)*exp(-abs(X(n-1)-X(n-4)))  (1+abs(X(n-1)-X(n-3))+abs(X(n-1)-X(n-3))^2/3)*exp(-abs(X(n-1)-X(n-3))) (1+abs(X(n-1)-X(n-2))+abs(X(n-1)-X(n-2))^2/3)*exp(-abs(X(n-1)-X(n-2))) (1+abs(X(n-1)-X(n-1))+abs(X(n-1)-X(n-1))^2/3)*exp(-abs(X(n-1)-X(n-1)))  (1+abs(X(n-1)-X(n))+abs(X(n-1)-X(n))^2/3)*exp(-abs(X(n-1)-X(n)));...
   (1+abs(X( n )-X(n-5))+abs(X( n )-X(n-5))^2/3)*exp(-abs(X( n )-X(n-5)))  (1+abs(X( n )-X(n-4))+abs(X( n )-X(n-4))^2/3)*exp(-abs(X( n )-X(n-4)))  (1+abs(X( n )-X(n-3))+abs(X( n )-X(n-3))^2/3)*exp(-abs(X( n )-X(n-3))) (1+abs(X( n )-X(n-2))+abs(X( n )-X(n-2))^2/3)*exp(-abs(X( n )-X(n-2))) (1+abs(X( n )-X(n-1))+abs(X( n )-X(n-1))^2/3)*exp(-abs(X( n )-X(n-1)))  (1+abs(X( n )-X(n))+abs(X( n )-X(n))^2/3)*exp(-abs(X( n )-X(n)))];
Phi_r_3=K*A_r_3;






S=[exp(-X(n-4:n));X(n-4:n).*exp(-X(n-4:n));X(n-4:n).^2.*exp(-X(n-4:n));exp(X(n-4:n))]';
[Q,~]=qr(S);
A_r_2=Q(:,end)/h;
K=[ (1+abs(X(n-3)-X(n-4))+abs(X(n-3)-X(n-4))^2/3)*exp(-abs(X(n-3)-X(n-4)))  (1+abs(X(n-3)-X(n-3))+abs(X(n-3)-X(n-3))^2/3)*exp(-abs(X(n-3)-X(n-3))) (1+abs(X(n-3)-X(n-2))+abs(X(n-3)-X(n-2))^2/3)*exp(-abs(X(n-3)-X(n-2))) (1+abs(X(n-3)-X(n-1))+abs(X(n-3)-X(n-1))^2/3)*exp(-abs(X(n-3)-X(n-1)))  (1+abs(X(n-3)-X(n))+abs(X(n-3)-X(n))^2/3)*exp(-abs(X(n-3)-X(n)));...
    (1+abs(X(n-2)-X(n-4))+abs(X(n-2)-X(n-4))^2/3)*exp(-abs(X(n-2)-X(n-4)))  (1+abs(X(n-2)-X(n-3))+abs(X(n-2)-X(n-3))^2/3)*exp(-abs(X(n-2)-X(n-3))) (1+abs(X(n-2)-X(n-2))+abs(X(n-2)-X(n-2))^2/3)*exp(-abs(X(n-2)-X(n-2))) (1+abs(X(n-2)-X(n-1))+abs(X(n-2)-X(n-1))^2/3)*exp(-abs(X(n-2)-X(n-1)))  (1+abs(X(n-2)-X(n))+abs(X(n-2)-X(n))^2/3)*exp(-abs(X(n-2)-X(n)));...
    (1+abs(X(n-1)-X(n-4))+abs(X(n-1)-X(n-4))^2/3)*exp(-abs(X(n-1)-X(n-4)))  (1+abs(X(n-1)-X(n-3))+abs(X(n-1)-X(n-3))^2/3)*exp(-abs(X(n-1)-X(n-3))) (1+abs(X(n-1)-X(n-2))+abs(X(n-1)-X(n-2))^2/3)*exp(-abs(X(n-1)-X(n-2))) (1+abs(X(n-1)-X(n-1))+abs(X(n-1)-X(n-1))^2/3)*exp(-abs(X(n-1)-X(n-1)))  (1+abs(X(n-1)-X(n))+abs(X(n-1)-X(n))^2/3)*exp(-abs(X(n-1)-X(n)));...
    (1+abs(X( n )-X(n-4))+abs(X( n )-X(n-4))^2/3)*exp(-abs(X( n )-X(n-4)))  (1+abs(X( n )-X(n-3))+abs(X( n )-X(n-3))^2/3)*exp(-abs(X( n )-X(n-3))) (1+abs(X( n )-X(n-2))+abs(X( n )-X(n-2))^2/3)*exp(-abs(X( n )-X(n-2))) (1+abs(X( n )-X(n-1))+abs(X( n )-X(n-1))^2/3)*exp(-abs(X( n )-X(n-1)))  (1+abs(X( n )-X(n))+abs(X( n )-X(n))^2/3)*exp(-abs(X( n )-X(n)))];
Phi_r_2=K*A_r_2;


S=[exp(-X(n-3:n));X(n-3:n).*exp(-X(n-3:n));X(n-3:n).^2.*exp(-X(n-3:n))]';
[Q,~]=qr(S);
A_r_1=Q(:,end)/h;
K=[  (1+abs(X(n-2)-X(n-3))+abs(X(n-2)-X(n-3))^2/3)*exp(-abs(X(n-2)-X(n-3))) (1+abs(X(n-2)-X(n-2))+abs(X(n-2)-X(n-2))^2/3)*exp(-abs(X(n-2)-X(n-2))) (1+abs(X(n-2)-X(n-1))+abs(X(n-2)-X(n-1))^2/3)*exp(-abs(X(n-2)-X(n-1)))  (1+abs(X(n-2)-X(n))+abs(X(n-2)-X(n))^2/3)*exp(-abs(X(n-2)-X(n)));...
     (1+abs(X(n-1)-X(n-3))+abs(X(n-1)-X(n-3))^2/3)*exp(-abs(X(n-1)-X(n-3))) (1+abs(X(n-1)-X(n-2))+abs(X(n-1)-X(n-2))^2/3)*exp(-abs(X(n-1)-X(n-2))) (1+abs(X(n-1)-X(n-1))+abs(X(n-1)-X(n-1))^2/3)*exp(-abs(X(n-1)-X(n-1)))  (1+abs(X(n-1)-X(n))+abs(X(n-1)-X(n))^2/3)*exp(-abs(X(n-1)-X(n)));...
     (1+abs(X( n )-X(n-3))+abs(X( n )-X(n-3))^2/3)*exp(-abs(X( n )-X(n-3))) (1+abs(X( n )-X(n-2))+abs(X( n )-X(n-2))^2/3)*exp(-abs(X( n )-X(n-2))) (1+abs(X( n )-X(n-1))+abs(X( n )-X(n-1))^2/3)*exp(-abs(X( n )-X(n-1)))  (1+abs(X( n )-X(n))+abs(X( n )-X(n))^2/3)*exp(-abs(X( n )-X(n)))];
Phi_r_1=K*A_r_1;


A=spdiags(  [A_l_1(1) A_l_2(2) A_l_3(3) A_central(4,:)   A_r_3(end-2) A_r_2(end-1) A_r_1(end)]',0,n,n);
A=A+spdiags([A_l_1(2) A_l_2(3) A_l_3(4) A_central(5,:)   A_r_3(end-1) A_r_2(end)             ]',-1,n,n);
A=A+spdiags([A_l_1(3) A_l_2(4) A_l_3(5) A_central(6,:)   A_r_3(end)                          ]',-2,n,n);
A=A+spdiags([A_l_1(4) A_l_2(5) A_l_3(6) A_central(7,:)                                       ]',-3,n,n);

A=A+spdiags([A_l_2(1) A_l_3(2) A_central(3,:) A_r_3(3) A_r_2(3) A_r_1(3)]',-1,n,n)';
A=A+spdiags([         A_l_3(1) A_central(2,:) A_r_3(2) A_r_2(2) A_r_1(2)]',-2,n,n)';
A=A+spdiags([                  A_central(1,:) A_r_3(1) A_r_2(1) A_r_1(1)]',-3,n,n)';

Phi=    spdiags([Phi_l_1(1) Phi_l_2(2) Phi_l_3(3) Phi_central(3,:) Phi_r_3(end-2) Phi_r_2(end-1) Phi_r_1(end)]',0,n,n);
Phi=Phi+spdiags([Phi_l_1(2) Phi_l_2(3) Phi_l_3(4) Phi_central(4,:) Phi_r_3(end-1) Phi_r_2(end) ]',-1,n,n);
Phi=Phi+spdiags([Phi_l_1(3) Phi_l_2(4) Phi_l_3(5) Phi_central(5,:) Phi_r_3(end)  ]',-2,n,n);


Phi=Phi+spdiags([Phi_l_2(1) Phi_l_3(2) Phi_central(2,:) Phi_r_3(2) Phi_r_2(2) Phi_r_1(2) ]',-1,n,n)';
Phi=Phi+spdiags([           Phi_l_3(1) Phi_central(1,:) Phi_r_3(1) Phi_r_2(1) Phi_r_1(1) ]',-2,n,n)';


end






