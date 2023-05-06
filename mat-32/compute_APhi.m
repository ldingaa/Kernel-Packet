function [A, Phi] = compute_APhi(X,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function computes the kernel matrix factorization KA=Phi where K is %
%is the kernel covariance matrix induced by the following kernel function:%
%                                                                         %
%                 k(x,y)= [1+theta|x-y|]exp(-theta |x-y|).                %
%                                                                         %
%A is a two-banded matrix and Phi is a one-banded matrix                  %
%                                                                         %
%input: X: 1D data points sorted in increasing order                      %
%       theta: scale parameter of kernel exp(-theta|x-y|)                 %
%                                                                         %
%output: A: a two-banded matrix where each of its i-th column solve the   %
%        following system of equations                                    %
%          Σ_{j=i-2}^{i+2} A_[j,i]exp(-theta X_[j])        =0             %
%          Σ_{j=i-2}^{i+2} A_[j,i]exp( theta X_[j])        =0             %
%          Σ_{j=i-2}^{i+2} A_[j,i]X_[j]exp(-theta X_[j])   =0             %
%          Σ_{j=i-2}^{i+2} A_[j,i]X_[j]exp( theta X_[j])   =0             %
%        Phi: Phi=KA, it must be a one-banded matrix for any X            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,n]=size(X);
X=theta*X;
h=1/n^2;

S=[exp(X(1:3));X(1:3).*exp(X(1:3)) ]';
[Q,~]=qr(S);
A_l_1=Q(:,end)/h;
K=[(1+abs(X(1)-X(1)))*exp(-abs(X(1)-X(1)))  (1+abs(X(1)-X(2)))*exp(-abs(X(1)-X(2)))  (1+abs(X(1)-X(3)))*exp(-abs(X(3)-X(1))) ;...
    (1+abs(X(2)-X(1)))*exp(-abs(X(2)-X(1)))  (1+abs(X(2)-X(2)))*exp(-abs(X(2)-X(2)))  (1+abs(X(2)-X(3)))*exp(-abs(X(3)-X(2)))  ];

Phi_l_1=K*A_l_1;




S=[exp(X(1:4));X(1:4).*exp(X(1:4));exp(-X(1:4))]';
[Q,~]=qr(S);
A_l_2=Q(:,end)/h;

K=[(1+abs(X(1)-X(1)))*exp(-abs(X(1)-X(1)))  (1+abs(X(1)-X(2)))*exp(-abs(X(1)-X(2)))  (1+abs(X(1)-X(3)))*exp(-abs(X(1)-X(3)))  (1+abs(X(1)-X(4)))*exp(-abs(X(1)-X(4)));...
   (1+abs(X(2)-X(1)))*exp(-abs(X(2)-X(1)))  (1+abs(X(2)-X(2)))*exp(-abs(X(2)-X(2)))  (1+abs(X(2)-X(3)))*exp(-abs(X(2)-X(3)))  (1+abs(X(2)-X(4)))*exp(-abs(X(2)-X(4)));...
   (1+abs(X(3)-X(1)))*exp(-abs(X(3)-X(1)))  (1+abs(X(3)-X(2)))*exp(-abs(X(3)-X(2)))  (1+abs(X(3)-X(3)))*exp(-abs(X(3)-X(3)))  (1+abs(X(3)-X(4)))*exp(-abs(X(3)-X(4)))];
Phi_l_2=K*A_l_2;





A_central=zeros(5,n-4);
Phi_central=zeros(3,n-4);
parfor i=3:n-2
   S=[exp(X(i-2:i+2));exp(-X(i-2:i+2));X(i-2:i+2).*exp(X(i-2:i+2));X(i-2:i+2).*exp(-X(i-2:i+2))]';
   [Q,~]=qr(S);
   A_central(:,i-2)=-Q(:,end)/h;
   K=[(1+abs(X(i-1)-X(i-2)))*exp(-abs(X(i-1)-X(i-2)))  (1+abs(X(i-1)-X(i-1)))*exp(-abs(X(i-1)-X(i-1)))  (1+abs(X(i-1)-X(i)))*exp(-abs(X(i-1)-X(i)))  (1+abs(X(i-1)-X(i+1)))*exp(-abs(X(i-1)-X(i+1))) (1+abs(X(i-1)-X(i+2)))*exp(-abs(X(i-1)-X(i+2)));... 
      (1+abs(X(i)-X(i-2)))*exp(-abs(X(i)-X(i-2)))  (1+abs(X(i)-X(i-1)))*exp(-abs(X(i)-X(i-1)))  (1+abs(X(i)-X(i)))*exp(-abs(X(i)-X(i)))  (1+abs(X(i)-X(i+1)))*exp(-abs(X(i)-X(i+1))) (1+abs(X(i)-X(i+2)))*exp(-abs(X(i)-X(i+2)));...
      (1+abs(X(i+1)-X(i-2)))*exp(-abs(X(i+1)-X(i-2)))  (1+abs(X(i+1)-X(i-1)))*exp(-abs(X(i+1)-X(i-1)))  (1+abs(X(i+1)-X(i)))*exp(-abs(X(i+1)-X(i)))  (1+abs(X(i+1)-X(i+1)))*exp(-abs(X(i+1)-X(i+1))) (1+abs(X(i+1)-X(i+2)))*exp(-abs(X(i+1)-X(i+2)))];
       
   Phi_central(:,i-2)=K*A_central(:,i-2);
end



S=[exp(-X(n-3:n));X(n-3:n).*exp(-X(n-3:n));exp(X(n-3:n))]';
[Q,~]=qr(S);
A_r_2=Q(:,end)/h;
K=[(1+abs(X(n-2)-X(n-3)))*exp(-abs(X(n-2)-X(n-3)))  (1+abs(X(n-2)-X(n-2)))*exp(-abs(X(n-2)-X(n-2)))  (1+abs(X(n-2)-X(n-1)))*exp(-abs(X(n-2)-X(n-1)))  (1+abs(X(n-2)-X(n)))*exp(-abs(X(n-2)-X(n)));...
   (1+abs(X(n-1)-X(n-3)))*exp(-abs(X(n-1)-X(n-3)))  (1+abs(X(n-1)-X(n-2)))*exp(-abs(X(n-1)-X(n-2)))  (1+abs(X(n-1)-X(n-1)))*exp(-abs(X(n-1)-X(n-1)))  (1+abs(X(n-1)-X(n)))*exp(-abs(X(n-1)-X(n)));...   
   (1+abs(X(n)-X(n-3)))*exp(-abs(X(n)-X(n-3)))  (1+abs(X(n)-X(n-2)))*exp(-abs(X(n)-X(n-2)))  (1+abs(X(n)-X(n-1)))*exp(-abs(X(n)-X(n-1)))  (1+abs(X(n)-X(n)))*exp(-abs(X(n)-X(n)))];
Phi_r_2=K*A_r_2;



S=[exp(-X(n-2:n));X(n-2:n).*exp(-X(n-2:n));exp(X(n-2:n))]';
[Q,~]=qr(S);
A_r_1=Q(:,end)/h;
K=[(1+abs(X(n-1)-X(n-2)))*exp(-abs(X(n-1)-X(n-2)))  (1+abs(X(n-1)-X(n-1)))*exp(-abs(X(n-1)-X(n-1)))  (1+abs(X(n-1)-X(n)))*exp(-abs(X(n-1)-X(n)));... 
   (1+abs(X(n)-X(n-2)))*exp(-abs(X(n)-X(n-2)))  (1+abs(X(n)-X(n-1)))*exp(-abs(X(n)-X(n-1)))  (1+abs(X(n)-X(n)))*exp(-abs(X(n)-X(n)))];
Phi_r_1=K*A_r_1;


A=spdiags([A_l_1(1) A_l_2(2) A_central(3,:) A_r_2(end-1) A_r_1(end)]',0,n,n);
A=A+spdiags([A_l_1(2) A_l_2(3) A_central(4,:) A_r_2(end) ]',-1,n,n);
A=A+spdiags([A_l_1(end) A_l_2(end) A_central(5,:)]',-2,n,n);
A=A+spdiags([A_l_2(1) A_central(2,:) A_r_2(2) A_r_1(2)]',-1,n,n)';
A=A+spdiags([A_central(1,:) A_r_2(1) A_r_1(1)]',-2,n,n)';

Phi=spdiags([Phi_l_1(1) Phi_l_2(2) Phi_central(2,:) Phi_r_2(end-1) Phi_r_1(end)]',0,n,n);
Phi=Phi+spdiags([Phi_l_1(2) Phi_l_2(3) Phi_central(3,:) Phi_r_2(end) ]',-1,n,n);
Phi=Phi+spdiags( [Phi_l_2(1) Phi_central(1,:) Phi_r_2(1) Phi_r_1(1)  ]',-1,n,n)';


end






