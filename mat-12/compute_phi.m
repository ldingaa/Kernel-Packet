function [vec_phi] = compute_phi(x,X,A,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function computes the vector vec_phi=k(x,X)K^{-1}                   %
% where k(x,X) is the kernel covariance vector induced by the following   %
% kernel function:                                                        %
%                                                                         %
%                       k(x,y)= exp(-theta |x-y|).                        %
%                                                                         %
% and vec_phi at most has two non-zero entries                            %
%                                                                         %
%input: x: 1D input point, a real number                                  %
%       X: 1D data points sorted in increasing order                      %
%       A: inverse of the kernel covaiance matrix K. It is tridiagonal    %
%       theta: scale parameter of kernel exp(-theta|x-y|)                 %
%                                                                         %
%output: vec_phi(x)=k(x,X)K^{-1}. It has at most two non-zero entries     %                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,n]=size(X);

ind=find(X>x,1);
if isempty(ind)
    ind=n;
end

X=X*theta;
x=x*theta;

if ind>2 && ind <=n-1
    parfor i=-1:0
        ind_phi=ind+i;
        k_vec=exp(-abs(x-X(ind_phi-1:ind_phi+1)));
        vec(i+2)=k_vec*A(ind_phi-1:ind_phi+1,ind_phi);
    end
    vec_phi=sparse(1,ind-1:ind,vec,1,n);
elseif ind==2
    k_vec=exp(-abs(x-X(1:2)));
    vec(1)=k_vec*A(1:2,1);
    k_vec=exp(-abs(x-X(1:3)));
    vec(2)=k_vec*A(1:3,2);
    vec_phi=sparse(1,1:2,vec,1,n);
elseif ind==1
    k_vec=exp(-abs(x-X(1:2)));
    vec=k_vec*A(1:2,1);
    vec_phi=sparse(1,1,vec,1,n);
elseif ind==n
    k_vec=exp(-abs(x-X(n-2:n)));
    vec(1)=k_vec*A(n-2:n,n-1);
    k_vec=exp(-abs(x-X(n-1:n)));
    vec(2)=k_vec*A(n-1:n,n);
    vec_phi=sparse(1,n-1:n,vec,1,n);

end

