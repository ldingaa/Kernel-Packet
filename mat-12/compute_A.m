function [A] = compute_A(X,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function computes the inver kernel matrix factorization A=K^{-1}    %
% where K is the kernel covariance matrix induced by the following kernel %
%function:                                                                %
%                       k(x,y)= exp(-theta |x-y|).                        %
%                                                                         %
%K^{-1} must be a one-banded matrix                                       %
%                                                                         %
%input: X: 1D data points sorted in increasing order                      %
%       theta: scale parameter of kernel exp(-theta|x-y|)                 %
%                                                                         %
%output: A: inverse of the kernel covaiance matrix K. It is tridiagonal   %                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



N=numel(X);
X=X*theta;
vec1=X(2:end)-X(1:end-1);
upper_diag=1./(exp(vec1)-exp(-vec1));

vec2=X(3:end)-X(1:end-2);
diag_m=(exp(vec2)-exp(-vec2)).*upper_diag(2:end);
diag_m=diag_m.*upper_diag(1:end-1);

A=spdiags([exp(X(2)-X(1))*upper_diag(1) diag_m exp(X(end)-X(end-1))*upper_diag(end)]',0,N,N)+spdiags(-upper_diag',-1,N,N)+transpose(spdiags(-upper_diag',-1,N,N));


end

