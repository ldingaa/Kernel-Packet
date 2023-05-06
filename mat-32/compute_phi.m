function [vec_phi] = compute_phi(x,X,A,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function computes the vector vec_phi=k(x,X)A                        %
% where k(x,X) is the kernel covariance vector induced by the following   %
% kernel function:                                                        %
%                                                                         %
%                k(x,y)= [1+theta|x-y|]exp(-theta |x-y|).                 %
%                                                                         %
%A is the two-banded matrix and vec_phi at most has four non-zero entries %
%                                                                         %
%input: x: 1D input point, a real number                                  %
%       X: 1D data points sorted in increasing order                      %
%       A: the two-banded matrix A returned by compute_APhi(X,theta)      %
%       theta: scale parameter of kernel exp(-theta|x-y|)                 %
%                                                                         %
%output: vec_phi(x)=k(x,X)A. It has at most four non-zero entries         %                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,n]=size(X);

ind=find(X>x,1);
if isempty(ind)
    ind=n;
end

X=X*theta;
x=x*theta;


if ind>4 && ind<=n-3

    parfor i=-2:1
        ind_phi=ind+i;
        k_vec=(1+abs(x-X(ind_phi-2:ind_phi+2))).*exp(-abs(x-X(ind_phi-2:ind_phi+2)));
        vec(i+3)=k_vec*A(ind_phi-2:ind_phi+2,ind_phi);
    end
    vec_phi=sparse(1,ind-2:ind+1,vec,1,n);

elseif ind==4
    k_vec=(1+abs(x-X(1:4))).*exp(-abs(x-X(1:4)));
    vec(1)=k_vec*A(1:4,2);
    parfor i=3:5
        ind_phi=i;
        k_vec=(1+abs(x-X(ind_phi-2:ind_phi+2))).*exp(-abs(x-X(ind_phi-2:ind_phi+2)));
        vec(i-1)=k_vec*A(ind_phi-2:ind_phi+2,ind_phi);
    end
    vec_phi=sparse(1,ind-2:ind+1,vec,1,n);
elseif ind==3
    k_vec=(1+abs(x-X(1:3))).*exp(-abs(x-X(1:3)));
    vec(1)=k_vec*A(1:3,1);
    k_vec=(1+abs(x-X(1:4))).*exp(-abs(x-X(1:4)));
    vec(2)=k_vec*A(1:4,2);
    parfor i=3:4
        ind_phi=i;
        k_vec=(1+abs(x-X(ind_phi-2:ind_phi+2))).*exp(-abs(x-X(ind_phi-2:ind_phi+2)));
        vec(i)=k_vec*A(ind_phi-2:ind_phi+2,ind_phi);
    end
    vec_phi=sparse(1,1:ind+1,vec,1,n);
elseif ind==2
    k_vec=(1+abs(x-X(1:3))).*exp(-abs(x-X(1:3)));
    vec(1)=k_vec*A(1:3,1);
    k_vec=(1+abs(x-X(1:4))).*exp(-abs(x-X(1:4)));
    vec(2)=k_vec*A(1:4,2);
    k_vec=(1+abs(x-X(1:5))).*exp(-abs(x-X(1:5)));
    vec(3)=k_vec*A(1:5,3);
    vec_phi=sparse(1,1:ind+1,vec,1,n);
elseif ind<=1
    k_vec=(1+abs(x-X(1:3))).*exp(-abs(x-X(1:3)));
    vec(1)=k_vec*A(1:3,1);
    k_vec=(1+abs(x-X(1:4))).*exp(-abs(x-X(1:4)));
    vec(2)=k_vec*A(1:4,2);
    vec_phi=sparse(1,1:ind+1,vec,1,n);


 


elseif ind== (n-2)
    
    parfor i=-2:0
        ind_phi=ind+i;
        k_vec=(1+abs(x-X(ind_phi-2:ind_phi+2))).*exp(-abs(x-X(ind_phi-2:ind_phi+2)));
        vec(i+3)=k_vec*A(ind_phi-2:ind_phi+2,ind_phi);
    end
    k_vec=(1+abs(x-X(n-3:n))).*exp(-abs(x-X(n-3:n)));
    vec(4)=k_vec*A(n-3:n,n-1);
    
    vec_phi=sparse(1,ind-2:ind+1,vec,1,n);
    
elseif ind==(n-1) 
    
    parfor i=-2:-1
        ind_phi=ind+i;
        k_vec=(1+abs(x-X(ind_phi-2:ind_phi+2))).*exp(-abs(x-X(ind_phi-2:ind_phi+2)));
        vec(i+3)=k_vec*A(ind_phi-2:ind_phi+2,ind_phi);
    end
    k_vec=(1+abs(x-X(n-3:n))).*exp(-abs(x-X(n-3:n)));
    vec(3)=k_vec*A(n-3:n,n-1);
    k_vec=(1+abs(x-X(n-2:n))).*exp(-abs(x-X(n-2:n)));
    vec(4)=k_vec*A(n-2:n,n);
    vec_phi=sparse(1,ind-2:ind+1,vec,1,n);
elseif ind==n
    k_vec=(1+abs(x-X(n-4:n))).*exp(-abs(x-X(n-4:n)));
    vec(1)=k_vec*A(n-4:n,n-2);
    k_vec=(1+abs(x-X(n-3:n))).*exp(-abs(x-X(n-3:n)));
    vec(2)=k_vec*A(n-3:n,n-1);
    k_vec=(1+abs(x-X(n-2:n))).*exp(-abs(x-X(n-2:n)));
    vec(3)=k_vec*A(n-2:n,n);

    vec_phi=sparse(1,n-2:n,vec,1,n);
end

end