function [vec_phi] = compute_phi(x,X,A,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function computes the vector vec_phi=k(x,X)A                        %
% where k(x,X) is the kernel covariance vector induced by the following   %
% kernel function:                                                        %
%                                                                         %
%   k(x,y)= [1+theta|x-y|+(theta|x-y|)^2/3]exp(-theta |x-y|).             %
%                                                                         %
%A is the three-banded matrix and vec_phi at most has six non-zero entries%
%                                                                         %
%input: x: 1D input point, a real number                                  %
%       X: 1D data points sorted in increasing order                      %
%       A: the three-banded matrix A returned by compute_APhi(X,theta)    %
%       theta: scale parameter of kernel function                         %
%                                                                         %
%output: vec_phi(x)=k(x,X)A. It has at most six non-zero entries          %                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,n]=size(X);

ind=find(X>x,1);
if isempty(ind)
    ind=n;
end

X=X*theta;
x=x*theta;


if ind>6 && ind<=n-5

    parfor i=-3:2
        ind_phi=ind+i;
        k_vec=(1+abs(x-X(ind_phi-3:ind_phi+3))+abs(x-X(ind_phi-3:ind_phi+3)).^2/3).*exp(-abs(x-X(ind_phi-3:ind_phi+3)));
        vec(i+4)=k_vec*A(ind_phi-3:ind_phi+3,ind_phi);
    end
    vec_phi=sparse(1,ind-3:ind+2,vec,1,n);

elseif ind==6
    k_vec=(1+abs(x-X(1:6))+abs(x-X(1:6)).^2/3).*exp(-abs(x-X(1:6)));
    vec(1)=k_vec*A(1:6,3);
    parfor i=-2:2
        ind_phi=ind+i;
        k_vec=(1+abs(x-X(ind_phi-3:ind_phi+3))+abs(x-X(ind_phi-3:ind_phi+3)).^2/3).*exp(-abs(x-X(ind_phi-3:ind_phi+3)));
        vec(i+4)=k_vec*A(ind_phi-3:ind_phi+3,ind_phi);
    end
    vec_phi=sparse(1,ind-3:ind+2,vec,1,n);
elseif ind==5
    k_vec=(1+abs(x-X(1:5))+abs(x-X(1:5)).^2/3).*exp(-abs(x-X(1:5)));
    vec(1)=k_vec*A(1:5,2);
    k_vec=(1+abs(x-X(1:6))+abs(x-X(1:6)).^2/3).*exp(-abs(x-X(1:6)));
    vec(2)=k_vec*A(1:6,3);
    parfor i=-1:2
        ind_phi=ind+i;
        k_vec=(1+abs(x-X(ind_phi-3:ind_phi+3))+abs(x-X(ind_phi-3:ind_phi+3)).^2/3).*exp(-abs(x-X(ind_phi-3:ind_phi+3)));
        vec(i+4)=k_vec*A(ind_phi-3:ind_phi+3,ind_phi);
    end
    vec_phi=sparse(1,ind-3:ind+2,vec,1,n);
elseif ind==4
    k_vec=(1+abs(x-X(1:4))+abs(x-X(1:4)).^2/3).*exp(-abs(x-X(1:4)));
    vec(1)=k_vec*A(1:4,1);
    k_vec=(1+abs(x-X(1:5))+abs(x-X(1:5)).^2/3).*exp(-abs(x-X(1:5)));
    vec(2)=k_vec*A(1:5,2);
    k_vec=(1+abs(x-X(1:6))+abs(x-X(1:6)).^2/3).*exp(-abs(x-X(1:6)));
    vec(3)=k_vec*A(1:6,3);
    parfor i=0:2
        ind_phi=ind+i;
        k_vec=(1+abs(x-X(ind_phi-3:ind_phi+3))+abs(x-X(ind_phi-3:ind_phi+3)).^2/3).*exp(-abs(x-X(ind_phi-3:ind_phi+3)));
        vec(i+4)=k_vec*A(ind_phi-3:ind_phi+3,ind_phi);
    end
    vec_phi=sparse(1,ind-3:ind+2,vec,1,n);
elseif ind==3
    k_vec=(1+abs(x-X(1:4))+abs(x-X(1:4)).^2/3).*exp(-abs(x-X(1:4)));
    vec(1)=k_vec*A(1:4,1);
    k_vec=(1+abs(x-X(1:5))+abs(x-X(1:5)).^2/3).*exp(-abs(x-X(1:5)));
    vec(2)=k_vec*A(1:5,2);
    k_vec=(1+abs(x-X(1:6))+abs(x-X(1:6)).^2/3).*exp(-abs(x-X(1:6)));
    vec(3)=k_vec*A(1:6,3);
    parfor i=1:2
        ind_phi=ind+i;
        k_vec=(1+abs(x-X(ind_phi-3:ind_phi+3))+abs(x-X(ind_phi-3:ind_phi+3)).^2/3).*exp(-abs(x-X(ind_phi-3:ind_phi+3)));
        vec(i+3)=k_vec*A(ind_phi-3:ind_phi+3,ind_phi);
    end
    vec_phi=sparse(1,1:ind+2,vec,1,n);
elseif ind==2
    k_vec=(1+abs(x-X(1:4))+abs(x-X(1:4)).^2/3).*exp(-abs(x-X(1:4)));
    vec(1)=k_vec*A(1:4,1);
    k_vec=(1+abs(x-X(1:5))+abs(x-X(1:5)).^2/3).*exp(-abs(x-X(1:5)));
    vec(2)=k_vec*A(1:5,2);
    k_vec=(1+abs(x-X(1:6))+abs(x-X(1:6)).^2/3).*exp(-abs(x-X(1:6)));
    vec(3)=k_vec*A(1:6,3);
    k_vec=(1+abs(x-X(1:7))+abs(x-X(1:7)).^2/3).*exp(-abs(x-X(1:7)));
    vec(4)=k_vec*A(1:7,4);
    vec_phi=sparse(1,1:ind+2,vec,1,n);
elseif ind<=1
    k_vec=(1+abs(x-X(1:4))+abs(x-X(1:4)).^2/3).*exp(-abs(x-X(1:4)));
    vec(1)=k_vec*A(1:4,1);
    k_vec=(1+abs(x-X(1:5))+abs(x-X(1:5)).^2/3).*exp(-abs(x-X(1:5)));
    vec(2)=k_vec*A(1:5,2);
    k_vec=(1+abs(x-X(1:6))+abs(x-X(1:6)).^2/3).*exp(-abs(x-X(1:6)));
    vec(3)=k_vec*A(1:6,3);
    vec_phi=sparse(1,1:ind+2,vec,1,n);


elseif ind== (n-4)
    
    parfor i=-3:1
        ind_phi=ind+i;
        k_vec=(1+abs(x-X(ind_phi-3:ind_phi+3))+abs(x-X(ind_phi-3:ind_phi+3)).^2/3).*exp(-abs(x-X(ind_phi-3:ind_phi+3)));
        vec(i+4)=k_vec*A(ind_phi-3:ind_phi+3,ind_phi);
    end
    k_vec=(1+abs(x-X(n-5:n))+abs(x-X(n-5:n)).^2/3).*exp(-abs(x-X(n-5:n)));
    vec(6)=k_vec*A(n-5:n,n-2);
    vec_phi=sparse(1,ind-3:ind+2,vec,1,n);

elseif ind== (n-3)
    
    parfor i=-3:0
        ind_phi=ind+i;
        k_vec=(1+abs(x-X(ind_phi-3:ind_phi+3))+abs(x-X(ind_phi-3:ind_phi+3)).^2/3).*exp(-abs(x-X(ind_phi-3:ind_phi+3)));
        vec(i+4)=k_vec*A(ind_phi-3:ind_phi+3,ind_phi);
    end
    k_vec=(1+abs(x-X(n-5:n))+abs(x-X(n-5:n)).^2/3).*exp(-abs(x-X(n-5:n)));
    vec(5)=k_vec*A(n-5:n,n-2);
    k_vec=(1+abs(x-X(n-4:n))+abs(x-X(n-4:n)).^2/3).*exp(-abs(x-X(n-4:n)));
    vec(6)=k_vec*A(n-4:n,n-1);
    vec_phi=sparse(1,ind-3:ind+2,vec,1,n);

elseif ind== (n-2)
    
    parfor i=-3:-1
        ind_phi=ind+i;
        k_vec=(1+abs(x-X(ind_phi-3:ind_phi+3))+abs(x-X(ind_phi-3:ind_phi+3)).^2/3).*exp(-abs(x-X(ind_phi-3:ind_phi+3)));
        vec(i+4)=k_vec*A(ind_phi-3:ind_phi+3,ind_phi);
    end
    k_vec=(1+abs(x-X(n-5:n))+abs(x-X(n-5:n)).^2/3).*exp(-abs(x-X(n-5:n)));
    vec(4)=k_vec*A(n-5:n,n-2);
    k_vec=(1+abs(x-X(n-4:n))+abs(x-X(n-4:n)).^2/3).*exp(-abs(x-X(n-4:n)));
    vec(5)=k_vec*A(n-4:n,n-1);
    k_vec=(1+abs(x-X(n-3:n))+abs(x-X(n-3:n)).^2/3).*exp(-abs(x-X(n-3:n)));
    vec(6)=k_vec*A(n-3:n,n);
    vec_phi=sparse(1,ind-3:ind+2,vec,1,n);
    
elseif ind==(n-1) 
    
    parfor i=-3:-2
        ind_phi=ind+i;
        k_vec=(1+abs(x-X(ind_phi-3:ind_phi+3))+abs(x-X(ind_phi-3:ind_phi+3)).^2/3).*exp(-abs(x-X(ind_phi-3:ind_phi+3)));
        vec(i+4)=k_vec*A(ind_phi-3:ind_phi+3,ind_phi);
    end
    k_vec=(1+abs(x-X(n-5:n))+abs(x-X(n-5:n)).^2/3).*exp(-abs(x-X(n-5:n)));
    vec(3)=k_vec*A(n-5:n,n-2);
    k_vec=(1+abs(x-X(n-4:n))+abs(x-X(n-4:n)).^2/3).*exp(-abs(x-X(n-4:n)));
    vec(4)=k_vec*A(n-4:n,n-1);
    k_vec=(1+abs(x-X(n-3:n))+abs(x-X(n-3:n)).^2/3).*exp(-abs(x-X(n-3:n)));
    vec(5)=k_vec*A(n-3:n,n);
    vec_phi=sparse(1,ind-3:ind+1,vec,1,n);
elseif ind==n
    k_vec=(1+abs(x-X(n-5:n))+abs(x-X(n-5:n)).^2/3).*exp(-abs(x-X(n-5:n)));
    vec(1)=k_vec*A(n-5:n,n-2);
    k_vec=(1+abs(x-X(n-4:n))+abs(x-X(n-4:n)).^2/3).*exp(-abs(x-X(n-4:n)));
    vec(2)=k_vec*A(n-4:n,n-1);
    k_vec=(1+abs(x-X(n-3:n))+abs(x-X(n-3:n)).^2/3).*exp(-abs(x-X(n-3:n)));
    vec(3)=k_vec*A(n-3:n,n);
    vec_phi=sparse(1,n-2:n,vec,1,n);
end

end