# Kernel-Packet
Codes for paper "Kernel Packet: An Exact and Scalable Algorithm for Gaussian Process Regression with Matérn Correlations"

The mat-12,32,52 folder are codes for KP factorization of Matern-1/2, 3/2, and 5/2 kernel, respectively. compute_APhi (compute_A for Matern-1/2) returns the kernel covariance matrix factorization AK=Phi where A is one-banded matrix for Matern-1/2 , two-banded matrix for Matern-3/2, and three-banded matrix for Matern-5/2 and Phi is the identity matrix for Matern-1/2 , one-banded matrix for Matern-3/2, and two-banded matrix for Matern-5/2.

demo.m in each folder is an example of using KP to solve the Gaussian Process regression k(x,X)K^{-1}Y using KP.
