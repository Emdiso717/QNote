# Krylov Subspace Methods

## Basic

!!! question "Why krylov"
    When A is large and sparse , Krylov methods do not deal directly with A,but rather with matrix-vector products .So it is cheep.

**Krylov Sequence**: A set of Vectors ${b,Ab,A^2b,A^3b,.....}$

**Krylov Subsapces** :$\K_m(A,b)=span{b,A,A^2b,.....，A^{m-1}b}$


## Arnoldi Iteration
Arnoldi Iteration that reduces A to upper-Hessnberg Form

**Hessenberg Form** 
- **Upper-Hessenberg** : 类似上三角矩阵 $a_{ij}=0, i>j+1$

$$
\begin{pmatrix}
    h_{1,1} & h_{1,2} & \dots & h_{1,m-1} & h_{1,m}\\\\
    h_{2,1} & h_{2,2} & \dots &h_{2,m-1} & h_{2,m}\\\\
    0 & h_{3,2} & \dots &h_{3,m-1} & h_{3,m}\\\\
    \vdots & \vdots & \ddots&\ddots & \vdots\\\\
    0 & 0 & \dots & h_{m,m-1} & h_{m,m}\\\\
    0 & 0 & \dots & 0 & h_{m+1,m}
\end{pmatrix}
$$

- **Lower-Hessenberg** : 类似上三角矩阵 $a_{ij}=0, j>i+1$