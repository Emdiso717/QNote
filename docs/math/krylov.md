# Krylov Subspace Methods

## Basic

!!! question "Why krylov"
    When A is large and sparse , Krylov methods do not deal directly with A,but rather with matrix-vector products .So it is cheep.比如计算 $f(A)$ 等难以处理的函数时，多项式逼近这个函数，也就是 krylov 空间线性组合。

**Krylov Sequence**: A set of Vectors ${b,Ab,A^2b,A^3b,.....}$

**Krylov Subsapces** :$K_m(A,b)=span{b,A,A^2b,.....，A^{m-1}b}$


## Arnoldi Iteration
Arnoldi Iteration that reduces A to upper-Hessnberg Form.Also we are looking for finding a set of orthogonal basis.
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

- 具体算法，不断利用 Schmidt orthogonalization：
```
ALGORITHIM Arnoldi
1. choose a vector v1,s.t. v1 = b / b.norm
2. For j =1,.......m_max Do:
    For i = 1,......j ,Do:
        w = Av(j)
        h(i,j) = Av(j).dot(v(i))
        w = w - h(i,j) - v(i)
    h(j+1,j) = w.norm
    if h(j+1,j) = 0:
        break
    else
        v(j+1) = w / w.norm
```

- 如此算法计算会有 $AV_m = V_{m+1}H_{m+1}$ ,进一步 $AV_m = V_mH_m + v_{m+1}H(m+1,m)e_m^{T}$
- 通常一般会简化成 $A_V_m = V_m H_m$