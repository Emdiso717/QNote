# A Stiffly Accurate Integrator for Elastodynamic Problems

!!! info
    本篇文章主要介绍一个新的 时间积分近似仿真算法 应用于 弹性系统。提出三个关键点：
    1. 重构对于原问题的数学表示
    2. 用指数近似非线性算子
    3. 自适应 Krylov 投影算法优化的刚性精确指数方法
    重构之后 能够使用 指数积分器 EPIRK类型进行迭代求解。
    更加接近于计算机工程实际的视觉渲染等应用，而非纯数值计算验证。

## Reformulating Elastodynamic Systems
这个部分介绍了经典的 将二阶ODE 转化为 一阶ODE 的过程，对于一个常规的牛顿等式：

$$
Mx^{\prime \prime}(t) + D x^{\prime}(t) + Kx(t) = f(x(t))
$$

忽略阻尼部分，并且同时乘以M的逆矩阵，$L = M^{-1}K$,则得到的结果为：$x^{\prime \prime}(t) + Lx(t) = g(x(t))$

此时将方程转化为一个二阶的ODE，但是常用的积分求解都是面对一阶ODE，于是需要进一步转化，使用 $X(t)=(x(t),x^{\prime}(t))^T$，于是原方程可以写成如下形式：
$$
X^{\prime}(t)=\begin{pmatrix}
0&I\\\\
-L&0\end{pmatrix}X(t) + \begin{pmatrix} 0 \\\\ g(x) \end{pmatrix} \\\\
X^{\prime}(t)=UX(t) + G(X(t))
$$

??? note "关于系统高频和低频"
    首先，对于一个 stiff system 而言，高频和低频的特征差异比较大。意味着，高频特征使系统变化快，而低频变换慢。就如同一块布在摆动时，受到高频的内部弹力导致其产生明显的纹理变化，同时受到低频的重力，产生并不明显的形变化。
    大多数情况这个刚性的引入通过 $U$引入的，对于不同的积分方法，对于特征的保留并不同，这个样便体现出了step size 和收敛性的关系：
    - explicit ： 与 $U$ 的特征值有关，通常在特征值比较大or步长比较大时，$Uh$就会很大使得求解不稳定。
    - Implicit ：这个通常与$(I-hU)^{-1}$ 的特征值有关系，当h较大时，特征值反而小，于是隐式方法在步长稳定性上表现良好。但是这个矩阵求逆往往开销很大。
    -  Semi-Implicit 是把线性部分做隐式求解，其他部分是显示。这种情况对于外力影响不大的系统十分友好，但是对于外力影响刚度大的系统，此时求解往往会忽略刚度的影响。
    - 对于指数求解器，收敛性往往与 $exp(hU)$关联，对于大的负特征值，会降低这个指数部分的值，能够有效降频率。
    最后，对于更加普遍的情况而言，通常考虑系统的 Jacobian matrix ：$x^{\prime} = F(x)$,通常对于 $F$ 进行泰勒展开，并保留一阶项，忽略高阶小量。这样最后结果其实便是 $x^{\prime} = Jx$ 那么系统的刚性就和 J 有关（在之前的部分中 就是 $U+g^{\prime}$）
        - explicit:没有考虑刚性 所以容易不收敛
        - implicit ：准确来看应该是 $(I-hJ)^{-1}$ 考虑所有刚性
        - Semi ：只考虑了线性部分的刚性

之后，为了将原方程获得一个良好的辛方程的表达，于是进行了如下代换 $\Omega = \sqrt{L},X(t)=\begin{pmatrix}\Omega x(t) \\\\ x^{\prime}(t)\end{pmatrix}$ 于是结果获得如下形式：

$$
    X_{\prime}=F(X(t)) = \begin{pmatrix} 0 & -\Omega \\\\ \Omega & 0 \end{pmatrix} X(t) + G(X(t))
$$

## Exponential Integration 
这个部分和前一篇论文中解释一样，使用指数积分器求解。在本篇中重点介绍 EPIRK Methods 一个类 荣格库塔 形式的积分器。

$$
X_{ni} = X_n + a_{i1} \psi_{i1}(g_{i1} h_n A_{i1}) h_n F_n \\
\quad + h_n \sum_{j=2}^{i-1} a_{ij} \psi_{ij}(g_{ij} h_n A_{ij}) \Delta^{(j-1)} R(X_n), 
\quad i = 2, \ldots, s, \\
X_{n+1} = X_n + b_{1} \psi_{s+1,1}(g_{s+1,1} h_n A_{i1}) h_n F_n \\
\quad + h_n \sum_{j=2}^{s} b_{j} \psi_{s+1,j}(g_{s+1,j} h_n A_{ij}) 
\Delta^{(j-1)} R(X_n),
$$

这里 $X_{ni}$ 都是中间项，对于系数求解部分 关注本文做了一个简单介绍。

首先是需要保证展开之后的低阶项的系数一致。但是通常 J 的谱半径过大，虽然系数带来误差很小，但是会被 谱半径 放大。但是这里还是有无数种系数的取值。

于是使用 Stiff Accuracy 来限制系数，是的误差更小。

## STABILITY AND CONVERGENCE ANALYSIS 
在这个文章中 重点放在了总体误差上，省略计算过程可以知道 迭代n+1次之后的误差 $\|E_{n+1}\| \leq Ch^k$ 

但是在推导过程中有一个离散算子的想法可以进行借鉴。

对于原方程 $x_n+h\phi_1(hJ)F(X_n)$ 和 $e^{jh}X_n$ 是等价的。
所以定义一个离散算子 $D(hJ_n)$则此时 $X_{n+1} = D(hJ_n)X_n = \Pi D X_0$

如果不使用数值方法拟合积分，则此时算子应当可以近似成 $e^{hJ}$ 的幂次。此时就如之前所述的一样， 指数能够有效的屏蔽高频特征，凸显低频特征。