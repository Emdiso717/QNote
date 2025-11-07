# Exponential Propagation Iterative EPI


## Exponential Time Differencing
相比于之前介绍的指数积分器（EPI），这里这个方法是更为传统的一个方法：
$$
\hat{u} = cu + F(u,t)
$$
对原ODE进行展开成两部分 “线性+非线性” ，之后两边同时积分,就的到一个我们十分熟悉的式子（但是注意与之前介绍的方法的区别，在指数部分这里是 $c$,而之前是 $J_n$）
$$
\u(t_{n+1}) = u(t_n)e^{ch} + e^{ch} \int^f_0 e^{-c\theta}F(u(t_n + \theta , t_n+\theta))d\theta
$$
于是会有一比较简单的形式：
- **ETD1:** $u_{n+1} = u_ne^{ch} + F_n(e^{ch}-1)/$$
