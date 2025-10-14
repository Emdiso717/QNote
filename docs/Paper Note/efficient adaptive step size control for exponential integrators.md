# Efficient adaptive step size control for exponential integrators

!!! info 
    这篇文章关键在于对于原本的指数积分器实现了一个步长自适应的机制。关注点也先放在这个机制上

## Leja interpolation
[TODO]

## Adaptive Step-size Controller

首先，对于常见的显示，或者一般的隐式方法求解时，无论步长是多少，每一步的计算成本是固定的，隐式固定在于对于性质良好矩阵进行LU分解。当衡量每一步的最大步长时，往往是通过每一步的截断误差判断。对于这样固定的开销的方法，截断误差也往往是线性的 $e^n = D(\Delta t^n)^{p+1}$ p是方法的阶数。 通常会设置一个 tol 来限制误差。 所以很容易推倒出下一步步长 $\Delta t^{n+1} = \Delta t^{n}(\frac{tol}{e^n})^{\frac{1}{p+1}}$

但是对于使用每一步开销需要使用迭代法求解的，比如 krylov 等，就不能简单考虑成线性求解了。此时估计一个开销函数 $c^n = \frac{i^n}{\Delta t^n}$ 来计算每一个时间下的开销。 目的就是使得 $c^n t $ 结果最小 ,于是使用梯度下降，$T^{n+1} = T^{n} - k ∇C^n(T^n)$ ，并将梯度结果近似为 ：$\frac{C^n(T^n)-C^{n-1}(T^{n-1})}{T^n-T^{n-1}}$