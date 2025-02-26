# Fisher-Kolmogorov equations

$\begin{cases}\Large\frac{\delta c}{\delta t}\normalsize-\nabla\cdot(\underline D\nabla c)-\alpha c(1-c)=0\\
(\underline D\nabla c)\underline n=0\\
c_{t=0}=c_0\end{cases}$

$c$: concentration of misfolded protein ($0\leq c \leq 1$)

$\underline D=\underline D_{\tilde p}$: diffusion coefficient of misfolded protein.<br>
If it is isotropic it can be a scalar ($D\in\mathbb{R}$) instead of a matrix.<br>
If it is anisotropic $\underline D\in M^{n\times n},\qquad\underline D=d^\text{ext}\underline I+d^\text{axn}(\bold{n}\bigotimes\bold{n})$<br>
where $\bold n$ is the direction of axonal diffusion<br>
Axonal transport is usually faster than extracellular diffusion ($d^\text{axn}\geq d^\text{ext}$)<br>

$\alpha$: growth of the concentration

### Interesting things not useful for the project
$\alpha=k_{12}\Large\frac{k_0}{k_1}\normalsize-\tilde{k_1}$<br>
$k_0$: production rate of healthy protein<br>
$k_1$: clearance rate of healthy protein<br>
$\tilde{k_1}$: clearance rate of misfolded protein<br>
$k_{12}$: conversion rate from healthy to misfolded<br>
$c=\Large\frac{\tilde p}{\tilde p_\text{max}}$<br>
$p$: concentration of healthy protein<br>
$\tilde p$: concentration of misfolded protein<br>
$\tilde p_\text{max}=\Large\frac{k_1^2}{k_{12}^2k_0}\normalsize\alpha$