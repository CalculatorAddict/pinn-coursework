# Physics-Informed Neural Networks: an overview

## 1. Multivariable Calculus

When in doubt, break it down into components!

### Notes on notation

Remember to state whether numerator or denominator convention is being used.
- Numerator: $\frac{df}{d\mathbf x}$ is a row vector (easier to calculate via matrix algebra)
- Denominator: $\frac{df}{d\mathbf x}$ is a column vector (used in optimization)

$\nabla^2$ denotes the Hessian in optimisation contexts, Laplacian in PDE contexts. State which one as well if needed!

### Derivative of vector wrt matrix

$$\frac{d\mathbf{f}}{dA} = \left(\frac{\partial f_i}{\partial A}\right)_{i=1}^M \in \mathbb{R}^{M \times (M \times N)}, \quad \frac{\partial f_i}{\partial A} := \frac{\partial f_i}{\partial a_{jk}} \in \mathbb{R}^{1 \times (M \times N)}$$

### Matrix-Vector Calculus identities

Note: these are in numerator convention

$$\frac{\partial}{\partial X}\mathbf f(X)^\top=\left(\frac{\partial\mathbf f(X)}{\partial X}\right)^\top$$
$$\frac{\partial}{\partial X}\mathbf f(X)^{-1}=-\mathbf f(X)^{-1}\frac{\partial\mathbf f(X)}{\partial X}\mathbf f(X)^{-1}$$
$$\frac{\partial}{\partial X}\mathbf a^\top X\mathbf b=\mathbf a\mathbf b^\top$$
$$\frac{\partial}{\partial\mathbf x}\mathbf x^\top A\mathbf x=\mathbf x^\top(A+A^\top)$$

### A tale of three convexities

| Order | Condition |
|-------|-----------|
| Zero | $f(cx+(1-c)y)\le cf(x)+(1-c)f(y)$ |
| First | $f(y)\ge f(x)+\nabla f(x)^\top(y-x)$ |
| Second | $\nabla^2 f \succeq 0$ (positive semi-definite) |

Definitions are equivalent when function is sufficiently smooth.

### Divergence Theorem

$$\int_\Omega\nabla \cdot F\,dV=\oint_{\partial \Omega}\nabla F\cdot\,dS$$

The total divergence over a closed volume is equal to the flux across its surface.

#### Green's identity

Derive by applying divergence theorem to $F=u\nabla v$, using that $\Delta=\nabla\cdot\nabla$.
$$\int_\Omega(u\Delta v+\nabla u\cdot\nabla v)\,dV=\oint_{\partial \Omega}u\nabla v\cdot\,dS$$
- Analogous to IBP.

## 2. Classical Methods

### Collocation

Ansatz is a linear combination of basis functions $\{\phi_i\}$. Goal is to minimize PINN loss function, but can solve linear system for $\phi_i$ explicitly via conjugate method (rather than approximating optimum as PINN does).

### Which scheme to use?

| PDE type | FD scheme | Stability |
|----------|-----------|-----------|
| Elliptic | Centred differences in space, solve linear system directly | No time-stepping; system solve |
| Parabolic | Forward Euler (explicit) or Backward Euler/CN (implicit) | Fwd: $\Delta t/h^2 \leq 1/2$; Bwd/CN: unconditional |
| Hyperbolic | LF or upwind in space, forward Euler in time | CFL: $c\Delta t/h \leq 1$ |

In PINN all cases are treated the same (PDE residual + BC + IC loss); time-blocking handles long-time problems.

### Von Neumann stability analysis

Just ansatz on approximation: $U_k^n=A(\omega)^n\exp(i\omega kh)$ where $h$ is the mesh coarseness.
- Stability: $|A(\omega)|\le 1$ for all $\omega$.

### $\theta$-Update rules

If $\frac{du}{dt}=f(t,u)$ then the theta-approximations are given by $$\hat u_{t+1}=\hat u_t+(1-\theta) f(t,\hat u_t)+\theta f(t+1,\hat u_{t+1}))\Delta t$$

| $\theta$ | Update Rule | Stability | Convergence Speed | Spatial Accuracy |
|-|-|-|-|-|
| $0$ | forward Euler | stable if $\frac{\Delta t}{h^2} \leq \frac12$ | $O(\Delta t)$ | $O(h^2)$ 
| $\frac12$ | Crank-Nicolson | unconditionally stable | $O(\Delta t^2)$ | $O(h^2)$ 
| $1$ | backward Euler | unconditionally stable | $O(\Delta t)$ | $O(h^2)$ 

### Conservative law

A PDE is a conservative law if it is of form $\partial_tu+\partial_xF(u)=0$.
- Movement in and out of a region over time corresponds exactly to the transportation of $u$ as determined by $F$.

Similarly, a numerical method is in conservative form if it can be written $$U_j^{n+1}=U_j^n-\frac{\Delta t}{h}\left[F(U_j^n,U_{j+1}^n)-F(U_{j-1}^n,U_j^n)\right]$$

#### Lax-Friedrichs numerical flux

$$F_{LF}(U_j, U_{j+1}) = \frac{1}{2}\left(f(U_j) + f(U_{j+1})\right) - \frac{h}{2\Delta t}\left(U_{j+1} - U_j\right)$$

#### Why do we want a Tory law??

If a scheme for a Tory PDE satisfies:

1. Discrete conservation (the flux law holds for any interval)
2. Consistency: $$\lim_{V,W\rightarrow U}F(V,W)=f(U)$$

then the numerical scheme converges to a weak solution of the PDE!

## 3. PINNs

Remember to add weighting to all penalties in the loss function (and mention that they are hyperparameters)!

Assuming that the PDE and ICs/BCs are rearranged s.t. RHS = 0, the PINN loss function is given by something like
$$\mathcal L(\mathcal N)=\sum_{i=1}^{d_\Omega}\text{PDE}[\mathcal N](x^i_\Omega)^2+\sum_k\lambda_k\sum_{i=1}^{d_{\Gamma_k}}\text{BC}_k[\mathcal N](x^i_{\Gamma_k})^2$$

### Approximation theorems

- Universal approximation theorem: NN can approximate any continuous function on compact support to arbitrary precision
- Barron's theorem: NN approximation quality is dependent on width of network, not depth

slot under "## 3. PINNs" before failure modes, as a new "### Optimisation" subsection.

### Optimisation

| Method | Update | Cost/step | Convergence | Notes |
|--------|--------|-----------|-------------|-------|
| GD | $x \leftarrow x - \gamma \nabla f$ | $O(n)$ | linear | step size $\gamma$ critical: too large oscillates, too small slow |
| Newton | $x \leftarrow x - (\nabla^2)^{-1}\nabla f$ | $O(n^3)$ | quadratic | exact Hessian, expensive |
| Rank-1 | $B^{-1} \leftarrow B^{-1} + \alpha uu^\top$, $x\leftarrow x - B^{-1}\nabla f$ | $O(n^2)$ | superlinear | doesn't preserve positive definiteness, can diverge |
| BFGS | $B^{-1} \leftarrow B^{-1} + \alpha uu^\top + \beta vv^\top$, $x\leftarrow x - B^{-1}\nabla f$ | $O(n^2)$ | superlinear | satisfies secant equation $Bs = y$, $s^\top y > 0$ guarantees descent |

**Secant equation**: $B_{k+1}s_k = y_k$ where $s_k = x_{k+1}-x_k$, $y_k = \nabla f_{k+1} - \nabla f_k$. Imposes that $B_{k+1}$ reproduces observed gradient change. The hyperparameters $\alpha$ in Rank-1 and $\alpha,\beta$ in BFGS are picked so that the methods satisfy the secant equation.

For BFGS: positive definite $B^{-1}$ means $d = -B^{-1}\nabla f$ satisfies $\nabla f^\top d < 0$, so guaranteed descent direction. Strictly convex $f$ ensures $s^\top y > 0$ automatically.

Full BFGS stores the $n \times n$ matrix $B^{-1}$, which takes $O(n^2)$ memory. L-BFGS instead stores only the last $m$ pairs $(s_k, y_k)$ and implicitly applies the BFGS update via two-loop recursion -- $O(mn)$ memory and cost. $m = 10$--$20$ typically sufficient.

#### Sherman-Morrison

$$(A + uv^\top)^{-1} = A^{-1} - \frac{A^{-1}uv^\top A^{-1}}{1 + v^\top A^{-1}u}$$

Used in BFGS: track $B^{-1}$ rather than $B$; each rank-2 update to $B^{-1}$ costs $O(n^2)$ via this formula rather than $O(n^3)$ re-inversion.

### Failure modes of PINN

| Failure Mode | Description |
|-------|-----------|
| Spectral bias | High-frequency content learned slowly; network prefers smooth low-frequency solutions |
| Corner singularities | Non-convex domains give solutions with unbounded derivatives near re-entrant corners; gradient blows up, slow convergence |
| Non-smooth forcing | Approximation quality degrades where $f$ or solution lacks regularity |
| Shocks | Discontinuous solutions can't be represented by smooth network; strong-form residual ill-posed at shock; network smears or finds wrong weak solution |
| BC drift | Without explicit BC enforcement in loss, network drifts at boundaries (e.g. advection without periodic BC) |
| Loss imbalance | Competing loss terms trade off -- IC accuracy sacrificed for PDE residual when problem is hard (e.g. small $\varepsilon$ Burgers) |

Higher derivatives mean less accurate loss due to vanishing gradients, but changing to coupled equations makes loss landscape more complex and increases compute required.

### Ritz PINN

Minimizing the integral $J[u] = \int_a^b L(x, u, u') \, dx$ is equivalent to solving the PDE $$\frac{\partial L}{\partial u} - \frac{d}{dx}\frac{\partial L}{\partial u'} = 0.$$

So can change to loss function to reflect this. Higher-order derivatives are not in $L$ so computation becomes cheaper. Furthermore the complexity added by converting to a coupled system is not introduced.

Approximate integral in loss function via weighted sum over integrand. Weights $\omega_i$ are given by the area of the Voronoi cell around datapoints $x^i$.