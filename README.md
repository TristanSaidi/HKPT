# Hellinger-Kantorovich Parallel Transport Implementation

This repository contains an implementation of the parallel transport algorithm on Hellinger-Kantorovich (HK) space, based on the paper at `/Users/tristansaidi/Research/HKPT/main.tex`.

## Files

### Core Implementation: `hk_parallel_transport.py`

The source module implementing all algorithms from the paper:

#### Key Classes

- **`EmpiricalMeasure`**: Represents non-negative measures as weighted samples
  - Stores sample locations and weights
  - Provides push-forward operation via transport maps

- **`ConeMeasure`**: Represents measures on the cone metric space
  - Stores spatial coordinates, radial coordinates, and weights
  - Lifts HK measures onto $C_\Omega$ (metric cone)

#### Main Functions

1. **`solve_let_unbalanced_transport(mu0, mu1, ..., method='pot_mm')`**
   - Solves the discrete Logarithmic Entropy Transport (LET) problem
   - Default backend uses POT's non-entropic `mm_unbalanced` solver
   - Alternate backend `method='lbfgsb'` keeps the in-module dense direct solver
   - Returns optimal coupling and cost matrix

2. **`let_lift(mu0, mu1, N)`**
   - Builds the endpoint LET cone lift and its cone geodesic interpolation
   - Useful when you want the direct endpoint lift induced by a single LET solve
   - Returns list of `ConeMeasure` objects and radii at each sampled time

3. **`isometric_lift(mu0, mu1, N)`**
   - Implements the recursive isometric-lift algorithm box from `main.tex`
   - Uses the endpoint LET lift only to obtain the intermediate base measures
   - Rebuilds the lifted path by local LET solves, barycentric maps, and radial updates
   - Returns list of `ConeMeasure` objects and radii at each step

4. **`cone_parallel_transport_explicit(a0, b0, r_vals, v_geodesic, t_eval)`**
   - Explicit Cone PT formula (Proposition, lines 887-928)
   - Handles both radial (q=0) and non-radial (q≠0) cases
   - Returns transported tangent components (a_t, b_t)

5. **`hk_parallel_transport(mu0, mu1, u0, N)`**
   - Main Algorithm: Approximate HK Parallel Transport via Cone Transport
   - Algorithm at lines 717-749 in paper
   - Steps:
     1. Lift to cone via `isometric_lift`
     2. Apply cone PT using explicit formulas
     3. Project back to HK tangent space
   - Returns transported tangent vector $(v, \beta)$ at `mu1`

6. **`hk_distance(mu0, mu1)`**
   - Computes approximate HK distance via LET functional
   - Equation (407): HK = sqrt(E(π*))

#### Lifting and Projection

- **`lift_tangent()`**: Maps HK tangent $(v, \beta)$ to cone tangent
- **`project_tangent()`**: Maps cone tangent back to HK tangent
- These maintain the isometry proven in Theorem "Lifting by characteristics two"

#### Support Functions

- **`entropy_function()`**: F(ρ) = ρ log(ρ) - ρ + 1 from LET functional
- **`transport_cost()`**: c(L) = -2 log(cos(L)) for geodesic cost
- **`let_functional()`**: Evaluates $E(\pi; \mu_0, \mu_1)$
- **`compute_monge_map_from_coupling()`**: Extracts deterministic map from coupling
- **`cone_coordinates()` / `project_from_cone()`**: Convert between coordinate systems

## Usage

### Installation

Requirements:
```bash
pip install numpy scipy matplotlib POT
```

The default LET backend uses POT. To force the older in-module solver instead,
pass `method='lbfgsb'` to `solve_let_unbalanced_transport(...)` or
`let_solver='lbfgsb'` to `hk_distance(...)`, `let_lift(...)`,
`isometric_lift(...)`, and `hk_parallel_transport(...)`.

### Running the Notebook

```bash
cd <path/to/HKPT/>
jupyter notebook HK_Parallel_Transport_Visualization.ipynb
```

### In Python

```python
from hk_parallel_transport import (
    EmpiricalMeasure, hk_parallel_transport, hk_distance
)
import numpy as np

# Create empirical measures
n = 50
samples_0 = np.random.randn(n, 2) * 0.5
samples_1 = np.random.randn(n, 2) * 1.0 + np.array([2, 1])

mu0 = EmpiricalMeasure(samples_0, np.ones(n))
mu1 = EmpiricalMeasure(samples_1, np.ones(n))

# Define tangent vector at mu0
v0 = np.random.randn(n, 2) * 0.1
beta0 = np.ones(n) * 0.05
u0 = (v0, beta0)

# Compute parallel transport
u1 = hk_parallel_transport(mu0, mu1, u0, N=10)
v1, beta1 = u1

# Compute HK distance
dist = hk_distance(mu0, mu1)
```

## Mathematical Background

### The Hellinger-Kantorovich Space

The HK distance on non-negative measures allows both transport and mass creation/destruction:

```math
\text{HK}^2(\mu_0, \mu_1)
= \inf \int_0^1 \left(\|v_t\|_{L^2(\mu_t)}^2 + 4|\beta_t|_{L^2(\mu_t)}^2\right)\,dt
```

subject to

```math
\partial_t \mu_t + \nabla \cdot (\mu_t v_t) = 4 \beta_t \mu_t
```

Where:
- **$v_t$**: spatial velocity (transport component)
- **$\beta_t$**: reaction rate (mass creation/destruction)

### The Cone Lift

Key insight: embed HK geodesics into Wasserstein geodesics on a cone metric space:

- **Base space**: $\Omega$ (original sample space)
- **Fiber**: radial coordinate $r \in (0, \infty)$
- **Metric**: $g = dr^2 + r^2(d\theta^2)$ (warped product)
- **Measure on cone**: $\lambda_t = (x, r_t(x))_{\\#}\eta_t$

The radial function implicitly captures mass growth:

```math
\mu_t = \mathfrak{B}\lambda_t
```

that is, $\mu_t$ is the projection of $\lambda_t$ back to the base.
