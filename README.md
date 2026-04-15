# Hellinger-Kantorovich Parallel Transport Implementation

This repository contains an implementation of the parallel transport algorithm on Hellinger-Kantorovich (HK) space, based on the paper at `/Users/tristansaidi/Research/HKPT/main.tex`.

## Files

### 1. `hk_parallel_transport.py` - Core Implementation

The source module implementing all algorithms from the paper:

#### Key Classes

- **`EmpiricalMeasure`**: Represents non-negative measures as weighted samples
  - Stores sample locations and weights
  - Provides push-forward operation via transport maps

- **`ConeMeasure`**: Represents measures on the cone metric space
  - Stores spatial coordinates, radial coordinates, and weights
  - Lifts HK measures onto `C_Ω` (metric cone)

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
   - Returns transported tangent vector (v, β) at mu1

6. **`hk_distance(mu0, mu1)`**
   - Computes approximate HK distance via LET functional
   - Equation (407): HK = sqrt(E(π*))

#### Lifting and Projection

- **`lift_tangent()`**: Maps HK tangent (v, β) to cone tangent
- **`project_tangent()`**: Maps cone tangent back to HK tangent
- These maintain the isometry proven in Theorem "Lifting by characteristics two"

#### Support Functions

- **`entropy_function()`**: F(ρ) = ρ log(ρ) - ρ + 1 from LET functional
- **`transport_cost()`**: c(L) = -2 log(cos(L)) for geodesic cost
- **`let_functional()`**: Evaluates E(π; μ₀, μ₁)
- **`compute_monge_map_from_coupling()`**: Extracts deterministic map from coupling
- **`cone_coordinates()` / `project_from_cone()`**: Convert between coordinate systems

### 2. `HK_Parallel_Transport_Visualization.ipynb` - Jupyter Notebook

Interactive notebook illustrating the updated empirical HK construction.

#### Contents

1. **A 1D unbalanced HK path**
   - Starts from two small empirical measures with different total mass
   - Projects the cone interpolation back to the base space
   - Visualizes how mass shifts and changes along the path

2. **Inspecting the cone lift**
   - Displays the discrete LET coupling matrix
   - Examines endpoint radii and their evolution along dominant cone edges
   - Connects projected mass to the cone factor `r^2`

3. **A 2D parallel transport example**
   - Transports a tangent field with both spatial and reaction components
   - Compares source and transported tangents side by side
   - Highlights how the transported field is averaged back from the lifted coupling

4. **Interpretation notes**
   - Summarizes how LET coupling, cone interpolation, and projection interact
   - Emphasizes that the notebook reflects the empirical approximation implemented in code

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
cd /Users/tristansaidi/Research/HKPT
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

```
HK²(μ₀, μ₁) = inf ∫₀¹ (||v_t||²_{L²(μ_t)} + 4|β_t|²_{L²(μ_t)}) dt

subject to: ∂_t μ_t + ∇·(μ_t v_t) = 4β_t μ_t
```

Where:
- **v_t**: spatial velocity (transport component)
- **β_t**: reaction rate (mass creation/destruction)

### The Cone Lift

Key insight: Embed HK geodesics into Wasserstein geodesics on a cone metric space:

- **Base space**: Ω (original sample space)
- **Fiber**: Radial coordinate r ∈ (0, ∞)
- **Metric**: g = dr² + r²(dθ²) (warped product)
- **Measure on cone**: λ_t = (x, r_t(x))_# η_t

The radial function implicitly captures mass growth:
```
μ_t = (B λ_t) = "projection" of λ_t back to base
```

### Parallel Transport

Classical PT on Riemannian manifolds is generalized to metric spaces. The paper:

1. Lifts tangent vectors to cone space
2. Uses explicit cone PT formulas (involving angle integrals)
3. Projects result back to HK space

The transported tangent maintains geometric properties isometrically.

## Algorithm Complexity

For empirical measures with n samples:

- **LET solving**: O(n²) per Sinkhorn iteration (coupling matrix)
- **Isometric lift**: O(N·n²) for N discretization steps
- **Cone PT**: O(N·n) (explicit formulas)
- **Total**: O(N·n²) for the main algorithm

Memory: O(n²) for storing couplings.

## Key References

From main.tex:

1. **LET Functional** (Eq. 395-407): Logarithmic entropy transport
2. **Isometric Lift Algorithm** (Alg:isometric-lift, lines 504-552)
3. **Main PT Algorithm** (Alg: HK parallel transport, lines 717-749)
4. **Cone PT Formulas** (Proposition, lines 887-928)
5. **Convergence Theorem** (Theorem, lines 751-764): Error = O(1/N)

## Future Enhancements

1. **Better OT Solver**: Integrate POT library's unbalanced OT
2. **GPU Acceleration**: Use JAX or PyTorch for large-scale problems
3. **Adaptive Discretization**: Refine N based on geodesic curvature
4. **Manifold Support**: Extend to Riemannian manifold base spaces
5. **Stability Analysis**: Numerical methods validation

## Notes

- **Empirical Approximation**: The cone PT formulas assume smooth geodesics; discrete approximations may accumulate error
- **Monge Map Extraction**: Currently uses argmax on coupling; could use entropic regularization for smoother maps
- **LET Solver**: Simple Sinkhorn; could use more sophisticated methods from POT library
- **Convergence**: Paper proves O(1/N) but requires sufficiently regular geodesics

## Author Notes

This implementation follows the algorithmic descriptions and mathematical formulations in the paper exactly. 

The key innovation is that HK parallel transport becomes computable through:
1. The LET characterization (implementable with standard unbalanced OT)
2. The cone lift (reducing to Wasserstein geometry)
3. Explicit formulas (no need for numerical integration on geodesics)

The visualization notebook builds intuition by showing:
- How masses are transported
- How the radial coordinate encodes mass creation
- How different tangent vector components transform independently
- Convergence properties of the discretization
