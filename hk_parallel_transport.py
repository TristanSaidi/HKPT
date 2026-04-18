"""
Empirical Hellinger-Kantorovich parallel transport utilities.

This module follows the paper's cone-lifting construction in an empirical setting:

1. solve the discrete logarithmic entropy transport (LET) problem
2. build the induced endpoint lifts on the cone from the exact discrete marginals
3. interpolate the lifted coupling by cone geodesics and project back to the base
4. lift an HK tangent at the source, transport it incrementally on the cone along
   the characteristic lift, and project the result back to the target empirical
   measure

The implementation targets empirical measures with varying total mass rather than
the population objects in the paper, so it should be read as a discrete
approximation of the continuous theory.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

try:
    import ot
except ImportError:  # pragma: no cover - optional dependency
    ot = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


_TOL = 1e-12


def _maybe_tqdm(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


# ============================================================================
# Utility classes
# ============================================================================


class EmpiricalMeasure:
    """Represents a non-negative empirical measure as weighted samples."""

    def __init__(self, samples: np.ndarray, weights: np.ndarray):
        samples = np.asarray(samples, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        if weights.ndim != 1:
            raise ValueError("weights must be a one-dimensional array")
        if samples.shape[0] != weights.shape[0]:
            raise ValueError("samples and weights must have the same length")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")

        self.samples = samples
        self.weights = weights
        self.n_samples = samples.shape[0]
        self.d = samples.shape[1]

    def push_forward(self, T: Callable[[np.ndarray], np.ndarray]) -> "EmpiricalMeasure":
        """Apply a pointwise map T to the measure."""
        new_samples = np.asarray([T(sample) for sample in self.samples], dtype=float)
        return EmpiricalMeasure(new_samples, self.weights.copy())

    def total_mass(self) -> float:
        return float(np.sum(self.weights))


class ConeMeasure:
    """Represents an empirical measure on the cone."""

    def __init__(self, samples: np.ndarray, radii: np.ndarray, weights: np.ndarray):
        samples = np.asarray(samples, dtype=float)
        radii = np.asarray(radii, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        if radii.ndim != 1 or weights.ndim != 1:
            raise ValueError("radii and weights must be one-dimensional arrays")
        if not (samples.shape[0] == radii.shape[0] == weights.shape[0]):
            raise ValueError("samples, radii, and weights must have the same length")
        if np.any(radii < 0):
            raise ValueError("radii must be non-negative")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")

        self.samples = samples
        self.radii = radii
        self.weights = weights
        self.n_samples = samples.shape[0]
        self.d = samples.shape[1]

    def to_cone_samples(self) -> np.ndarray:
        return np.hstack([self.samples, self.radii[:, np.newaxis]])

    def total_mass(self) -> float:
        return float(np.sum(self.weights))


@dataclass
class LiftedCoupling:
    """Edgewise representation of the lifted coupling on the cone."""

    source_positions: np.ndarray
    target_positions: np.ndarray
    source_radii: np.ndarray
    target_radii: np.ndarray
    weights: np.ndarray
    source_indices: np.ndarray
    target_indices: np.ndarray

    def lambda_source(self) -> ConeMeasure:
        return ConeMeasure(self.source_positions, self.source_radii, self.weights)

    def lambda_target(self) -> ConeMeasure:
        return ConeMeasure(self.target_positions, self.target_radii, self.weights)


@dataclass
class LocalHKStep:
    """Discrete local HK data produced during isometric lifting."""

    map_positions: np.ndarray
    q: np.ndarray
    v: np.ndarray
    beta: np.ndarray
    row_mass: np.ndarray
    col_mass: np.ndarray
    coupling: Optional[np.ndarray] = None


def _validate_hk_tangent(
    tangent_hk: Tuple[np.ndarray, np.ndarray],
    measure: EmpiricalMeasure,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(tangent_hk, EmpiricalMeasure):
        raise TypeError(
            "hk_exponential_map expects a tangent tuple (v, beta), not a target measure. "
            "To get the HK tangent from a source and target empirical measure, call "
            "`hk_logarithmic_map(mu_source, mu_target)`."
        )

    if not isinstance(tangent_hk, tuple) or len(tangent_hk) != 2:
        raise TypeError("tangent_hk must be a tuple (v, beta)")

    v, beta = tangent_hk
    v = np.asarray(v, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if v.shape != (measure.n_samples, measure.d):
        raise ValueError("v must have shape (n_samples, d)")
    if beta.shape != (measure.n_samples,):
        raise ValueError("beta must have shape (n_samples,)")
    return v, beta


# ============================================================================
# LET functional and solver
# ============================================================================


def entropy_function(rho: np.ndarray) -> np.ndarray:
    """F(rho) = rho log rho - rho + 1 with F(0) = 1."""
    rho = np.asarray(rho, dtype=float)
    result = np.ones_like(rho)
    positive = rho > 0
    result[positive] = rho[positive] * np.log(rho[positive]) - rho[positive] + 1.0
    return result


def transport_cost(dist: float) -> float:
    """c(L) = -2 log(cos(L)) for L < pi/2, infinity otherwise."""
    if dist >= np.pi / 2:
        return np.inf
    return float(-2.0 * np.log(np.cos(dist)))


def _pairwise_transport_cost(
    mu0: EmpiricalMeasure, mu1: EmpiricalMeasure
) -> Tuple[np.ndarray, np.ndarray]:
    ground_dist = cdist(mu0.samples, mu1.samples, metric="euclidean")
    cost_matrix = np.full_like(ground_dist, np.inf, dtype=float)
    admissible = ground_dist < (np.pi / 2)
    cost_matrix[admissible] = -2.0 * np.log(np.cos(ground_dist[admissible]))
    return ground_dist, cost_matrix


def _discrete_entropy_term(marginal: np.ndarray, reference: np.ndarray) -> float:
    if np.any((reference <= _TOL) & (marginal > _TOL)):
        return np.inf

    term = 0.0
    positive_reference = reference > _TOL
    if np.any(positive_reference):
        ratio = np.zeros_like(marginal)
        ratio[positive_reference] = marginal[positive_reference] / reference[positive_reference]
        term += float(np.sum(reference[positive_reference] * entropy_function(ratio[positive_reference])))
    return term


def let_functional(
    pi_matrix: np.ndarray,
    mu0: EmpiricalMeasure,
    mu1: EmpiricalMeasure,
    pi0_marginal: np.ndarray,
    pi1_marginal: np.ndarray,
    ground_dist: np.ndarray,
) -> float:
    """Evaluate the discrete LET functional."""
    cost_matrix = np.full_like(ground_dist, np.inf, dtype=float)
    admissible = ground_dist < (np.pi / 2)
    cost_matrix[admissible] = -2.0 * np.log(np.cos(ground_dist[admissible]))

    if np.any((~admissible) & (pi_matrix > _TOL)):
        return np.inf

    term1 = _discrete_entropy_term(pi0_marginal, mu0.weights)
    term2 = _discrete_entropy_term(pi1_marginal, mu1.weights)
    term3 = float(np.sum(pi_matrix[admissible] * cost_matrix[admissible]))
    return term1 + term2 + term3


def _let_objective_and_gradient(
    coupling_values: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    cost_values: np.ndarray,
    mu0_weights: np.ndarray,
    mu1_weights: np.ndarray,
    n0: int,
    n1: int,
) -> Tuple[float, np.ndarray]:
    row_mass = np.bincount(row_idx, weights=coupling_values, minlength=n0)
    col_mass = np.bincount(col_idx, weights=coupling_values, minlength=n1)

    term1 = _discrete_entropy_term(row_mass, mu0_weights)
    term2 = _discrete_entropy_term(col_mass, mu1_weights)
    if not np.isfinite(term1) or not np.isfinite(term2):
        return np.inf, np.full_like(coupling_values, np.inf)

    objective = term1 + term2 + float(np.dot(coupling_values, cost_values))

    row_log = np.zeros(n0, dtype=float)
    positive_row_ref = mu0_weights > _TOL
    row_log[positive_row_ref] = np.log(
        np.maximum(row_mass[positive_row_ref], _TOL) / mu0_weights[positive_row_ref]
    )

    col_log = np.zeros(n1, dtype=float)
    positive_col_ref = mu1_weights > _TOL
    col_log[positive_col_ref] = np.log(
        np.maximum(col_mass[positive_col_ref], _TOL) / mu1_weights[positive_col_ref]
    )

    gradient = row_log[row_idx] + col_log[col_idx] + cost_values
    return objective, gradient


def solve_let_unbalanced_transport_lbfgsb(
    mu0: EmpiricalMeasure,
    mu1: EmpiricalMeasure,
    max_iterations: int = 500,
    stop_threshold: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the discrete LET problem directly over the coupling matrix via L-BFGS-B.
    """
    n0 = mu0.n_samples
    n1 = mu1.n_samples
    ground_dist, cost_matrix = _pairwise_transport_cost(mu0, mu1)

    feasible = np.isfinite(cost_matrix)
    feasible &= mu0.weights[:, np.newaxis] > _TOL
    feasible &= mu1.weights[np.newaxis, :] > _TOL

    if not np.any(feasible):
        return np.zeros((n0, n1), dtype=float), cost_matrix

    row_idx, col_idx = np.nonzero(feasible)
    cost_values = cost_matrix[feasible]

    scale = max(mu0.total_mass(), mu1.total_mass(), 1.0)
    initial = (
        (mu0.weights[row_idx] * mu1.weights[col_idx]) / scale
        * np.exp(-np.minimum(cost_values, 50.0))
    )
    initial = np.maximum(initial, 1e-16)

    def objective(values: np.ndarray) -> Tuple[float, np.ndarray]:
        return _let_objective_and_gradient(
            values, row_idx, col_idx, cost_values, mu0.weights, mu1.weights, n0, n1
        )

    result = minimize(
        fun=lambda x: objective(x)[0],
        x0=initial,
        jac=lambda x: objective(x)[1],
        bounds=[(0.0, None)] * len(initial),
        method="L-BFGS-B",
        options={"maxiter": max_iterations, "ftol": stop_threshold, "gtol": stop_threshold},
    )

    acceptable_status = result.success or "ITERATIONS REACHED LIMIT" in result.message.upper()
    if not acceptable_status:
        raise RuntimeError(f"LET optimization failed: {result.message}")

    pi = np.zeros((n0, n1), dtype=float)
    pi[feasible] = np.maximum(result.x, 0.0)

    final_energy = let_functional(
        pi,
        mu0,
        mu1,
        pi.sum(axis=1),
        pi.sum(axis=0),
        ground_dist,
    )
    if not np.isfinite(final_energy):
        raise RuntimeError("LET optimizer returned a coupling with infinite energy")

    return pi, cost_matrix


def solve_let_unbalanced_transport_pot(
    mu0: EmpiricalMeasure,
    mu1: EmpiricalMeasure,
    max_iterations: int = 500,
    stop_threshold: float = 1e-8,
    reg_m: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the discrete LET problem with POT's non-entropic MM unbalanced solver.

    This backend matches the discrete LET objective when all pairwise distances
    are strictly below pi/2, so the cost matrix is finite everywhere.
    """
    if ot is None:
        raise ImportError(
            "POT is required for method='pot_mm'. Install it with `pip install POT`."
        )

    ground_dist, cost_matrix = _pairwise_transport_cost(mu0, mu1)
    if np.any(~np.isfinite(cost_matrix)):
        raise ValueError(
            "method='pot_mm' requires all pairwise distances to stay below pi/2 so "
            "the LET cost matrix is finite everywhere. Use method='lbfgsb' otherwise."
        )

    pi = ot.unbalanced.mm_unbalanced(
        mu0.weights,
        mu1.weights,
        cost_matrix,
        reg_m=reg_m,
        reg=0.0,
        div="kl",
        numItermax=max_iterations,
        stopThr=stop_threshold,
    )
    pi = np.asarray(pi, dtype=float)
    pi = np.maximum(pi, 0.0)

    final_energy = let_functional(
        pi,
        mu0,
        mu1,
        pi.sum(axis=1),
        pi.sum(axis=0),
        ground_dist,
    )
    if not np.isfinite(final_energy):
        raise RuntimeError("POT mm_unbalanced returned a coupling with infinite energy")

    return pi, cost_matrix


def solve_let_unbalanced_transport_pot_entropic(
    mu0: EmpiricalMeasure,
    mu1: EmpiricalMeasure,
    entropy_reg: float = 0.01,
    max_iterations: int = 500,
    stop_threshold: float = 1e-8,
    reg_m: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the discrete LET problem with POT's entropic unbalanced Sinkhorn solver.
    """
    if ot is None:
        raise ImportError(
            "POT is required for method='pot_sinkhorn'. Install it with `pip install POT`."
        )
    if entropy_reg <= 0:
        raise ValueError("entropy_reg must be positive for method='pot_sinkhorn'")

    ground_dist, cost_matrix = _pairwise_transport_cost(mu0, mu1)
    if np.any(~np.isfinite(cost_matrix)):
        raise ValueError(
            "method='pot_sinkhorn' requires all pairwise distances to stay below pi/2 so "
            "the LET cost matrix is finite everywhere. Use method='lbfgsb' otherwise."
        )

    pi = ot.unbalanced.sinkhorn_unbalanced(
        mu0.weights,
        mu1.weights,
        cost_matrix,
        reg=entropy_reg,
        reg_m=reg_m,
        numItermax=max_iterations,
        stopThr=stop_threshold,
    )
    pi = np.asarray(pi, dtype=float)
    pi = np.maximum(pi, 0.0)

    final_energy = let_functional(
        pi,
        mu0,
        mu1,
        pi.sum(axis=1),
        pi.sum(axis=0),
        ground_dist,
    )
    if not np.isfinite(final_energy):
        raise RuntimeError("POT sinkhorn_unbalanced returned a coupling with infinite energy")

    return pi, cost_matrix


def solve_let_unbalanced_transport(
    mu0: EmpiricalMeasure,
    mu1: EmpiricalMeasure,
    entropy_reg: float = 0.01,
    max_iterations: int = 500,
    method: str = "pot_mm",
    stop_threshold: float = 1e-8,
    reg_m: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the discrete LET problem with a selectable backend.

    Supported methods:
    - 'pot_mm': POT's non-entropic MM unbalanced OT solver
    - 'pot_sinkhorn': POT's entropic unbalanced Sinkhorn solver
    - 'lbfgsb': the in-module direct dense L-BFGS-B solver
    """
    if method == "pot_mm":
        return solve_let_unbalanced_transport_pot(
            mu0,
            mu1,
            max_iterations=max_iterations,
            stop_threshold=stop_threshold,
            reg_m=reg_m,
        )
    if method == "pot_sinkhorn":
        return solve_let_unbalanced_transport_pot_entropic(
            mu0,
            mu1,
            entropy_reg=entropy_reg,
            max_iterations=max_iterations,
            stop_threshold=stop_threshold,
            reg_m=reg_m,
        )
    if method == "lbfgsb":
        return solve_let_unbalanced_transport_lbfgsb(
            mu0,
            mu1,
            max_iterations=max_iterations,
            stop_threshold=stop_threshold,
        )
    raise ValueError("method must be one of 'pot_mm', 'pot_sinkhorn', or 'lbfgsb'")


# ============================================================================
# Cone geometry helpers
# ============================================================================


def cone_coordinates(sample: np.ndarray, r: float) -> np.ndarray:
    return np.concatenate([np.asarray(sample, dtype=float), [float(r)]])


def project_from_cone(cone_sample: np.ndarray) -> Tuple[np.ndarray, float]:
    cone_sample = np.asarray(cone_sample, dtype=float)
    return cone_sample[:-1], float(cone_sample[-1])


def cone_distance(
    x0: np.ndarray, r0: float, x1: np.ndarray, r1: float
) -> float:
    theta = min(np.linalg.norm(np.asarray(x1) - np.asarray(x0)), np.pi)
    return float(np.sqrt(max(r0 * r0 + r1 * r1 - 2.0 * r0 * r1 * np.cos(theta), 0.0)))


def cone_geodesic_step(
    x0: np.ndarray, r0: float, x1: np.ndarray, r1: float, s: float
) -> Tuple[np.ndarray, float]:
    if s <= 0.0:
        return np.asarray(x0, dtype=float).copy(), float(r0)
    if s >= 1.0:
        return np.asarray(x1, dtype=float).copy(), float(r1)

    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)

    if r0 <= _TOL and r1 <= _TOL:
        return x0.copy(), 0.0
    if r0 <= _TOL:
        return x1.copy(), float(s * r1)
    if r1 <= _TOL:
        return x0.copy(), float((1.0 - s) * r0)

    theta = min(np.linalg.norm(x1 - x0), np.pi)
    cos_theta = np.cos(theta)
    radius_sq = (
        (1.0 - s) ** 2 * r0**2
        + s**2 * r1**2
        + 2.0 * s * (1.0 - s) * r0 * r1 * cos_theta
    )
    radius = float(np.sqrt(max(radius_sq, 0.0)))

    if theta <= _TOL or radius <= _TOL:
        return x0.copy(), radius

    cos_arg = ((1.0 - s) * r0 + s * r1 * cos_theta) / radius
    cos_arg = float(np.clip(cos_arg, -1.0, 1.0))
    rho = float(np.arccos(cos_arg) / theta)
    position = (1.0 - rho) * x0 + rho * x1
    return position, radius


def interpolate_lifted_measure(coupling: LiftedCoupling, s: float) -> ConeMeasure:
    positions = np.zeros_like(coupling.source_positions)
    radii = np.zeros_like(coupling.source_radii)

    for idx in range(coupling.weights.shape[0]):
        positions[idx], radii[idx] = cone_geodesic_step(
            coupling.source_positions[idx],
            coupling.source_radii[idx],
            coupling.target_positions[idx],
            coupling.target_radii[idx],
            s,
        )

    return ConeMeasure(positions, radii, coupling.weights.copy())


def project_cone_measure(cone_measure: ConeMeasure) -> EmpiricalMeasure:
    projected_weights = cone_measure.weights * cone_measure.radii**2
    return EmpiricalMeasure(cone_measure.samples.copy(), projected_weights)


# ============================================================================
# Lift construction
# ============================================================================


def compute_monge_map_from_coupling(
    pi: np.ndarray, mu0: EmpiricalMeasure, mu1: EmpiricalMeasure
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Barycentric projection of a coupling to a deterministic map on source atoms.
    """
    row_mass = pi.sum(axis=1)
    barycenters = mu0.samples.copy()
    positive_rows = row_mass > _TOL
    if np.any(positive_rows):
        barycenters[positive_rows] = pi[positive_rows] @ mu1.samples
        barycenters[positive_rows] /= row_mass[positive_rows, np.newaxis]

    def monge_map(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        closest_idx = int(np.argmin(np.linalg.norm(mu0.samples - x[np.newaxis, :], axis=1)))
        return barycenters[closest_idx].copy()

    return monge_map


def _preserve_order_unique_rows(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return samples.reshape(0, 0), np.zeros(0, dtype=int)

    unique_rows, first_idx, inverse = np.unique(
        samples, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first_idx)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return unique_rows[order], remap[inverse]


def aggregate_empirical_measure(measure: EmpiricalMeasure) -> EmpiricalMeasure:
    """Aggregate exactly identical support locations by summing their masses."""
    unique_samples, inverse = _preserve_order_unique_rows(measure.samples)
    if unique_samples.size == 0:
        return EmpiricalMeasure(np.zeros((0, measure.d)), np.zeros(0))

    weights = np.bincount(inverse, weights=measure.weights, minlength=len(unique_samples))
    return EmpiricalMeasure(unique_samples, weights)


def align_samples_to_support(
    samples: np.ndarray,
    support: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Match each row of ``samples`` to rows of ``support``.

    The isometric-lift recursion only aggregates exactly coincident support points,
    so this alignment should be exact up to floating-point noise.
    """
    samples = np.asarray(samples, dtype=float)
    support = np.asarray(support, dtype=float)

    if samples.ndim != 2 or support.ndim != 2:
        raise ValueError("samples and support must be two-dimensional arrays")
    if samples.shape[1] != support.shape[1]:
        raise ValueError("samples and support must have the same ambient dimension")
    if samples.shape[0] == 0:
        return np.zeros(0, dtype=int)
    if support.shape[0] == 0:
        raise ValueError("cannot align to an empty support")

    if samples.shape[0] == support.shape[0]:
        distances = cdist(samples, support)
        row_ind, col_ind = linear_sum_assignment(distances)
        matched_distances = distances[row_ind, col_ind]
        if np.any(matched_distances > tol):
            raise ValueError(
                "Failed to align empirical supports during the characteristic lift; "
                "this indicates the local HK recursion drifted off the expected support."
            )
        indices = np.full(samples.shape[0], -1, dtype=int)
        indices[row_ind] = col_ind
        return indices

    tree = cKDTree(support)
    distances, indices = tree.query(samples, k=1)
    if np.any(distances > tol):
        raise ValueError(
            "Failed to align empirical supports during the characteristic lift; "
            "this indicates the local HK recursion drifted off the expected support."
        )
    return np.asarray(indices, dtype=int)


def compress_empirical_measure_weighted_kmeans(
    measure: EmpiricalMeasure,
    max_atoms: int,
    max_iterations: int = 20,
    random_state: int = 0,
) -> EmpiricalMeasure:
    """
    Compress an empirical measure to at most `max_atoms` support points.

    The compressed support uses weighted k-means centroids and preserves total mass
    by summing cluster masses.
    """
    measure = aggregate_empirical_measure(measure)
    n_atoms = measure.n_samples
    if max_atoms <= 0:
        raise ValueError("max_atoms must be positive")
    if n_atoms <= max_atoms:
        return measure

    samples = measure.samples
    weights = measure.weights
    k = min(max_atoms, n_atoms)
    rng = np.random.default_rng(random_state)

    probabilities = weights / weights.sum()
    init_idx = rng.choice(n_atoms, size=k, replace=False, p=probabilities)
    centers = samples[init_idx].copy()

    for _ in range(max_iterations):
        tree = cKDTree(centers)
        distances, labels = tree.query(samples, k=1)

        cluster_mass = np.bincount(labels, weights=weights, minlength=k)
        new_centers = centers.copy()
        for dim in range(measure.d):
            numerator = np.bincount(
                labels,
                weights=weights * samples[:, dim],
                minlength=k,
            )
            positive = cluster_mass > _TOL
            new_centers[positive, dim] = numerator[positive] / cluster_mass[positive]

        empty = np.where(cluster_mass <= _TOL)[0]
        if len(empty) > 0:
            score = weights * distances**2
            refill_idx = np.argsort(score)[-len(empty):]
            new_centers[empty] = samples[refill_idx]

        if np.allclose(new_centers, centers, atol=1e-10, rtol=0.0):
            centers = new_centers
            break
        centers = new_centers

    tree = cKDTree(centers)
    _, labels = tree.query(samples, k=1)
    cluster_mass = np.bincount(labels, weights=weights, minlength=k)

    compressed_samples = np.zeros((k, measure.d), dtype=float)
    for dim in range(measure.d):
        numerator = np.bincount(
            labels,
            weights=weights * samples[:, dim],
            minlength=k,
        )
        positive = cluster_mass > _TOL
        compressed_samples[positive, dim] = numerator[positive] / cluster_mass[positive]

    positive = cluster_mass > _TOL
    return EmpiricalMeasure(compressed_samples[positive], cluster_mass[positive])


def _aggregate_pushforward_cone(
    mapped_samples: np.ndarray,
    reference_weights: np.ndarray,
    updated_radii: np.ndarray,
    radial_aggregation_mode: str = "mass_preserving",
) -> ConeMeasure:
    """
    Aggregate an exact pushforward when multiple source atoms share a destination.

    The new reference mass is the sum of pushed reference masses, and the new
    deterministic radius is chosen from the incoming transported radii according
    to ``radial_aggregation_mode``.
    """
    mapped_samples = np.asarray(mapped_samples, dtype=float)
    reference_weights = np.asarray(reference_weights, dtype=float)
    updated_radii = np.asarray(updated_radii, dtype=float)

    if mapped_samples.shape[0] == 0:
        dim = mapped_samples.shape[1] if mapped_samples.ndim == 2 else 0
        return ConeMeasure(np.zeros((0, dim)), np.zeros(0), np.zeros(0))

    if radial_aggregation_mode not in {"mass_preserving", "mean_radius"}:
        raise ValueError(
            "radial_aggregation_mode must be 'mass_preserving' or 'mean_radius'"
        )

    unique_samples, inverse = _preserve_order_unique_rows(mapped_samples)
    reference_mass = np.bincount(
        inverse, weights=reference_weights, minlength=len(unique_samples)
    )

    new_radii = np.zeros(len(unique_samples), dtype=float)
    positive = reference_mass > _TOL
    if radial_aggregation_mode == "mass_preserving":
        projected_mass = np.bincount(
            inverse,
            weights=reference_weights * updated_radii**2,
            minlength=len(unique_samples),
        )
        new_radii[positive] = np.sqrt(projected_mass[positive] / reference_mass[positive])
    else:
        mean_radius = np.bincount(
            inverse,
            weights=reference_weights * updated_radii,
            minlength=len(unique_samples),
        )
        new_radii[positive] = mean_radius[positive] / reference_mass[positive]
    return ConeMeasure(unique_samples, new_radii, reference_mass)


def _pushforward_cone_through_local_plan(
    current_lambda: ConeMeasure,
    mu_source: EmpiricalMeasure,
    mu_target: EmpiricalMeasure,
    pi: np.ndarray,
    radial_aggregation_mode: str = "mass_preserving",
) -> ConeMeasure:
    """
    Push the current lift forward through a local LET coupling and aggregate onto
    the exact target base support.

    Each target base atom receives one cone atom. Incoming reference masses and
    transported radii are aggregated according to the local coupling weights.
    """
    pi = np.asarray(pi, dtype=float)
    if current_lambda.n_samples != mu_source.n_samples:
        raise ValueError("current_lambda and mu_source must have the same number of atoms")
    if pi.shape != (mu_source.n_samples, mu_target.n_samples):
        raise ValueError("pi must have shape (mu_source.n_samples, mu_target.n_samples)")
    if radial_aggregation_mode not in {"mass_preserving", "mean_radius"}:
        raise ValueError(
            "radial_aggregation_mode must be 'mass_preserving' or 'mean_radius'"
        )

    row_mass = pi.sum(axis=1)
    col_mass = pi.sum(axis=0)
    positive_rows = row_mass > _TOL
    positive_cols = col_mass > _TOL

    u_source = np.zeros(mu_source.n_samples, dtype=float)
    u_source[positive_rows] = mu_source.weights[positive_rows] / row_mass[positive_rows]
    u_target = np.zeros(mu_target.n_samples, dtype=float)
    u_target[positive_cols] = mu_target.weights[positive_cols] / col_mass[positive_cols]

    next_reference_weights = np.zeros(mu_target.n_samples, dtype=float)
    radii_numer = np.zeros(mu_target.n_samples, dtype=float)
    projected_mass = np.zeros(mu_target.n_samples, dtype=float)

    support = np.argwhere(pi > _TOL)
    for source_idx, target_idx in support:
        edge_reference_mass = (
            current_lambda.weights[source_idx] * pi[source_idx, target_idx] / row_mass[source_idx]
        )
        q_edge = np.sqrt(max(u_target[target_idx] / u_source[source_idx], 0.0))
        transported_radius = current_lambda.radii[source_idx] * q_edge

        next_reference_weights[target_idx] += edge_reference_mass
        radii_numer[target_idx] += edge_reference_mass * transported_radius
        projected_mass[target_idx] += edge_reference_mass * transported_radius**2

    next_radii = np.zeros(mu_target.n_samples, dtype=float)
    positive_targets = next_reference_weights > _TOL
    if radial_aggregation_mode == "mass_preserving":
        next_radii[positive_targets] = np.sqrt(
            projected_mass[positive_targets] / next_reference_weights[positive_targets]
        )
    else:
        next_radii[positive_targets] = (
            radii_numer[positive_targets] / next_reference_weights[positive_targets]
        )

    missing_targets = (mu_target.weights > _TOL) & ~positive_targets
    if np.any(missing_targets):
        warnings.warn(
            "Local LET plan assigned zero incoming coupling mass to target atoms with "
            "positive base mass during isometric_lift; corresponding lifted atoms were "
            "left at zero weight/radius.",
            RuntimeWarning,
        )

    return ConeMeasure(
        mu_target.samples.copy(),
        next_radii,
        next_reference_weights,
    )


def build_optimal_lifted_coupling(
    mu0: EmpiricalMeasure, mu1: EmpiricalMeasure, pi: np.ndarray
) -> LiftedCoupling:
    pi0 = pi.sum(axis=1)
    pi1 = pi.sum(axis=0)

    source_radius_by_atom = np.zeros(mu0.n_samples, dtype=float)
    source_matched = pi0 > _TOL
    source_radius_by_atom[source_matched] = np.sqrt(mu0.weights[source_matched] / pi0[source_matched])

    target_radius_by_atom = np.zeros(mu1.n_samples, dtype=float)
    target_matched = pi1 > _TOL
    target_radius_by_atom[target_matched] = np.sqrt(mu1.weights[target_matched] / pi1[target_matched])

    source_positions = []
    target_positions = []
    source_radii = []
    target_radii = []
    weights = []
    source_indices = []
    target_indices = []

    for i, j in np.argwhere(pi > _TOL):
        source_positions.append(mu0.samples[i])
        target_positions.append(mu1.samples[j])
        source_radii.append(source_radius_by_atom[i])
        target_radii.append(target_radius_by_atom[j])
        weights.append(pi[i, j])
        source_indices.append(i)
        target_indices.append(j)

    apex = np.zeros(mu0.d, dtype=float)

    for i in np.where((pi0 <= _TOL) & (mu0.weights > _TOL))[0]:
        source_positions.append(mu0.samples[i])
        target_positions.append(apex)
        source_radii.append(1.0)
        target_radii.append(0.0)
        weights.append(mu0.weights[i])
        source_indices.append(i)
        target_indices.append(-1)

    for j in np.where((pi1 <= _TOL) & (mu1.weights > _TOL))[0]:
        source_positions.append(apex)
        target_positions.append(mu1.samples[j])
        source_radii.append(0.0)
        target_radii.append(1.0)
        weights.append(mu1.weights[j])
        source_indices.append(-1)
        target_indices.append(j)

    if not weights:
        empty_positions = np.zeros((0, mu0.d), dtype=float)
        empty_scalars = np.zeros(0, dtype=float)
        empty_indices = np.zeros(0, dtype=int)
        return LiftedCoupling(
            source_positions=empty_positions,
            target_positions=empty_positions.copy(),
            source_radii=empty_scalars,
            target_radii=empty_scalars.copy(),
            weights=empty_scalars.copy(),
            source_indices=empty_indices,
            target_indices=empty_indices.copy(),
        )

    return LiftedCoupling(
        source_positions=np.asarray(source_positions, dtype=float),
        target_positions=np.asarray(target_positions, dtype=float),
        source_radii=np.asarray(source_radii, dtype=float),
        target_radii=np.asarray(target_radii, dtype=float),
        weights=np.asarray(weights, dtype=float),
        source_indices=np.asarray(source_indices, dtype=int),
        target_indices=np.asarray(target_indices, dtype=int),
    )


def let_lift(
    mu0: EmpiricalMeasure,
    mu1: EmpiricalMeasure,
    N: int,
    let_solver: str = "pot_mm",
    entropy_reg: float = 0.01,
    let_max_iterations: int = 500,
    let_stop_threshold: float = 1e-8,
    let_reg_m: float = 1.0,
) -> Tuple[list, list]:
    """
    Build the discrete cone path induced directly by the endpoint LET lift.
    """
    if N <= 0:
        raise ValueError("N must be positive")

    pi, _ = solve_let_unbalanced_transport(
        mu0,
        mu1,
        entropy_reg=entropy_reg,
        max_iterations=let_max_iterations,
        method=let_solver,
        stop_threshold=let_stop_threshold,
        reg_m=let_reg_m,
    )
    coupling = build_optimal_lifted_coupling(mu0, mu1, pi)

    lambda_list = []
    radii_list = []
    for s in np.linspace(0.0, 1.0, N + 1):
        cone_measure = interpolate_lifted_measure(coupling, float(s))
        lambda_list.append(cone_measure)
        radii_list.append(cone_measure.radii.copy())
    # obtain total masses and check that they are the same
    total_masses = [np.sum(cone_measure.weights) for cone_measure in lambda_list]
    if not np.allclose(total_masses, total_masses[0], atol=_TOL):
        raise RuntimeError("Inconsistent total masses along the lifted path")

    return lambda_list, radii_list


def _compute_local_hk_step(
    mu_source: EmpiricalMeasure,
    mu_target: EmpiricalMeasure,
    pi: np.ndarray,
    dt: float,
    approximation_mode: str = "barycentric",
) -> LocalHKStep:
    row_mass = pi.sum(axis=1)
    col_mass = pi.sum(axis=0)
    positive_rows = row_mass > _TOL

    u_source = np.zeros(mu_source.n_samples, dtype=float)
    u_source[positive_rows] = mu_source.weights[positive_rows] / row_mass[positive_rows]

    u_target = np.zeros(mu_target.n_samples, dtype=float)
    positive_cols = col_mass > _TOL
    u_target[positive_cols] = mu_target.weights[positive_cols] / col_mass[positive_cols]
    v = np.zeros_like(mu_source.samples)
    beta = np.zeros(mu_source.n_samples, dtype=float)
    beta[~positive_rows] = -0.5 / dt

    positive_source_indices = np.where(positive_rows)[0]
    if len(positive_source_indices) > 0:
        if approximation_mode == "barycentric":
            conditional = pi[positive_rows] / row_mass[positive_rows, np.newaxis]
        elif approximation_mode == "argmax":
            conditional = np.zeros_like(pi[positive_rows])
            argmax_targets = np.argmax(pi[positive_rows], axis=1)
            conditional[np.arange(len(argmax_targets)), argmax_targets] = 1.0
        else:
            raise ValueError(
                "approximation_mode must be 'barycentric' or 'argmax'"
            )

        source_samples = mu_source.samples[positive_rows]
        source_density = u_source[positive_rows]

        displacement_edge = mu_target.samples[np.newaxis, :, :] - source_samples[:, np.newaxis, :]
        distance_edge = np.linalg.norm(displacement_edge, axis=2)

        q_edge = np.zeros_like(distance_edge)
        valid_edge_cols = positive_cols[np.newaxis, :]
        q_edge[:, positive_cols] = np.sqrt(
            np.maximum(
                u_target[positive_cols][np.newaxis, :] / source_density[:, np.newaxis],
                0.0,
            )
        )

        edge_velocity = np.zeros_like(displacement_edge)
        moving_edges = distance_edge > _TOL
        if np.any(moving_edges):
            edge_speed = np.zeros_like(distance_edge)
            edge_speed[moving_edges] = (
                q_edge[moving_edges] * np.sin(distance_edge[moving_edges])
            ) / (dt * distance_edge[moving_edges])
            edge_velocity[moving_edges] = (
                edge_speed[moving_edges][:, np.newaxis] * displacement_edge[moving_edges]
            )

        edge_beta = 0.5 / dt * (q_edge * np.cos(distance_edge) - 1.0)

        v[positive_rows] = np.einsum("ij,ijd->id", conditional, edge_velocity)
        beta[positive_rows] = np.sum(conditional * edge_beta, axis=1)

    speeds = np.linalg.norm(v, axis=1)
    a_t = dt * speeds
    b_t = 1.0 + 2.0 * dt * beta
    q = np.sqrt(a_t**2 + b_t**2)

    barycenters = mu_source.samples.copy()
    moving = speeds > _TOL
    if np.any(moving):
        directions = np.zeros_like(v)
        directions[moving] = v[moving] / speeds[moving, np.newaxis]
        phi_t = np.arctan2(a_t[moving], b_t[moving])
        barycenters[moving] += directions[moving] * phi_t[:, np.newaxis]

    return LocalHKStep(
        map_positions=barycenters,
        q=q,
        v=v,
        beta=beta,
        row_mass=row_mass,
        col_mass=col_mass,
        coupling=np.asarray(pi, dtype=float),
    )


def _compute_exact_hk_log_step(
    mu_source: EmpiricalMeasure,
    mu_target: EmpiricalMeasure,
    pi: np.ndarray,
    dt: float,
) -> LocalHKStep:
    """
    Exact discrete HK log step under the paper's Monge-map assumptions.

    This requires the LET coupling to be supported on a map and to have no
    unmatched source or target mass, mirroring the assumptions in the explicit
    HK log/exp formulas from ``main.tex``.
    """
    row_mass = pi.sum(axis=1)
    col_mass = pi.sum(axis=0)
    positive_rows = row_mass > _TOL
    positive_cols = col_mass > _TOL

    source_has_mass = mu_source.weights > _TOL
    target_has_mass = mu_target.weights > _TOL
    if not np.all(positive_rows[source_has_mass]):
        raise ValueError(
            "Explicit HK logarithmic map requires mu_source << pi_0; found "
            "source atoms with positive mass but zero transported mass."
        )
    if not np.all(positive_cols[target_has_mass]):
        raise ValueError(
            "Explicit HK logarithmic map requires mu_target << pi_1; found "
            "target atoms with positive mass but zero incoming transported mass."
        )

    row_counts = np.count_nonzero(pi > _TOL, axis=1)
    if np.any(row_counts[source_has_mass] > 1):
        raise ValueError(
            "Explicit HK logarithmic map requires the LET coupling to be supported "
            "on a map. The current optimal coupling splits source mass across "
            "multiple target atoms. Use allow_approximation=True to recover the "
            "previous barycentric approximation."
        )

    target_indices = np.full(mu_source.n_samples, -1, dtype=int)
    if np.any(positive_rows):
        target_indices[positive_rows] = np.argmax(pi[positive_rows], axis=1)

    map_positions = mu_source.samples.copy()
    map_positions[positive_rows] = mu_target.samples[target_indices[positive_rows]]

    u_source = np.zeros(mu_source.n_samples, dtype=float)
    u_source[positive_rows] = mu_source.weights[positive_rows] / row_mass[positive_rows]

    u_target = np.zeros(mu_target.n_samples, dtype=float)
    u_target[positive_cols] = mu_target.weights[positive_cols] / col_mass[positive_cols]

    q = np.zeros(mu_source.n_samples, dtype=float)
    valid_q = positive_rows & (u_source > _TOL)
    q[valid_q] = np.sqrt(u_target[target_indices[valid_q]] / u_source[valid_q])

    displacement = map_positions - mu_source.samples
    distance = np.linalg.norm(displacement, axis=1)

    v = np.zeros_like(mu_source.samples)
    moving = distance > _TOL
    if np.any(moving):
        speed_factor = (q[moving] * np.sin(distance[moving])) / (dt * distance[moving])
        v[moving] = speed_factor[:, np.newaxis] * displacement[moving]

    beta = 0.5 / dt * (q * np.cos(distance) - 1.0)
    return LocalHKStep(
        map_positions=map_positions,
        q=q,
        v=v,
        beta=beta,
        row_mass=row_mass,
        col_mass=col_mass,
        coupling=np.asarray(pi, dtype=float),
    )


def hk_logarithmic_map(
    mu_source: EmpiricalMeasure,
    mu_target: EmpiricalMeasure,
    let_solver: str = "pot_mm",
    entropy_reg: float = 0.01,
    let_max_iterations: int = 500,
    let_stop_threshold: float = 1e-8,
    let_reg_m: float = 1.0,
    dt: float = 1.0,
    allow_approximation: bool = False,
    approximation_mode: str = "barycentric",
    return_step: bool = False,
):
    """
    Empirical HK logarithmic map from ``mu_source`` to ``mu_target``.

    This exposes the explicit source-supported HK tangent from ``main.tex`` when
    the LET coupling is supported on a map and there is no unmatched mass. If
    those assumptions fail, the function raises by default because the explicit
    HK log/exp identities need not hold. Set ``allow_approximation=True`` to use
    the empirical approximation employed internally by the isometric-lift and
    parallel-transport routines. ``approximation_mode='barycentric'`` averages
    edgewise HK tangent contributions row-wise, while
    ``approximation_mode='argmax'`` first projects each row to its largest-mass
    target.

    The returned tangent ``(v, beta)`` is supported on the atoms of
    ``mu_source``. The optional ``dt`` parameter rescales the tangent so that it
    generates the local step over a time horizon of length ``dt``.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    pi, _ = solve_let_unbalanced_transport(
        mu_source,
        mu_target,
        entropy_reg=entropy_reg,
        max_iterations=let_max_iterations,
        method=let_solver,
        stop_threshold=let_stop_threshold,
        reg_m=let_reg_m,
    )
    try:
        local_step = _compute_exact_hk_log_step(mu_source, mu_target, pi, dt)
    except ValueError:
        if not allow_approximation:
            raise
        local_step = _compute_local_hk_step(
            mu_source,
            mu_target,
            pi,
            dt,
            approximation_mode=approximation_mode,
        )
    tangent = (local_step.v, local_step.beta)
    if return_step:
        return tangent, local_step
    return tangent


def hk_exponential_map(
    mu_source: EmpiricalMeasure,
    tangent_hk: Tuple[np.ndarray, np.ndarray],
    t: float = 1.0,
    aggregate: bool = True,
) -> EmpiricalMeasure:
    """
    Push an empirical HK tangent forward via the explicit closed-form HK exp map.

    The input tangent is assumed to use the same normalization as the rest of
    this module, namely the one induced by the continuity-reaction equation
    ``partial_t mu + div(mu v) = 4 beta mu`` and the cone lifting
    ``(v, beta) -> (v, 2 beta r)``.
    """
    if t < 0:
        raise ValueError("t must be non-negative")

    v, beta = _validate_hk_tangent(tangent_hk, mu_source)
    speeds = np.linalg.norm(v, axis=1)

    a_t = t * speeds
    b_t = 1.0 + 2.0 * t * beta
    q_t = np.sqrt(a_t**2 + b_t**2)
    phi_t = np.arctan2(a_t, b_t)

    transported_samples = mu_source.samples.copy()
    moving = speeds > _TOL
    if np.any(moving):
        directions = np.zeros_like(v)
        directions[moving] = v[moving] / speeds[moving, np.newaxis]
        transported_samples[moving] += directions[moving] * phi_t[moving, np.newaxis]

    transported_weights = mu_source.weights * q_t**2
    positive = transported_weights > _TOL
    image = EmpiricalMeasure(
        transported_samples[positive],
        transported_weights[positive],
    )
    if aggregate:
        return aggregate_empirical_measure(image)
    return image


def isometric_lift(
    mu0: EmpiricalMeasure,
    mu1: EmpiricalMeasure,
    N: int,
    let_solver: str = "pot_mm",
    entropy_reg: float = 0.01,
    let_max_iterations: int = 500,
    let_stop_threshold: float = 1e-8,
    let_reg_m: float = 1.0,
    approximation_mode: str = "barycentric",
    radial_aggregation_mode: str = "mass_preserving",
    compression_max_atoms: Optional[int] = None,
    compression_kmeans_iterations: int = 20,
    return_tangents: bool = False,
    return_steps: bool = False,
) -> Tuple[list, list]:
    """
    Approximate Algorithm 1 from the paper using empirical local LET solves.

    The endpoint LET lift is used only to produce target base measures at the
    sampled times. The lifted path itself is then rebuilt recursively using the
    local LET plans and radial updates from the isometric-lift algorithm. In the
    finite-sample setting, each next lifted measure is supported exactly on the
    next base support, and incoming radial contributions are aggregated there
    according to the local coupling weights.
    When multiple source atoms land on the same target point, the finite-sample
    radial merge is controlled by ``radial_aggregation_mode``:
    ``'mass_preserving'`` preserves projected base mass, while
    ``'mean_radius'`` averages the incoming transported radii directly.
    """
    if N <= 0:
        raise ValueError("N must be positive")

    dt = 1.0 / N
    endpoint_lambdas, _ = let_lift(
        mu0,
        mu1,
        N,
        let_solver=let_solver,
        entropy_reg=entropy_reg,
        let_max_iterations=let_max_iterations,
        let_stop_threshold=let_stop_threshold,
        let_reg_m=let_reg_m,
    )
    if compression_max_atoms is None:
        compression_max_atoms = max(mu0.n_samples, mu1.n_samples)

    target_base_path = []
    for cone_measure in endpoint_lambdas:
        projected = aggregate_empirical_measure(project_cone_measure(cone_measure))
        projected = compress_empirical_measure_weighted_kmeans(
            projected,
            max_atoms=compression_max_atoms,
            max_iterations=compression_kmeans_iterations,
        )
        target_base_path.append(projected)

    current_base = aggregate_empirical_measure(mu0)
    current_lambda = ConeMeasure(
        current_base.samples.copy(),
        np.ones(current_base.n_samples, dtype=float),
        current_base.weights.copy(),
    )

    lambda_list = [current_lambda]
    radii_list = [current_lambda.radii.copy()]
    lifted_tangents = []
    local_steps = []
    for i in range(N):
        target_base = target_base_path[i + 1]
        pi, _ = solve_let_unbalanced_transport(
            current_base,
            target_base,
            entropy_reg=entropy_reg,
            max_iterations=let_max_iterations,
            method=let_solver,
            stop_threshold=let_stop_threshold,
            reg_m=let_reg_m,
        )

        local_step = _compute_local_hk_step(
            current_base,
            target_base,
            pi,
            dt,
            approximation_mode=approximation_mode,
        )
        local_steps.append(local_step)
        lifted_tangents.append(
            lift_tangent((local_step.v, local_step.beta), current_base, current_lambda)
        )

        current_lambda = _pushforward_cone_through_local_plan(
            current_lambda,
            current_base,
            target_base,
            pi,
            radial_aggregation_mode=radial_aggregation_mode,
        )
        current_base = target_base

        lambda_list.append(current_lambda)
        radii_list.append(current_lambda.radii.copy())

    if return_tangents and return_steps:
        return lambda_list, radii_list, lifted_tangents, local_steps
    if return_tangents:
        return lambda_list, radii_list, lifted_tangents
    if return_steps:
        return lambda_list, radii_list, local_steps
    return lambda_list, radii_list


# ============================================================================
# Tangent lifting, projection, and transport
# ============================================================================


def lift_tangent(
    tangent_hk: Tuple[np.ndarray, np.ndarray],
    measure: EmpiricalMeasure,
    cone_measure: ConeMeasure,
    atom_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Lift an HK tangent (v, beta) to the cone via (v(x), 2 beta(x) r).
    """
    v, beta = tangent_hk
    v = np.asarray(v, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if v.shape != (measure.n_samples, measure.d):
        raise ValueError("v must have shape (n_samples, d)")
    if beta.shape != (measure.n_samples,):
        raise ValueError("beta must have shape (n_samples,)")

    if atom_indices is None:
        if cone_measure.n_samples != measure.n_samples:
            raise ValueError("cone_measure must align with measure when atom_indices is omitted")
        radial_component = 2.0 * cone_measure.radii * beta
        return np.hstack([v, radial_component[:, np.newaxis]])

    atom_indices = np.asarray(atom_indices, dtype=int)
    if atom_indices.shape != (cone_measure.n_samples,):
        raise ValueError("atom_indices must have one entry per cone atom")

    spatial = np.zeros((cone_measure.n_samples, measure.d), dtype=float)
    radial = np.zeros(cone_measure.n_samples, dtype=float)
    valid = atom_indices >= 0
    spatial[valid] = v[atom_indices[valid]]
    radial[valid] = 2.0 * cone_measure.radii[valid] * beta[atom_indices[valid]]
    return np.hstack([spatial, radial[:, np.newaxis]])


def project_tangent(
    cone_tangent: np.ndarray,
    cone_measure: ConeMeasure,
    atom_indices: Optional[np.ndarray] = None,
    n_base_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project a cone tangent back to the HK tangent space.

    Without `atom_indices`, this applies the deterministic-radial-law formula
    pointwise. With `atom_indices`, it applies the full discrete projection
    formula grouped by base atoms.
    """
    cone_tangent = np.asarray(cone_tangent, dtype=float)
    if cone_tangent.shape != (cone_measure.n_samples, cone_measure.d + 1):
        raise ValueError("cone_tangent must have shape (n_cone_atoms, d + 1)")

    spatial = cone_tangent[:, :-1]
    radial = cone_tangent[:, -1]

    if atom_indices is None:
        v = spatial.copy()
        beta = np.zeros(cone_measure.n_samples, dtype=float)
        positive = cone_measure.radii > _TOL
        beta[positive] = radial[positive] / (2.0 * cone_measure.radii[positive])
        return v, beta

    if n_base_points is None:
        raise ValueError("n_base_points is required when atom_indices are provided")

    atom_indices = np.asarray(atom_indices, dtype=int)
    if atom_indices.shape != (cone_measure.n_samples,):
        raise ValueError("atom_indices must have one entry per cone atom")

    valid = atom_indices >= 0
    weights_r2 = cone_measure.weights[valid] * cone_measure.radii[valid] ** 2
    denom = np.bincount(atom_indices[valid], weights=weights_r2, minlength=n_base_points)

    v = np.zeros((n_base_points, cone_measure.d), dtype=float)
    for dim in range(cone_measure.d):
        numer = np.bincount(
            atom_indices[valid],
            weights=weights_r2 * spatial[valid, dim],
            minlength=n_base_points,
        )
        nonzero = denom > _TOL
        v[nonzero, dim] = numer[nonzero] / denom[nonzero]

    beta = np.zeros(n_base_points, dtype=float)
    numer_beta = np.bincount(
        atom_indices[valid],
        weights=cone_measure.weights[valid] * cone_measure.radii[valid] * radial[valid],
        minlength=n_base_points,
    )
    nonzero = denom > _TOL
    beta[nonzero] = numer_beta[nonzero] / (2.0 * denom[nonzero])
    return v, beta


def _cone_cost_matrix(lambda0: ConeMeasure, lambda1: ConeMeasure) -> np.ndarray:
    """
    Squared cone-distance cost matrix between two cone measures.
    """
    if lambda0.d != lambda1.d:
        raise ValueError("Cone measures must have the same ambient dimension")

    base_dist = cdist(lambda0.samples, lambda1.samples, metric="euclidean")
    theta = np.minimum(base_dist, np.pi)
    return (
        lambda0.radii[:, np.newaxis] ** 2
        + lambda1.radii[np.newaxis, :] ** 2
        - 2.0
        * lambda0.radii[:, np.newaxis]
        * lambda1.radii[np.newaxis, :]
        * np.cos(theta)
    )


def solve_balanced_cone_transport(
    lambda0: ConeMeasure,
    lambda1: ConeMeasure,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the balanced Wasserstein transport problem directly on the cone.
    """
    if ot is None:
        raise ImportError("POT is required for balanced cone transport solves")

    mass0 = lambda0.total_mass()
    mass1 = lambda1.total_mass()
    if not np.isclose(mass0, mass1, atol=1e-10, rtol=1e-10):
        raise ValueError("Balanced cone transport requires equal total mass")
    if mass0 <= _TOL:
        empty = np.zeros((lambda0.n_samples, lambda1.n_samples), dtype=float)
        return empty, empty

    cost_matrix = _cone_cost_matrix(lambda0, lambda1)
    coupling = ot.emd(lambda0.weights, lambda1.weights, cost_matrix)
    return np.asarray(coupling, dtype=float), cost_matrix


def _deterministic_targets_from_cone_coupling(
    coupling: np.ndarray,
    lambda_source: ConeMeasure,
    lambda_target: ConeMeasure,
    approximation_mode: str = "argmax",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a cone coupling into a deterministic target assignment on source atoms.
    """
    coupling = np.asarray(coupling, dtype=float)
    if coupling.shape != (lambda_source.n_samples, lambda_target.n_samples):
        raise ValueError("coupling shape does not match the provided cone measures")

    row_mass = coupling.sum(axis=1)
    target_samples = lambda_source.samples.copy()
    target_radii = lambda_source.radii.copy()
    positive_rows = row_mass > _TOL

    if not np.any(positive_rows):
        return target_samples, target_radii

    if approximation_mode == "argmax":
        target_indices = np.argmax(coupling[positive_rows], axis=1)
        target_samples[positive_rows] = lambda_target.samples[target_indices]
        target_radii[positive_rows] = lambda_target.radii[target_indices]
        return target_samples, target_radii

    if approximation_mode == "barycentric":
        conditional = np.zeros_like(coupling[positive_rows])
        conditional = coupling[positive_rows] / row_mass[positive_rows, np.newaxis]
        target_samples[positive_rows] = conditional @ lambda_target.samples
        target_radii[positive_rows] = conditional @ lambda_target.radii
        return target_samples, target_radii

    raise ValueError("approximation_mode must be 'argmax' or 'barycentric'")


def _cone_logarithmic_map_point(
    z0: Tuple[np.ndarray, float],
    z1: Tuple[np.ndarray, float],
) -> Tuple[np.ndarray, float]:
    """
    Exact cone tangent sending ``z0`` to ``z1`` under the pointwise cone
    exponential map at unit time.
    """
    x0, r0 = z0
    x1, r1 = z1
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    r0 = float(r0)
    r1 = float(r1)

    if r0 <= _TOL:
        raise ValueError("cone logarithmic map is undefined here for zero source radius")

    displacement = x1 - x0
    theta = min(np.linalg.norm(displacement), np.pi)
    if theta <= _TOL:
        spatial = np.zeros_like(x0)
    else:
        direction = displacement / theta
        spatial = direction * ((r1 / r0) * np.sin(theta))
    radial = r1 * np.cos(theta) - r0
    return spatial, float(radial)


def _aggregate_cone_tangent_under_deterministic_map(
    cone_tangent: np.ndarray,
    source_weights: np.ndarray,
    atom_indices: np.ndarray,
    n_target_atoms: int,
) -> np.ndarray:
    """
    Aggregate cone tangent vectors onto target atoms using the deterministic
    pushforward coupling induced by a pointwise cone map.
    """
    cone_tangent = np.asarray(cone_tangent, dtype=float)
    source_weights = np.asarray(source_weights, dtype=float)
    atom_indices = np.asarray(atom_indices, dtype=int)

    if cone_tangent.ndim != 2:
        raise ValueError("cone_tangent must be a two-dimensional array")
    if source_weights.shape != (cone_tangent.shape[0],):
        raise ValueError("source_weights must have one entry per source cone atom")
    if atom_indices.shape != (cone_tangent.shape[0],):
        raise ValueError("atom_indices must have one entry per source cone atom")
    if n_target_atoms < 0:
        raise ValueError("n_target_atoms must be non-negative")

    aggregated = np.zeros((n_target_atoms, cone_tangent.shape[1]), dtype=float)
    valid = atom_indices >= 0
    if not np.any(valid):
        return aggregated

    target_weights = np.bincount(
        atom_indices[valid],
        weights=source_weights[valid],
        minlength=n_target_atoms,
    )
    positive = target_weights > _TOL
    for dim in range(cone_tangent.shape[1]):
        numer = np.bincount(
            atom_indices[valid],
            weights=source_weights[valid] * cone_tangent[valid, dim],
            minlength=n_target_atoms,
        )
        aggregated[positive, dim] = numer[positive] / target_weights[positive]
    return aggregated


def _aggregate_cone_tangent_under_plan_coupling(
    cone_tangent: np.ndarray,
    current_lambda: ConeMeasure,
    next_lambda: ConeMeasure,
    coupling: np.ndarray,
) -> np.ndarray:
    """
    Aggregate cone tangent vectors onto the next lifted measure using the local
    LET coupling retained during the isometric lift.

    The source cone mass is split across target atoms in proportion to the
    coupling row, parallel transported to the target cone atoms of
    ``next_lambda``, and then averaged using those split reference masses.
    """
    cone_tangent = np.asarray(cone_tangent, dtype=float)
    coupling = np.asarray(coupling, dtype=float)

    if cone_tangent.shape != (current_lambda.n_samples, current_lambda.d + 1):
        raise ValueError("cone_tangent must match current_lambda")
    if coupling.shape != (current_lambda.n_samples, next_lambda.n_samples):
        raise ValueError("coupling shape must match current and next lifted measures")

    row_mass = coupling.sum(axis=1)
    valid_rows = row_mass > _TOL

    aggregated = np.zeros((next_lambda.n_samples, current_lambda.d + 1), dtype=float)
    target_reference_mass = np.zeros(next_lambda.n_samples, dtype=float)

    support = np.argwhere(coupling > _TOL)
    for source_idx, target_idx in support:
        if row_mass[source_idx] <= _TOL:
            continue

        edge_reference_mass = (
            current_lambda.weights[source_idx]
            * coupling[source_idx, target_idx]
            / row_mass[source_idx]
        )
        a1, b1 = cone_parallel_transport_explicit(
            cone_tangent[source_idx, :-1],
            cone_tangent[source_idx, -1],
            (current_lambda.samples[source_idx], current_lambda.radii[source_idx]),
            (next_lambda.samples[target_idx], next_lambda.radii[target_idx]),
        )
        aggregated[target_idx, :-1] += edge_reference_mass * a1
        aggregated[target_idx, -1] += edge_reference_mass * b1
        target_reference_mass[target_idx] += edge_reference_mass

    positive_targets = target_reference_mass > _TOL
    aggregated[positive_targets] /= target_reference_mass[positive_targets, np.newaxis]
    return aggregated


def _resolve_cone_atom_map(
    mapped_lambda: ConeMeasure,
    target_lambda: ConeMeasure,
    tol: float,
) -> np.ndarray:
    """
    Match a deterministic cone pushforward to the target cone atoms.

    First try the identity ordering, which is the natural case for direct
    deterministic cone geodesics. Fall back to nearest-neighbor matching on the
    augmented cone coordinates when the atoms are merely permuted.
    """
    if mapped_lambda.n_samples != target_lambda.n_samples:
        raise ValueError("Cone atom matching requires the same number of atoms")

    sample_error = np.linalg.norm(mapped_lambda.samples - target_lambda.samples, axis=1)
    radius_error = np.abs(mapped_lambda.radii - target_lambda.radii)
    if np.max(np.maximum(sample_error, radius_error)) <= tol:
        return np.arange(target_lambda.n_samples, dtype=int)

    mapped_cone = np.hstack([mapped_lambda.samples, mapped_lambda.radii[:, np.newaxis]])
    target_cone = np.hstack([target_lambda.samples, target_lambda.radii[:, np.newaxis]])
    tree = cKDTree(target_cone)
    distances, indices = tree.query(mapped_cone, k=1)
    if np.any(distances > tol):
        raise ValueError(
            "Failed to align deterministic cone pushforward with the target cone support."
        )
    return np.asarray(indices, dtype=int)


def _cone_exponential_map_step(
    cone_measure: ConeMeasure,
    cone_tangent: np.ndarray,
    t: float,
) -> ConeMeasure:
    """
    Apply the pointwise cone exponential map to a lifted tangent over time ``t``.
    """
    if t < 0:
        raise ValueError("t must be non-negative")

    cone_tangent = np.asarray(cone_tangent, dtype=float)
    if cone_tangent.shape != (cone_measure.n_samples, cone_measure.d + 1):
        raise ValueError("cone_tangent must have shape (n_cone_atoms, d + 1)")

    positions = cone_measure.samples.copy()
    radii = cone_measure.radii.copy()
    if t == 0 or cone_measure.n_samples == 0:
        return ConeMeasure(positions, radii, cone_measure.weights.copy())

    spatial = cone_tangent[:, :-1]
    radial = cone_tangent[:, -1]
    speeds = np.linalg.norm(spatial, axis=1)

    positive_radii = cone_measure.radii > _TOL
    b_t = np.ones(cone_measure.n_samples, dtype=float)
    b_t[positive_radii] += t * radial[positive_radii] / cone_measure.radii[positive_radii]
    a_t = t * speeds
    q_t = np.sqrt(a_t**2 + b_t**2)
    phi_t = np.arctan2(a_t, b_t)

    moving = speeds > _TOL
    if np.any(moving):
        directions = np.zeros_like(spatial)
        directions[moving] = spatial[moving] / speeds[moving, np.newaxis]
        positions[moving] += directions[moving] * phi_t[moving, np.newaxis]

    radii = cone_measure.radii * q_t
    return ConeMeasure(positions, radii, cone_measure.weights.copy())


def cone_exponential_map(
    cone_measure: ConeMeasure,
    cone_tangent: np.ndarray,
    t: float = 1.0,
    aggregate: bool = False,
) -> ConeMeasure:
    """
    Public cone exponential map for source-supported cone tangents.
    """
    image = _cone_exponential_map_step(cone_measure, cone_tangent, t)
    if aggregate:
        return _aggregate_pushforward_cone(
            image.samples,
            image.weights,
            image.radii,
        )
    return image


def cone_logarithmic_map(
    lambda_source: ConeMeasure,
    lambda_target: ConeMeasure,
    approximation_mode: str = "argmax",
) -> np.ndarray:
    """
    Empirical logarithmic map on the cone, supported on ``lambda_source``.
    """
    coupling, _ = solve_balanced_cone_transport(lambda_source, lambda_target)
    target_samples, target_radii = _deterministic_targets_from_cone_coupling(
        coupling,
        lambda_source,
        lambda_target,
        approximation_mode=approximation_mode,
    )

    cone_tangent = np.zeros((lambda_source.n_samples, lambda_source.d + 1), dtype=float)
    for idx in range(lambda_source.n_samples):
        spatial, radial = _cone_logarithmic_map_point(
            (lambda_source.samples[idx], lambda_source.radii[idx]),
            (target_samples[idx], target_radii[idx]),
        )
        cone_tangent[idx, :-1] = spatial
        cone_tangent[idx, -1] = radial
    return cone_tangent


def cone_wasserstein_geodesic(
    lambda0: ConeMeasure,
    lambda1: ConeMeasure,
    N: int,
    approximation_mode: str = "argmax",
) -> Tuple[list, list]:
    """
    Build a deterministic cone-Wasserstein geodesic approximation and its local
    step tangents directly on the cone.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    dt = 1.0 / N

    coupling, _ = solve_balanced_cone_transport(lambda0, lambda1)
    target_samples, target_radii = _deterministic_targets_from_cone_coupling(
        coupling,
        lambda0,
        lambda1,
        approximation_mode=approximation_mode,
    )

    lambda_list = []
    for s in np.linspace(0.0, 1.0, N + 1):
        positions = np.zeros_like(lambda0.samples)
        radii = np.zeros_like(lambda0.radii)
        for idx in range(lambda0.n_samples):
            positions[idx], radii[idx] = cone_geodesic_step(
                lambda0.samples[idx],
                lambda0.radii[idx],
                target_samples[idx],
                target_radii[idx],
                float(s),
            )
        lambda_list.append(ConeMeasure(positions, radii, lambda0.weights.copy()))

    step_tangents = []
    for k in range(N):
        tangent = np.zeros((lambda0.n_samples, lambda0.d + 1), dtype=float)
        for idx in range(lambda0.n_samples):
            spatial, radial = _cone_logarithmic_map_point(
                (lambda_list[k].samples[idx], lambda_list[k].radii[idx]),
                (lambda_list[k + 1].samples[idx], lambda_list[k + 1].radii[idx]),
            )
            tangent[idx, :-1] = spatial / dt
            tangent[idx, -1] = radial / dt
        step_tangents.append(tangent)
    return lambda_list, step_tangents


def cone_parallel_transport_explicit(
    a0: np.ndarray,
    b0: float,
    z0: Tuple[np.ndarray, float],
    z1: Tuple[np.ndarray, float],
) -> Tuple[np.ndarray, float]:
    """
    Explicit parallel transport along the cone geodesic joining z0 and z1.
    """
    x0, r0 = z0
    x1, r1 = z1
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    b0 = float(b0)
    r0 = float(r0)
    r1 = float(r1)

    if r0 <= _TOL or r1 <= _TOL:
        return np.zeros_like(a0), 0.0

    theta = min(np.linalg.norm(x1 - x0), np.pi)
    dist = cone_distance(x0, r0, x1, r1)
    if dist <= _TOL:
        return a0.copy(), b0

    if theta <= _TOL:
        unit_spatial_velocity = np.zeros_like(a0)
    else:
        unit_spatial_velocity = (
            np.sin(theta) * r1 * (x1 - x0) / (theta * r0 * dist)
        )
    unit_radial_velocity = (r1 * np.cos(theta) - r0) / dist

    q = r0**2 * unit_spatial_velocity
    c = np.linalg.norm(q)

    if c <= _TOL:
        return (r0 / r1) * a0, b0

    e = q / c
    a0_perp = a0 - np.dot(a0, e) * e
    alpha0 = float(np.dot(a0, e))
    theta_eval = np.arctan2(dist + r0 * unit_radial_velocity, c) - np.arctan2(
        r0 * unit_radial_velocity, c
    )

    a1 = (r0 / r1) * a0_perp + (
        (r0 * alpha0 * np.cos(theta_eval) - b0 * np.sin(theta_eval)) / r1
    ) * e
    b1 = r0 * alpha0 * np.sin(theta_eval) + b0 * np.cos(theta_eval)
    return a1, float(b1)


def cone_wasserstein_parallel_transport(
    lambda_list: list,
    lifted_tangents: Optional[list],
    cone_tangent0: np.ndarray,
    couplings: Optional[list] = None,
    step_size: Optional[float] = None,
    alignment_tol: float = 1e-2,
    return_path: bool = False,
    show_progress: bool = True,
):
    """
    Run the cone-side Wasserstein parallel transport recursion induced by a
    lifted path.

    This exposes the inner transport loop used by ``hk_parallel_transport`` so
    the cone dynamics can be inspected without projecting back to the HK tangent
    space. If ``couplings`` is provided, each step uses the retained local LET
    coupling to split source reference mass, parallel transport edgewise, and
    aggregate onto the next lifted support. Otherwise it falls back to the
    tangent-induced deterministic map update.
    """
    if len(lambda_list) == 0:
        raise ValueError("lambda_list must be non-empty")
    n_steps = len(lambda_list) - 1
    if lifted_tangents is not None and len(lifted_tangents) != n_steps:
        raise ValueError("lifted_tangents must have length len(lambda_list) - 1")
    if couplings is not None and len(couplings) != len(lambda_list) - 1:
        raise ValueError("couplings must have length len(lambda_list) - 1")
    if couplings is None and lifted_tangents is None:
        raise ValueError("Either couplings or lifted_tangents must be provided")

    current_lambda = lambda_list[0]
    cone_tangent0 = np.asarray(cone_tangent0, dtype=float)
    if cone_tangent0.shape != (current_lambda.n_samples, current_lambda.d + 1):
        raise ValueError("cone_tangent0 must have shape (lambda_list[0].n_samples, d + 1)")

    if step_size is None:
        if len(lambda_list) == 1:
            step_size = 0.0
        else:
            step_size = 1.0 / (len(lambda_list) - 1)
    if step_size < 0:
        raise ValueError("step_size must be non-negative")

    transported_tangent = cone_tangent0.copy()
    tangent_path = [transported_tangent.copy()] if return_path else None
    mapped_lambda_path = [] if return_path else None
    atom_index_path = [] if return_path else None
    coupling_path = [] if return_path else None

    for k in range(n_steps):
        if show_progress:
            print(f"Transporting step {k + 1}/{n_steps}...")

        next_lambda = lambda_list[k + 1]
        if couplings is not None:
            transported_tangent = _aggregate_cone_tangent_under_plan_coupling(
                transported_tangent,
                current_lambda,
                next_lambda,
                couplings[k],
            )
            current_lambda = next_lambda

            if return_path:
                tangent_path.append(transported_tangent.copy())
                mapped_lambda_path.append(next_lambda)
                atom_index_path.append(None)
                coupling_path.append(np.asarray(couplings[k], dtype=float).copy())
            continue

        lifted_tangent = np.asarray(lifted_tangents[k], dtype=float)
        if lifted_tangent.shape != (current_lambda.n_samples, current_lambda.d + 1):
            raise ValueError(
                f"lifted_tangents[{k}] must have shape ({current_lambda.n_samples}, {current_lambda.d + 1})"
            )
        mapped_lambda = _cone_exponential_map_step(current_lambda, lifted_tangent, step_size)
        mapped_atom_indices = _resolve_cone_atom_map(
            mapped_lambda,
            next_lambda,
            tol=1e-8, # hard code the tolerance here since it's an internal consistency check within the transport loop
        )
        next_tangent = np.zeros_like(mapped_lambda.to_cone_samples())

        iterator = range(current_lambda.n_samples)
        if show_progress:
            iterator = _maybe_tqdm(
                iterator,
                desc=f"Step {k + 1}/{len(lifted_tangents)}",
                leave=False,
            )

        for idx in iterator:
            a1, b1 = cone_parallel_transport_explicit(
                transported_tangent[idx, :-1],
                transported_tangent[idx, -1],
                (current_lambda.samples[idx], current_lambda.radii[idx]),
                (mapped_lambda.samples[idx], mapped_lambda.radii[idx]),
            )
            next_tangent[idx, :-1] = a1
            next_tangent[idx, -1] = b1

        transported_tangent = _aggregate_cone_tangent_under_deterministic_map(
            next_tangent,
            current_lambda.weights,
            mapped_atom_indices,
            next_lambda.n_samples,
        )
        current_lambda = next_lambda

        if return_path:
            tangent_path.append(transported_tangent.copy())
            mapped_lambda_path.append(mapped_lambda)
            atom_index_path.append(mapped_atom_indices.copy())
            coupling_path.append(None)

    if return_path:
        return transported_tangent, {
            "tangent_path": tangent_path,
            "mapped_lambda_path": mapped_lambda_path,
            "mapped_atom_indices": atom_index_path,
            "coupling_path": coupling_path,
        }
    return transported_tangent


# ============================================================================
# Main algorithms
# ============================================================================


def hk_parallel_transport(
    mu0: EmpiricalMeasure,
    mu1: EmpiricalMeasure,
    u0: Tuple[np.ndarray, np.ndarray],
    N: int = 5,
    let_solver: str = "pot_mm",
    entropy_reg: float = 0.01,
    let_max_iterations: int = 500,
    let_stop_threshold: float = 1e-8,
    let_reg_m: float = 1.0,
    radial_aggregation_mode: str = "mass_preserving",
    compression_max_atoms: Optional[int] = None,
    compression_kmeans_iterations: int = 20,
    alignment_tol: float = 1e-8,
    return_alignment_diagnostics: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical HK parallel transport via the paper's stepwise cone transport scheme.

    This follows Algorithm ``Approximate HK parallel transport via cone
    transport`` in ``main.tex`` using the lifted path and retained local LET
    couplings returned by ``isometric_lift`` as the single source of truth for
    the characteristic recursion.

    The Wasserstein tangent projection ``Pi_{lambda_t}`` appearing after each
    local cone transport step in the paper is intentionally omitted here, per the
    requested scope.

    If ``return_alignment_diagnostics`` is ``True``, the function returns a pair
    ``(transported_tangent, diagnostics)``. When the final empirical support
    cannot be aligned to ``mu1`` within ``alignment_tol``, the tangent result is
    ``None`` and the diagnostics dictionary contains the transported and target
    point clouds so they can be inspected visually.
    """
    if N <= 0:
        raise ValueError("N must be positive")

    lambda_list, _, local_steps = isometric_lift(
        mu0,
        mu1,
        N,
        let_solver=let_solver,
        entropy_reg=entropy_reg,
        let_max_iterations=let_max_iterations,
        let_stop_threshold=let_stop_threshold,
        let_reg_m=let_reg_m,
        radial_aggregation_mode=radial_aggregation_mode,
        compression_max_atoms=compression_max_atoms,
        compression_kmeans_iterations=compression_kmeans_iterations,
        return_steps=True,
    )
    local_couplings = [step.coupling for step in local_steps]
    current_lambda = lambda_list[0]
    cone_tangent0 = lift_tangent(
        u0,
        aggregate_empirical_measure(project_cone_measure(current_lambda)),
        current_lambda,
    )
    transported_tangent = cone_wasserstein_parallel_transport(
        lambda_list,
        None,
        cone_tangent0,
        couplings=local_couplings,
        step_size=1.0 / N,
        alignment_tol=alignment_tol,
        return_path=False,
        show_progress=True,
    )
    current_lambda = lambda_list[-1]

    if current_lambda.n_samples == mu1.n_samples and np.allclose(
        current_lambda.samples, mu1.samples, atol=1e-4, rtol=0.0
    ):
        result = project_tangent(transported_tangent, current_lambda)
        if return_alignment_diagnostics:
            return result, {
                "alignment_succeeded": True,
                "transported_support": current_lambda.samples.copy(),
                "target_support": mu1.samples.copy(),
                "nearest_neighbor_distances": np.zeros(current_lambda.n_samples, dtype=float),
                "alignment_tol": alignment_tol,
            }
        return result

    try:
        atom_indices = align_samples_to_support(
            current_lambda.samples,
            mu1.samples,
            tol=alignment_tol,
        )
    except ValueError:
        tree = cKDTree(mu1.samples)
        distances, _ = tree.query(current_lambda.samples, k=1)
        diagnostics = {
            "alignment_succeeded": False,
            "transported_support": current_lambda.samples.copy(),
            "target_support": mu1.samples.copy(),
            "nearest_neighbor_distances": np.asarray(distances, dtype=float),
            "alignment_tol": alignment_tol,
        }
        if return_alignment_diagnostics:
            return None, diagnostics
        raise

    result = project_tangent(
        transported_tangent,
        current_lambda,
        atom_indices=atom_indices,
        n_base_points=mu1.n_samples,
    )
    if return_alignment_diagnostics:
        tree = cKDTree(mu1.samples)
        distances, _ = tree.query(current_lambda.samples, k=1)
        return result, {
            "alignment_succeeded": True,
            "transported_support": current_lambda.samples.copy(),
            "target_support": mu1.samples.copy(),
            "nearest_neighbor_distances": np.asarray(distances, dtype=float),
            "alignment_tol": alignment_tol,
        }
    return result


def hk_distance(
    mu0: EmpiricalMeasure,
    mu1: EmpiricalMeasure,
    let_solver: str = "pot_mm",
    let_max_iterations: int = 500,
    let_stop_threshold: float = 1e-8,
    let_reg_m: float = 1.0,
) -> float:
    """Compute the empirical HK distance from the discrete LET energy."""
    pi, _ = solve_let_unbalanced_transport(
        mu0,
        mu1,
        max_iterations=let_max_iterations,
        method=let_solver,
        stop_threshold=let_stop_threshold,
        reg_m=let_reg_m,
    )
    ground_dist, _ = _pairwise_transport_cost(mu0, mu1)
    energy = let_functional(
        pi,
        mu0,
        mu1,
        pi.sum(axis=1),
        pi.sum(axis=0),
        ground_dist,
    )
    return float(np.sqrt(max(energy, 0.0)))
