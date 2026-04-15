"""
Empirical Hellinger-Kantorovich parallel transport utilities.

This module follows the paper's cone-lifting construction in an empirical setting:

1. solve the discrete logarithmic entropy transport (LET) problem
2. build the induced endpoint lifts on the cone from the exact discrete marginals
3. interpolate the lifted coupling by cone geodesics and project back to the base
4. lift an HK tangent at the source, transport it particlewise on the cone, and
   project the result back to the target empirical measure

The implementation targets empirical measures with varying total mass rather than
the population objects in the paper, so it should be read as a discrete
approximation of the continuous theory.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

try:
    import ot
except ImportError:  # pragma: no cover - optional dependency
    ot = None


_TOL = 1e-12


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
    - 'lbfgsb': the in-module direct dense L-BFGS-B solver

    The `entropy_reg` argument is retained for backward compatibility but is not
    used by either non-entropic backend.
    """
    del entropy_reg

    if method == "pot_mm":
        return solve_let_unbalanced_transport_pot(
            mu0,
            mu1,
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
    raise ValueError("method must be either 'pot_mm' or 'lbfgsb'")


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
) -> ConeMeasure:
    """
    Aggregate an exact pushforward when multiple source atoms share a destination.

    The new reference mass is the sum of pushed reference masses, and the new
    deterministic radius is chosen so that the projected base mass is preserved.
    """
    mapped_samples = np.asarray(mapped_samples, dtype=float)
    reference_weights = np.asarray(reference_weights, dtype=float)
    updated_radii = np.asarray(updated_radii, dtype=float)

    if mapped_samples.shape[0] == 0:
        dim = mapped_samples.shape[1] if mapped_samples.ndim == 2 else 0
        return ConeMeasure(np.zeros((0, dim)), np.zeros(0), np.zeros(0))

    unique_samples, inverse = _preserve_order_unique_rows(mapped_samples)
    reference_mass = np.bincount(
        inverse, weights=reference_weights, minlength=len(unique_samples)
    )
    projected_mass = np.bincount(
        inverse,
        weights=reference_weights * updated_radii**2,
        minlength=len(unique_samples),
    )

    new_radii = np.zeros(len(unique_samples), dtype=float)
    positive = reference_mass > _TOL
    new_radii[positive] = np.sqrt(projected_mass[positive] / reference_mass[positive])
    return ConeMeasure(unique_samples, new_radii, reference_mass)


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
) -> LocalHKStep:
    row_mass = pi.sum(axis=1)
    col_mass = pi.sum(axis=0)

    barycenters = mu_source.samples.copy()
    positive_rows = row_mass > _TOL
    if np.any(positive_rows):
        barycenters[positive_rows] = pi[positive_rows] @ mu_target.samples
        barycenters[positive_rows] /= row_mass[positive_rows, np.newaxis]

    u_source = np.zeros(mu_source.n_samples, dtype=float)
    u_source[positive_rows] = mu_source.weights[positive_rows] / row_mass[positive_rows]

    u_target = np.zeros(mu_target.n_samples, dtype=float)
    positive_cols = col_mass > _TOL
    u_target[positive_cols] = mu_target.weights[positive_cols] / col_mass[positive_cols]

    averaged_target_density = np.zeros(mu_source.n_samples, dtype=float)
    if np.any(positive_rows):
        averaged_target_density[positive_rows] = pi[positive_rows] @ u_target
        averaged_target_density[positive_rows] /= row_mass[positive_rows]

    q = np.zeros(mu_source.n_samples, dtype=float)
    valid_q = positive_rows & (u_source > _TOL)
    q[valid_q] = np.sqrt(
        np.maximum(averaged_target_density[valid_q] / u_source[valid_q], 0.0)
    )

    displacement = barycenters - mu_source.samples
    distance = np.linalg.norm(displacement, axis=1)
    v = np.zeros_like(mu_source.samples)
    moving = distance > _TOL
    if np.any(moving):
        speed_factor = (q[moving] * np.sin(distance[moving])) / (dt * distance[moving])
        v[moving] = speed_factor[:, np.newaxis] * displacement[moving]

    beta = 0.5 / dt * (q * np.cos(distance) - 1.0)
    beta[~positive_rows] = -0.5 / dt

    return LocalHKStep(
        map_positions=barycenters,
        q=q,
        v=v,
        beta=beta,
        row_mass=row_mass,
        col_mass=col_mass,
    )


def isometric_lift(
    mu0: EmpiricalMeasure,
    mu1: EmpiricalMeasure,
    N: int,
    let_solver: str = "pot_mm",
    let_max_iterations: int = 500,
    let_stop_threshold: float = 1e-8,
    let_reg_m: float = 1.0,
    compression_max_atoms: Optional[int] = None,
    compression_kmeans_iterations: int = 20,
    return_tangents: bool = False,
) -> Tuple[list, list]:
    """
    Approximate Algorithm 1 from the paper using empirical local LET solves.

    The endpoint LET lift is used only to produce target base measures at the
    sampled times. The lifted path itself is then rebuilt recursively using the
    barycentric local maps and radial updates from the isometric-lift algorithm.
    """
    if N <= 0:
        raise ValueError("N must be positive")

    dt = 1.0 / N
    endpoint_lambdas, _ = let_lift(
        mu0,
        mu1,
        N,
        let_solver=let_solver,
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
    for i in range(N):
        target_base = target_base_path[i + 1]
        pi, _ = solve_let_unbalanced_transport(
            current_base,
            target_base,
            max_iterations=let_max_iterations,
            method=let_solver,
            stop_threshold=let_stop_threshold,
            reg_m=let_reg_m,
        )

        local_step = _compute_local_hk_step(current_base, target_base, pi, dt)
        lifted_tangents.append(
            lift_tangent((local_step.v, local_step.beta), current_base, current_lambda)
        )

        updated_radii = current_lambda.radii * local_step.q
        current_lambda = _aggregate_pushforward_cone(
            local_step.map_positions,
            current_lambda.weights,
            updated_radii,
        )
        current_base = aggregate_empirical_measure(project_cone_measure(current_lambda))

        lambda_list.append(current_lambda)
        radii_list.append(current_lambda.radii.copy())

    if return_tangents:
        return lambda_list, radii_list, lifted_tangents
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


# ============================================================================
# Main algorithms
# ============================================================================


def hk_parallel_transport(
    mu0: EmpiricalMeasure,
    mu1: EmpiricalMeasure,
    u0: Tuple[np.ndarray, np.ndarray],
    N: int = 10,
    let_solver: str = "pot_mm",
    let_max_iterations: int = 500,
    let_stop_threshold: float = 1e-8,
    let_reg_m: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical HK parallel transport via endpoint cone lifts and cone transport.

    `N` is kept for API compatibility with the previous implementation. The
    transport itself is computed on the exact lifted endpoint coupling, while
    `isometric_lift` uses `N` to sample the intermediate cone path.
    """
    if N <= 0:
        raise ValueError("N must be positive")

    pi, _ = solve_let_unbalanced_transport(
        mu0,
        mu1,
        max_iterations=let_max_iterations,
        method=let_solver,
        stop_threshold=let_stop_threshold,
        reg_m=let_reg_m,
    )
    coupling = build_optimal_lifted_coupling(mu0, mu1, pi)
    lambda0 = coupling.lambda_source()
    lambda1 = coupling.lambda_target()

    U0 = lift_tangent(u0, mu0, lambda0, atom_indices=coupling.source_indices)
    U1 = np.zeros_like(U0)

    for idx in range(coupling.weights.shape[0]):
        a1, b1 = cone_parallel_transport_explicit(
            U0[idx, :-1],
            U0[idx, -1],
            (coupling.source_positions[idx], coupling.source_radii[idx]),
            (coupling.target_positions[idx], coupling.target_radii[idx]),
        )
        U1[idx, :-1] = a1
        U1[idx, -1] = b1

    return project_tangent(
        U1,
        lambda1,
        atom_indices=coupling.target_indices,
        n_base_points=mu1.n_samples,
    )


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
