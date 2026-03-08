"""
Translation of isomatch.m — the main entry point of the IsoMatch algorithm.

IsoMatch assigns N elements (described by a pairwise distance matrix) to
grid/arbitrary locations while preserving their mutual distances as faithfully
as possible.

Reference:
    O. Fried, A. Finkelstein, M. Agrawala (2015).
    "IsoMatch: Creating Informative Grid Layouts."
    Computer Graphics Forum (Proc. Eurographics) 34(2).

Key translation notes:
- MATLAB inputParser → standard Python keyword arguments with defaults.
- squareform(pdist(...)) from scipy matches MATLAB's squareform(pdist(...)).
- MATLAB sort() returns [sorted_vals, sort_indices]; we use np.argsort.
- 1-based index arrays → 0-based throughout.
"""

import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Optional

from isomap import isomap
from isomatch_helpers import (
    calculate_bounding_box_rectangle,
    generate_regular_grid_coordinates,
    match_to_grid,
    random_swaps,
    evaluate_objective_func,
)


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────────────────────────────────────

def _validate_distance_matrix(d_matrix: np.ndarray) -> bool:
    """Return True iff d_matrix is symmetric with zeros on the diagonal."""
    return (
        np.allclose(d_matrix, d_matrix.T)
        and not np.any(np.diag(d_matrix))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def isomatch_algorithm(
    d_matrix: np.ndarray,
    *,
    isomap_neighbor_count: int = 13,
    isomap_ndims: int = 2,
    grid_size: Optional[tuple[int, int]] = None,
    grid_coords: Optional[np.ndarray] = None,
    num_swaps: int = 0,
    swap_threshold: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Assign elements to locations while preserving pairwise distances.

    Args:
        d_matrix:               (n x n) symmetric distance matrix with zeros
                                on the diagonal.
        isomap_neighbor_count:  Number of neighbours for the Isomap k-NN graph.
        isomap_ndims:           Dimensionality of the Isomap embedding.
        grid_size:              (rows, cols) tuple for a regular grid layout.
                                Mutually exclusive with *grid_coords*.
        grid_coords:            (m x 2) array of arbitrary target locations.
                                Mutually exclusive with *grid_size*.
        num_swaps:              Number of random-swap refinement iterations.
        swap_threshold:         Stop random swaps once energy drops below this.

    Returns:
        reverse_assignment: 0-based permutation of length n.  Element i of
                            the *original* ordering maps to position
                            reverse_assignment[i] in the output grid.
        obj_val_fin:        Objective (energy) value after optimisation.
        obj_val_init:       Objective (energy) value before optimisation.

    Raises:
        ValueError: If d_matrix fails validation or required grid arguments
                    are missing.
    """
    if not _validate_distance_matrix(d_matrix):
        raise ValueError(
            "d_matrix must be symmetric with zeros on the diagonal."
        )

    # ── Isomap embedding ──────────────────────────────────────────────────────
    t0 = time.time()
    iso_options = {
        'dims':    np.array([isomap_ndims]),
        'display': 0,
        'verbose': 0,
    }
    Y, _, _ = isomap(d_matrix, 'k', isomap_neighbor_count, iso_options)
    print(f"isomap execution time: {time.time() - t0:.6f} seconds.")

    num_images = len(Y['index'])

    # Y.coords{1} in MATLAB is the first (and here only) entry; shape (d x n)
    isomap_coords = Y['coords'][0]                   # (isomap_ndims x num_images)
    isomap_result_x = isomap_coords[0, :].reshape(-1, 1)   # (n x 1)
    isomap_result_y = isomap_coords[1, :].reshape(-1, 1)   # (n x 1)

    # ── Coarse alignment — bounding box ───────────────────────────────────────
    grid_rectangle = calculate_bounding_box_rectangle(
        isomap_result_x.ravel(), isomap_result_y.ravel()
    )

    locations = np.hstack([isomap_result_x, isomap_result_y])  # (n x 2)

    # ── Build grid coordinates ────────────────────────────────────────────────
    print("Calculating bipartite matching...")

    if grid_coords is None:
        if grid_size is None:
            raise ValueError(
                "Either grid_size or grid_coords must be supplied."
            )
        rows, cols = grid_size
        gc = generate_regular_grid_coordinates(
            rows, cols,
            from_x=grid_rectangle[0], to_x=grid_rectangle[2],
            from_y=grid_rectangle[1], to_y=grid_rectangle[3],
        )
        # Discard excess grid cells (MATLAB: grid_coords(num_images+1:end, :) = [])
        gc = gc[:num_images, :]
    else:
        gc = np.asarray(grid_coords)

    # ── Bipartite matching ────────────────────────────────────────────────────
    _, assignment = match_to_grid(locations, gc)    # assignment: 0-based grid indices

    # Verify it is a valid permutation of 0 .. num_images-1
    assert np.array_equal(
        np.sort(assignment), np.arange(num_images)
    ), "assignment is not a permutation of 0:num_images-1"

    new_grid_coords = gc[assignment, :]              # reorder grid by assignment

    # ── Optional random-swap refinement ──────────────────────────────────────
    # grid_perm: 0-based permutation that further reorders new_grid_coords
    grid_perm = np.arange(num_images)

    if num_swaps > 0 or swap_threshold > 0.0:
        t0 = time.time()
        grid_dist_sq = squareform(pdist(new_grid_coords))
        grid_perm, _ = random_swaps(
            d_matrix, grid_dist_sq, num_swaps, swap_threshold
        )
        print(f"Random swaps execution time: {time.time() - t0:.6f} seconds.\n")

    # ── Objective function values ─────────────────────────────────────────────
    # MATLAB: d_list = squareform(d_matrix)   → extracts upper-triangle condensed vector
    # obj_val_init = EvaluateObjectiveFunc(d_list, pdist(grid_coords))
    # obj_val_fin  = EvaluateObjectiveFunc(d_list, pdist(new_grid_coords(grid_perm,:)))
    d_list_sq = squareform(d_matrix)   # condensed pairwise distances of inputs

    obj_val_init, _ = evaluate_objective_func(
        d_list_sq,
        pdist(gc)
    )
    obj_val_fin, _ = evaluate_objective_func(
        d_list_sq,
        pdist(new_grid_coords[grid_perm, :])
    )

    # ── Final reverse assignment ──────────────────────────────────────────────
    # MATLAB: [~, reverse_assignment] = sort(assignment(grid_perm))
    # sort() gives argsort in 0-based Python.
    combined = assignment[grid_perm]                 # apply grid_perm to assignment
    reverse_assignment = np.argsort(combined)        # 0-based

    return reverse_assignment, obj_val_fin, obj_val_init
