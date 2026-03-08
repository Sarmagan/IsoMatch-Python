"""
Translations of the remaining IsoMatch helper modules:

  - MinBipartiteMatching.m
  - MatchToGrid.m
  - GenerateRegularGridCoordinates.m
  - EvaluateObjectiveFunc.m
  - RandomSwaps.m
  - CalculateBoundingBoxRectangle.m  (uses image-processing approximation)
  - points2bwimage.m
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.ndimage import gaussian_filter
from typing import Optional

from munkres import munkres


# ─────────────────────────────────────────────────────────────────────────────
# MinBipartiteMatching
# ─────────────────────────────────────────────────────────────────────────────

def min_bipartite_matching(
    dist_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the minimum-cost bipartite matching via the Munkres algorithm.

    MATLAB transposes the input before calling munkres, then uses column-major
    ind2sub to extract pairs.  The critical invariant that isomatch.m relies on
    is that assignment_from is sorted [0, 1, ..., M-1], so that
    ``assignment_to[i]`` directly gives the grid cell for source point i.

    MATLAB achieves this automatically because column-major ind2sub on the
    (size_u x size_v) assignment matrix iterates columns (source indices) before
    rows (grid indices), yielding pairs sorted by source index.  NumPy's
    np.where sorts row-major, so we explicitly sort by assignment_from.

    Args:
        dist_matrix: (M x N) cost matrix — M source points, N grid cells.

    Returns:
        assignment_from: sorted array [0, 1, ..., M-1] of source indices.
        assignment_to:   corresponding grid-cell index for each source point.
    """
    # MATLAB: distMatrix = distMatrix'  (N_grid x N_source)
    dist_matrix_T = dist_matrix.T

    x, _ = munkres(dist_matrix_T)   # boolean (N_grid x N_source) matrix
                                    # x[grid_i, source_j] = True means source j -> grid i

    grid_idx, source_idx = np.where(x)

    # Sort by source index so assignment_from = [0, 1, ..., M-1] in ascending order.
    # This matches MATLAB's column-major ind2sub which naturally iterates source
    # indices (columns) first, yielding pairs sorted by assignment_from.
    sort_order      = np.argsort(source_idx)
    assignment_from = source_idx[sort_order]   # [0, 1, ..., M-1]
    assignment_to   = grid_idx[sort_order]     # grid cell for each source point

    return assignment_from, assignment_to


# ─────────────────────────────────────────────────────────────────────────────
# MatchToGrid
# ─────────────────────────────────────────────────────────────────────────────

def match_to_grid(
    orig_coords: np.ndarray,
    grid_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each input point to the best-matching grid location.

    Args:
        orig_coords: (N x 2) coordinates of the points to assign.
        grid_coords: (M x 2) grid coordinates.  Must satisfy M >= N.

    Returns:
        assignment_from: 0-based source indices.
        assignment_to:   0-based target (grid) indices for each source point.
    """
    # pdist2 in MATLAB → scipy cdist (both give pairwise Euclidean distances)
    D = cdist(orig_coords, grid_coords)              # (N x M)

    assignment_from, assignment_to = min_bipartite_matching(D)
    return assignment_from, assignment_to


# ─────────────────────────────────────────────────────────────────────────────
# GenerateRegularGridCoordinates
# ─────────────────────────────────────────────────────────────────────────────

def generate_regular_grid_coordinates(
    num_rows: int,
    num_cols: int,
    from_x: float = 0.0,
    to_x: float   = 1.0,
    from_y: float = 0.0,
    to_y: float   = 1.0,
) -> np.ndarray:
    """Create (num_rows * num_cols) x 2 regular grid coordinates.

    Args:
        num_rows: Number of grid rows.
        num_cols: Number of grid columns.
        from_x, to_x: X-axis range.
        from_y, to_y: Y-axis range.

    Returns:
        grid_coordinates: ((num_rows * num_cols) x 2) array of (x, y) pairs.

    Translation note:
        MATLAB meshgrid(y_vals, x_vals) produces Y(i,j)=y_vals(j) and
        X(i,j)=x_vals(i).  np.meshgrid replicates this with the same
        argument order, but numpy uses row-major flattening while MATLAB
        uses column-major.  We call np.meshgrid with indexing='ij' (matrix
        indexing) to match MATLAB's X(:) / Y(:) column-major flatten.
    """
    x_vals = np.linspace(from_x, to_x, num_cols)
    y_vals = np.linspace(from_y, to_y, num_rows)

    # MATLAB: [Y, X] = meshgrid(y_vals, x_vals)
    # indexing='ij' gives X[i,j]=x_vals[i], Y[i,j]=y_vals[j] — same layout.
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    grid_coordinates = np.column_stack([X.ravel(order='F'),
                                        Y.ravel(order='F')])
    return grid_coordinates


# ─────────────────────────────────────────────────────────────────────────────
# EvaluateObjectiveFunc  (internal + public)
# ─────────────────────────────────────────────────────────────────────────────

def _find_minimizer_l1(x: np.ndarray, y: np.ndarray) -> float:
    """Find C minimising f(C) = sum |C*x_i - y_i| via the 'fancy' sorted method.

    Translation note:
        MATLAB cumsum(x_sorted(end:-1:1)) reverses the array first; in NumPy
        we reverse with [::-1] then cumsum, then reverse back.
        The sign-product test for zero-crossing is kept identical.
    """
    x = x.ravel()
    y = y.ravel()
    assert x.size == y.size

    candidates = y / x                              # potential minimisers

    sort_order = np.argsort(candidates)
    x_sorted   = x[sort_order]

    # MATLAB: tmp1 = cumsum(x_sorted(end:-1:1)); tmp1 = [tmp1(end:-1:1); 0]
    rev_cumsum = np.cumsum(x_sorted[::-1])[::-1]   # (n,)
    tmp1 = np.append(rev_cumsum, 0.0)              # (n+1,)

    # MATLAB: tmp2 = [0; cumsum(x_sorted)]
    tmp2 = np.concatenate([[0.0], np.cumsum(x_sorted)])   # (n+1,)

    indicators = tmp2 - tmp1                        # (n+1,)

    # Zero-crossing: sign(indicators[0:n]) * sign(indicators[1:n+1]) <= 0
    indicators2 = np.sign(indicators[:-1]) * np.sign(indicators[1:])  # (n,)

    c_min_candidates = candidates[sort_order[indicators2 <= 0]]
    return float(c_min_candidates[0])              # take first (MATLAB: C_min(2:end)=[])


def _evaluate_objective_func_internal(
    distances1: np.ndarray,
    distances2: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Compute all three objective function variants.

    Returns:
        result, result_w, result_l1, C, C_w, C_l1
    """
    d1 = distances1.ravel().astype(float)
    d2 = distances2.ravel().astype(float)

    # ── Objective 1: unweighted L2 ───────────────────────────────────────────
    C = float(np.dot(d1, d2) / np.dot(d1, d1))
    result = float(
        np.sqrt(np.sum((C * d1 - d2) ** 2)) / np.sqrt(np.sum(d2 ** 2))
    )

    # ── Objective 2: weighted L2 ─────────────────────────────────────────────
    ALPHA = 2
    max_d1 = d1.max()
    D_w = (1.0 - d1 / max_d1) ** ALPHA
    C_w = float(np.dot(D_w * d1, d2) / np.dot(D_w * d1, d1))
    result_w = float(
        np.sqrt(np.sum((C_w * d1 - d2) ** 2 * D_w)) / np.sqrt(np.sum(d2 ** 2))
    )

    # ── Objective 3: L1 ──────────────────────────────────────────────────────
    C_l1 = _find_minimizer_l1(d1, d2)
    result_l1 = float(np.sum(np.abs(C_l1 * d1 - d2)) / np.sum(d2))

    return result, result_w, result_l1, C, C_w, C_l1


def evaluate_objective_func(
    distances1: np.ndarray,
    distances2: np.ndarray,
) -> tuple[float, float]:
    """Public wrapper matching MATLAB's EvaluateObjectiveFunc.

    MATLAB returns the L1 objective and its scale factor:
        [~, ~, result, ~, ~, C] = EvaluateObjectiveFunc_internal(...)
    i.e. the 3rd output (result_l1) and 6th output (C_l1).

    This wrapper is used both for final obj_val reporting in isomatch.m AND
    inside RandomSwaps (via its local EvaluateObjectiveFunc_internal which
    calls this public wrapper).  Both must use L1 to match MATLAB.

    Args:
        distances1: 1-D condensed pairwise distances for the input objects.
        distances2: 1-D condensed pairwise Euclidean distances on the grid.

    Returns:
        result_l1: L1 objective value (lower is better).
        C_l1:      Optimal L1 scale factor.
    """
    # MATLAB: [~, ~, result, ~, ~, C] = EvaluateObjectiveFunc_internal(...)
    # Positions:  0    1      2    3    4      5
    _, _, result_l1, _, _, C_l1 = _evaluate_objective_func_internal(distances1, distances2)
    return result_l1, C_l1


# ─────────────────────────────────────────────────────────────────────────────
# RandomSwaps
# ─────────────────────────────────────────────────────────────────────────────

def _swap_2_indices_matrix(m: np.ndarray, n: int, ind_pair: tuple[int, int]) -> np.ndarray:
    """Swap two rows and the corresponding columns in a square matrix.

    Replicates MATLAB's Swap2Indices_matrix (1-based → 0-based indices handled
    at call site).
    """
    indices = np.arange(n)
    i, j = ind_pair
    indices[i], indices[j] = j, i         # swap
    m = m[np.ix_(indices, indices)]
    return m


def _swap_2_indices_vector(v: np.ndarray, ind_pair: tuple[int, int]) -> np.ndarray:
    """Swap two elements in a vector (in-place copy)."""
    v = v.copy()
    i, j = ind_pair
    v[i], v[j] = v[j], v[i]
    return v


def random_swaps(
    image_dist_matrix: np.ndarray,
    grid_dist_matrix: np.ndarray,
    num_swaps: int = 30000,
    threshold: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Refine the grid assignment with random pairwise swaps.

    Args:
        image_dist_matrix: (n x n) symmetric distance matrix of input objects.
        grid_dist_matrix:  (n x n) symmetric Euclidean distance matrix of grid.
        num_swaps:         Maximum number of swap attempts.
        threshold:         Stop early if objective drops below this value.

    Returns:
        grid_permutation: 0-based permutation array of length n.
        obj_value:        Final objective value.

    Translation note:
        MATLAB randi(n, 1, 2) → two independent uniform integers in [1, n]
        → np.random.randint(0, n, size=2) in 0-based Python.
    """
    n = image_dist_matrix.shape[0]
    grid_permutation = np.arange(n)

    # Evaluate initial objective using squareform to extract upper-triangle vectors
    dist_images = squareform(image_dist_matrix)
    dist_grid   = squareform(grid_dist_matrix)
    obj_value, _ = evaluate_objective_func(dist_images, dist_grid)

    for _ in range(num_swaps):
        if obj_value < threshold:
            break

        # Draw two (possibly equal) swap indices — 0-based
        swap_indices = tuple(np.random.randint(0, n, size=2))

        grid_dist_matrix  = _swap_2_indices_matrix(grid_dist_matrix, n, swap_indices)
        grid_permutation  = _swap_2_indices_vector(grid_permutation, swap_indices)

        dist_grid_new = squareform(grid_dist_matrix)
        current_result, _ = evaluate_objective_func(dist_images, dist_grid_new)

        if current_result > obj_value:
            # Revert
            grid_dist_matrix = _swap_2_indices_matrix(grid_dist_matrix, n, swap_indices)
            grid_permutation = _swap_2_indices_vector(grid_permutation, swap_indices)
        else:
            obj_value = current_result

    return grid_permutation, obj_value


# ─────────────────────────────────────────────────────────────────────────────
# points2bwimage
# ─────────────────────────────────────────────────────────────────────────────

def points2bwimage(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    """Convert scattered point locations to a binary image.

    Translation note:
        MATLAB sub2ind(size(img), round(y_vals), round(x_vals)) uses 1-based
        row/col.  We subtract 1 after rounding to shift to 0-based indices.

    Returns:
        img:   Boolean array (rows × cols).
        mult_x, add_x, mult_y, add_y: Affine parameters to map pixel coords
            back to the original coordinate space.
    """
    IMG_NUM_COLS = 200

    max_x, min_x = x_vals.max(), x_vals.min()
    max_y, min_y = y_vals.max(), y_vals.min()
    range_x = max_x - min_x
    range_y = max_y - min_y

    img_num_rows = int(np.ceil(IMG_NUM_COLS * range_y / range_x))
    img = np.zeros((img_num_rows, IMG_NUM_COLS), dtype=bool)

    # Normalise x to [0, IMG_NUM_COLS-1]  (MATLAB: [1, IMG_NUM_COLS] 1-based)
    x_n = x_vals - min_x
    mult_x = x_n.max() / (IMG_NUM_COLS - 1)
    x_n = x_n / mult_x          # now in [0, IMG_NUM_COLS-1]

    y_n = y_vals - min_y
    mult_y = y_n.max() / (img_num_rows - 1)
    y_n = y_n / mult_y          # now in [0, img_num_rows-1]

    # Round to nearest pixel and clip to valid range (0-based indices)
    col_idx = np.clip(np.round(x_n).astype(int), 0, IMG_NUM_COLS - 1)
    row_idx = np.clip(np.round(y_n).astype(int), 0, img_num_rows - 1)

    img[row_idx, col_idx] = True

    add_x = min_x
    add_y = min_y

    return img, mult_x, add_x, mult_y, add_y


# ─────────────────────────────────────────────────────────────────────────────
# CalculateBoundingBoxRectangle
# ─────────────────────────────────────────────────────────────────────────────

def calculate_bounding_box_rectangle(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Estimate a loose bounding box for the point cloud, discarding outliers.

    Approximates the MATLAB version which used:
      fspecial('gaussian', 31, 10) * 20 → gaussian_filter with sigma≈10
      graythresh / im2bw                → Otsu threshold (skimage or manual)
      bwboundaries                      → contour / connected-component analysis

    Translation note:
        skimage.filters.threshold_otsu replicates graythresh.
        We avoid a full contour extraction and instead find the largest
        connected component via scipy.ndimage.label, then take its bounding box.
        This matches the intent of the MATLAB code (largest component → its
        extent in the original coordinate space).

    Returns:
        out_rectangle: [min_x, min_y, max_x, max_y] in original coordinates.
    """
    from scipy.ndimage import label

    img, mult_x, add_x, mult_y, add_y = points2bwimage(x, y)

    # Gaussian blur (sigma=10 approximates fspecial('gaussian',31,10))
    # The ×20 scale factor in MATLAB only affects the absolute grey values,
    # not the Otsu threshold decision — so we omit it.
    isomap_gray = gaussian_filter(img.astype(float), sigma=10)

    # Otsu thresholding — replaces graythresh/im2bw
    try:
        from skimage.filters import threshold_otsu
        level = threshold_otsu(isomap_gray)
    except ImportError:
        # Fallback: simple mean threshold if skimage is unavailable
        level = isomap_gray.mean()

    isomap_bw2 = isomap_gray >= level

    # Label connected components and pick the largest one
    labeled, num_features = label(isomap_bw2)
    if num_features == 0:
        # Degenerate: no foreground; fall back to full-image extents
        return np.array([x.min(), y.min(), x.max(), y.max()])

    comp_sizes = np.bincount(labeled.ravel())
    comp_sizes[0] = 0                          # ignore background label 0
    largest_label = int(comp_sizes.argmax())

    rows, cols = np.where(labeled == largest_label)

    # MATLAB fliplr(boundary_pixels) swaps (row, col) → (col, row) = (x, y).
    # Our 'cols' already correspond to x, 'rows' to y.

    # Map pixel indices (0-based) back to original isomap coordinates.
    # MATLAB: pixel_x = (col_1based - 1) * mult_x + add_x
    #       → pixel_x = col_0based * mult_x + add_x
    bp_x = cols.astype(float) * mult_x + add_x
    bp_y = rows.astype(float) * mult_y + add_y

    out_rectangle = np.array([bp_x.min(), bp_y.min(), bp_x.max(), bp_y.max()])
    return out_rectangle
