"""
Translation of external/IsomapR1/Isomap.m

Original algorithm by Tenenbaum, de Silva, and Langford (2000).
Reference: Science 290(5500): 2319-2323.

Key translation notes:
- MATLAB's `eigs(..., 'LR')` (largest real part) → scipy.sparse.linalg.eigs with which='LR'
- MATLAB's 1-based indexing → 0-based throughout
- MATLAB cell array Y.coords{di} → Python dict/list Y['coords'][di]
- Floyd-Warshall inner loop kept as-is (vectorised per iteration, matching original)
"""

import numpy as np
from scipy.sparse.linalg import eigs
from typing import Optional

from l2_distance import l2_distance


def isomap(
    D: np.ndarray,
    n_fcn: str,
    n_size,
    options: Optional[dict] = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compute Isomap embedding.

    Args:
        D:      (N x N) symmetric distance matrix.
        n_fcn:  Neighbourhood function: ``'k'`` or ``'epsilon'``.
        n_size: Neighbourhood size (integer k, or float epsilon).
        options: Dictionary with optional fields:
            dims    – list/array of embedding dimensionalities (default 1..10)
            comp    – which connected component to embed (1-based, default 1)
            display – plot results? (default 1; ignored in this Python port)
            overlay – overlay graph? (default 1; ignored)
            verbose – print progress? (default 1)

    Returns:
        Y: dict with keys:
            'coords' – list indexed by position in *dims*; each entry is
                       (d x n_points) coordinate array.
            'index'  – 0-based indices of the embedded points.
        R: (len(dims),) array of residual variances.
        E: (N x N) int8 edge matrix (1 = edge exists).
    """
    N = D.shape[0]
    if D.shape[1] != N:
        raise ValueError("D must be a square matrix.")

    if n_fcn == 'k':
        K = int(n_size)
    elif n_fcn == 'epsilon':
        epsilon = float(n_size)
    else:
        raise ValueError("n_fcn must be 'k' or 'epsilon'.")

    # ── defaults ──────────────────────────────────────────────────────────────
    if options is None:
        options = {}
    dims    = np.array(options.get('dims',    np.arange(1, 11)))  # 1-based dims
    comp    = int(options.get('comp',    1))   # 1-based component index
    verbose = int(options.get('verbose', 1))
    # display / overlay are accepted but not acted on (no GUI in Python port)

    INF = 1000.0 * D.max() * N   # effectively infinite distance

    Y: dict = {'coords': [None] * len(dims), 'index': None}
    R = np.zeros(len(dims))

    # ── Step 1: Construct neighbourhood graph ────────────────────────────────
    if verbose:
        print("Constructing neighborhood graph...")

    D = D.astype(float).copy()

    if n_fcn == 'k':
        # MATLAB: [tmp, ind] = sort(D)   → sorts each COLUMN; ind is (N×N)
        #         for i = 1:N
        #           D(i, ind(2+K:end, i)) = INF
        # Row i, columns ind(K+1:end, i) → set distances FROM point i to
        # all but its K nearest neighbours to INF.
        # ind[:,i] is the sorted order of column i (i.e. rows sorted by D[:,i]).
        # ind[0,i] = self (distance 0), ind[1..K,i] = K nearest neighbours.
        ind = np.argsort(D, axis=0)   # column-wise sort: ind[:,i] sorts column i
        for i in range(N):
            # MATLAB sets ROW i, columns ind[K+1:, i] to INF
            D[i, ind[K + 1:, i]] = INF

    elif n_fcn == 'epsilon':
        # Divide by itself where <= epsilon gives 1.0; elsewhere gives >1.
        # MATLAB: D = D ./ (D <= epsilon)  then clip to INF.
        mask = D <= epsilon
        with np.errstate(divide='ignore', invalid='ignore'):
            D = np.where(mask, 1.0, D / D)   # 1 where neighbour, NaN/inf elsewhere
        D = np.where(np.isfinite(D), D, INF)  # replace non-finite with INF
        D = np.minimum(D, INF)

    # Symmetrise
    D = np.minimum(D, D.T)

    # Edge matrix (1 where a finite edge exists)
    E = (D != INF).astype(np.int8)

    # ── Step 2: Compute all-pairs shortest paths (Floyd-Warshall) ────────────
    if verbose:
        print("Computing shortest paths...")

    for k in range(N):
        # D = min(D, D[:, k:k+1] + D[k:k+1, :])  — vectorised, matches MATLAB
        D = np.minimum(D, D[:, k:k+1] + D[k:k+1, :])
        if verbose and (k + 1) % 20 == 0:
            print(f"  Iteration: {k + 1}")

    # ── Remove outliers / find connected components ───────────────────────────
    if verbose:
        print("Checking for outliers...")

    # MATLAB: n_connect = sum(~(D==INF))
    # sum() on a matrix defaults to column sums in MATLAB → (1×N) vector
    # n_connect(j) = number of finite-distance entries in column j
    n_connect = np.sum(D != INF, axis=0)               # (N,) column sums

    # MATLAB: [tmp, firsts] = min(D==INF)
    # min() defaults to column-wise in MATLAB.
    # D==INF is boolean; min picks the first False (0) in each column.
    # firsts(j) = row index of the first non-INF entry in column j.
    firsts = np.argmin(D == INF, axis=0)               # (N,) column-wise

    # Each unique 'first' represents one connected component
    comps_unique = np.unique(firsts)                   # sorted unique representatives

    # Size of each component = n_connect at its representative
    size_comps = n_connect[comps_unique]

    # Sort components by descending size (MATLAB: sort ascending then reverse)
    order = np.argsort(size_comps)[::-1]
    comps_unique = comps_unique[order]
    size_comps   = size_comps[order]

    n_comps = len(comps_unique)
    if comp > n_comps:
        comp = 1   # default: largest component

    # 0-based component index
    comp_idx = comp - 1

    if verbose:
        print(f"  Number of connected components in graph: {n_comps}")
        print(f"  Embedding component {comp} with {size_comps[comp_idx]} points.")

    # Indices of points in the chosen component (0-based)
    # MATLAB: find(firsts == comps(comp))  [1-based]
    Y['index'] = np.where(firsts == comps_unique[comp_idx])[0]

    D = D[np.ix_(Y['index'], Y['index'])]
    N = len(Y['index'])

    # ── Step 3: Classical MDS ─────────────────────────────────────────────────
    if verbose:
        print("Constructing low-dimensional embeddings (Classical MDS)...")

    # Double-centring of squared distances
    D2 = D ** 2
    row_mean = D2.mean(axis=1, keepdims=True)   # (N x 1)
    col_mean = D2.mean(axis=0, keepdims=True)   # (1 x N)
    grand_mean = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + grand_mean)

    # Largest-real-part eigenpairs (equivalent to MATLAB eigs(...,'LR'))
    max_dim = int(max(dims))
    # eigs needs k < N; guard against tiny graphs
    k_eigs = min(max_dim, N - 1)

    # scipy eigs returns eigenvalues that may be complex; take real parts
    vals, vecs = eigs(B, k=k_eigs, which='LR')
    vals = np.real(vals)
    vecs = np.real(vecs)

    # Sort descending (MATLAB sorts ascending then reverses)
    sort_order = np.argsort(vals)[::-1]
    vals = vals[sort_order]
    vecs = vecs[:, sort_order]

    D_flat = D.ravel()   # for residual variance computation

    for di, d in enumerate(dims):
        d = int(d)
        if d <= N:
            # MATLAB: coords = (vecs(:,1:d) .* sqrt(vals(1:d))')'  → (d x N)
            sqrt_vals = np.sqrt(np.maximum(vals[:d], 0.0))  # guard negatives
            coords = (vecs[:, :d] * sqrt_vals[np.newaxis, :]).T  # (d x N)
            Y['coords'][di] = coords

            # Residual variance: 1 - corr(embedded_dists, geodesic_dists)^2
            emb_dists = l2_distance(coords, coords).ravel()
            corr_mat = np.corrcoef(emb_dists, D_flat)
            R[di] = 1.0 - corr_mat[0, 1] ** 2

            if verbose:
                print(
                    f"  Isomap on {N} points with dimensionality {d}"
                    f"  --> residual variance = {R[di]:.6f}"
                )

    return Y, R, E
