"""
Translation of external/munkres/munkres.m

Original author: Yi Cao, Cranfield University (17 June 2008).
Reference: "Munkres' Assignment Algorithm, Modified for Rectangular Matrices"
           http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html

Key translation notes:
- MATLAB logical arrays → NumPy bool arrays.
- MATLAB `find(x, 1)` → np.argwhere / np.unravel_index with [0].
- MATLAB `bsxfun(@minus, ...)` → NumPy broadcasting.
- All indexing shifted from 1-based to 0-based.
- `ind2sub` / `sub2ind` → np.unravel_index / np.ravel_multi_index.
"""

import numpy as np


def munkres(cost_mat: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve the linear assignment problem with the Munkres (Hungarian) algorithm.

    Args:
        cost_mat: (M x N) cost matrix.  NaN entries are treated as Inf.

    Returns:
        assignment: Boolean matrix of the same shape; True at assigned pairs.
        cost:       Total cost of the optimal assignment.
    """
    cost_mat = cost_mat.astype(float).copy()
    assignment = np.zeros(cost_mat.shape, dtype=bool)
    total_cost = 0.0

    # Replace NaN with Inf (mirrors MATLAB: costMat(costMat~=costMat) = Inf)
    cost_mat[np.isnan(cost_mat)] = np.inf

    valid_mat = np.isfinite(cost_mat)
    valid_col = valid_mat.any(axis=0)   # (N,)
    valid_row = valid_mat.any(axis=1)   # (M,)

    n_rows = int(valid_row.sum())
    n_cols = int(valid_col.sum())
    n = max(n_rows, n_cols)

    if n == 0:
        return assignment, total_cost

    # Working square matrix filled with zeros (padding for rectangular inputs)
    d_mat = np.zeros((n, n))
    d_mat[:n_rows, :n_cols] = cost_mat[np.ix_(valid_row, valid_col)]

    # ── Step 1: subtract row minima ──────────────────────────────────────────
    d_mat -= d_mat.min(axis=1, keepdims=True)

    # ── Step 2: star a zero in each uncovered row and column ─────────────────
    zp = (d_mat == 0)
    star_z = np.zeros((n, n), dtype=bool)
    while zp.any():
        r, c = np.argwhere(zp)[0]
        star_z[r, c] = True
        zp[r, :] = False
        zp[:, c] = False

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:
        # Step 3: cover columns that contain a starred zero
        cover_col = star_z.any(axis=0)          # (n,)
        if cover_col.all():
            break

        cover_row = np.zeros(n, dtype=bool)      # (n,)
        prime_z   = np.zeros((n, n), dtype=bool)

        while True:
            # Step 4: find an uncovered zero and prime it
            zp[:] = False
            zp[np.ix_(~cover_row, ~cover_col)] = (
                d_mat[np.ix_(~cover_row, ~cover_col)] == 0
            )

            step = 6
            # Iterate while there are uncovered zeros
            while zp[np.ix_(~cover_row, ~cover_col)].any():
                uz_r, uz_c = np.argwhere(zp)[0]
                prime_z[uz_r, uz_c] = True

                stz = star_z[uz_r, :]            # starred zeros in this row
                if not stz.any():
                    step = 5
                    break

                cover_row[uz_r] = True
                cover_col[stz]  = False
                zp[uz_r, :]     = False
                uncov_stz_col   = np.where(stz)[0]
                zp[np.ix_(~cover_row, uncov_stz_col)] = (
                    d_mat[np.ix_(~cover_row, uncov_stz_col)] == 0
                )

            if step == 6:
                # Step 6: adjust the matrix
                M = d_mat[np.ix_(~cover_row, ~cover_col)]
                min_val = M.min()
                if min_val == np.inf:
                    return assignment, total_cost
                d_mat[np.ix_( cover_row,  cover_col)] += min_val
                d_mat[np.ix_(~cover_row, ~cover_col)] -= min_val
            else:
                break

        # Step 5: augmenting path
        # In the original MATLAB, uZr and uZc are both *boolean* vectors used
        # as logical row/column selectors throughout this loop.
        # rowZ1 = starZ(:, uZc)         → column of starZ at primed-zero col
        # starZ(uZr, uZc) = true        → star the primed zero
        # while any(rowZ1):
        #   starZ(rowZ1, uZc) = false   → unstar
        #   uZc = primeZ(rowZ1, :)      → logical row → becomes col selector
        #   uZr = rowZ1
        #   rowZ1 = starZ(:, uZc)
        #   starZ(uZr, uZc) = true

        # Build boolean selectors matching MATLAB
        uZr_bool = np.zeros(n, dtype=bool)
        uZr_bool[uz_r] = True
        uZc_bool = np.zeros(n, dtype=bool)
        uZc_bool[uz_c] = True

        row_z1 = star_z[:, uZc_bool].ravel()     # (n,) boolean
        star_z[np.ix_(uZr_bool, uZc_bool)] = True

        while row_z1.any():
            star_z[np.ix_(row_z1, uZc_bool)] = False
            # primeZ(rowZ1, :) → selects row(s) of primeZ where rowZ1 is True
            # The result is used as a column-selector (uZc), so it's boolean over cols
            uZc_bool = prime_z[np.ix_(row_z1, np.ones(n, dtype=bool))].ravel()
            uZr_bool = row_z1.copy()
            row_z1 = star_z[:, uZc_bool].ravel()
            star_z[np.ix_(uZr_bool, uZc_bool)] = True

    # ── Extract assignment ────────────────────────────────────────────────────
    # Map back from the padded square to the original (valid) rows/cols
    valid_rows_idx = np.where(valid_row)[0]
    valid_cols_idx = np.where(valid_col)[0]

    # star_z[:n_rows, :n_cols] selects the relevant sub-block
    sub = star_z[:n_rows, :n_cols]
    assigned_row, assigned_col = np.where(sub)

    # Map back to original indices
    orig_rows = valid_rows_idx[assigned_row]
    orig_cols = valid_cols_idx[assigned_col]

    assignment[orig_rows, orig_cols] = True
    total_cost = float(cost_mat[assignment].sum())

    return assignment, total_cost
