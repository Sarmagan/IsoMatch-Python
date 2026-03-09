"""
Translation of test.m — demo for IsoMatch, including random-swap refinement.

Run with:
    uv run python test_isomatch.py

Requires: numpy, scipy, scikit-image (optional but recommended).
Optional: matplotlib (for visualisation).
"""

import time
import numpy as np
from scipy.spatial.distance import pdist, squareform

from isomatch import isomatch_algorithm


def create_color_grid(colors: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    """Reshape (n x 3) color array into (rows x cols x 3) image."""
    rows, cols = grid_size
    return colors.reshape(rows, cols, 3)


def test() -> None:
    grid_size = (32, 32)
    n = grid_size[0] * grid_size[1]

    # Generate random RGB colours (matches test.m: rand_colors = rand(prod(grid_size), 3))
    rand_colors = np.random.rand(n, 3)

    # Build distance matrix (matches test.m: d_list = pdist(rand_colors))
    d_list   = pdist(rand_colors)
    d_matrix = squareform(d_list)

    # ── Run 1: IsoMatch without random swaps ──────────────────────────────────
    t0 = time.time()
    result_assignment, obj_res, obj_orig = isomatch_algorithm(
        d_matrix,
        grid_size=grid_size,
    )
    print(f"IsoMatch (no swaps) total execution time: {time.time() - t0:.4f} seconds.")
    print(f"  Objective: {obj_orig:.4f} -> {obj_res:.4f}\n")

    # ── Run 2: IsoMatch with random swaps ─────────────────────────────────────
    # Matches test.m commented section: options.num_swaps = 3e3
    NUM_SWAPS = 50000
    t0 = time.time()
    result_assignment_rs, obj_res_rs, obj_orig_rs = isomatch_algorithm(
        d_matrix,
        grid_size=grid_size,
        num_swaps=NUM_SWAPS,
    )
    print(f"IsoMatch ({NUM_SWAPS} random swaps) total execution time: {time.time() - t0:.4f} seconds.")
    print(f"  Objective: {obj_orig_rs:.4f} -> {obj_res_rs:.4f}\n")

    # ── Optional visualisation ────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(create_color_grid(rand_colors, grid_size))
        axes[0].set_title(f"Original  {obj_orig:.3f}", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(create_color_grid(rand_colors[result_assignment], grid_size))
        axes[1].set_title(f"Result  {obj_res:.3f}", fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(create_color_grid(rand_colors[result_assignment_rs], grid_size))
        axes[2].set_title(f"Result (rand swaps)  {obj_res_rs:.3f}", fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()
        
        # Save the figure to the current working directory
        image_filename = 'isomatch_comparison.png'
        plt.savefig(image_filename, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved as '{image_filename}'")
        
        plt.show()

    except ImportError:
        print("matplotlib not available — skipping visualisation.")


if __name__ == '__main__':
    test()