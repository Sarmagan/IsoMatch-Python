# IsoMatch Python 

Python implementation of the [IsoMatch MATLAB implementation](https://github.com/ohadf/isomatch).

---

## File Map

| Python file | Original MATLAB file(s) |
|---|---|
| `isomatch.py` | `isomatch.m` |
| `isomatch_helpers.py` | `EvaluateObjectiveFunc.m`, `RandomSwaps.m`, `MinBipartiteMatching.m`, `MatchToGrid.m`, `GenerateRegularGridCoordinates.m`, `CalculateBoundingBoxRectangle.m`, `points2bwimage.m` |
| `isomap.py` | `external/IsomapR1/Isomap.m` |
| `l2_distance.py` | `external/IsomapR1/L2_distance.m` |
| `munkres.py` | `external/munkres/munkres.m` |
| `test_isomatch.py` | `test.m` |
| `pyproject.toml` | — project & dependency metadata |
| `.python-version` | — pins Python 3.12 for `uv` |

---

## Quickstart with `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project manager.
It reads `pyproject.toml` and `.python-version` automatically, so setup is a
single command.

### 1. Install `uv`

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify the install:
```bash
uv --version
```

---

### 2. Create the virtual environment and install dependencies

```bash
# from the isomatch_py/ directory:
uv sync
```

`uv sync` will:
1. Read `.python-version` and download Python 3.12 if it is not already present.
2. Create a `.venv/` virtual environment inside the project folder.
3. Install all packages listed in `pyproject.toml` (`numpy`, `scipy`,
   `scikit-image`, `matplotlib`).

You do **not** need to activate the virtual environment for the commands below —
`uv run` handles that automatically.

---

### 3. Run the demo

```bash
uv run python test_isomatch.py
```

This runs the 20 × 20 random-colour grid test and (if `matplotlib` is available)
shows a before/after comparison plot.

---

### 4. Use IsoMatch in your own script

```bash
uv run python my_script.py
```

where `my_script.py` contains, for example:

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from isomatch import isomatch_algorithm

# Build a pairwise distance matrix for your data
colors = np.random.rand(400, 3)
d_matrix = squareform(pdist(colors))

# Run IsoMatch — assign 400 items to a 20×20 grid
assignment, obj_final, obj_initial = isomatch_algorithm(
    d_matrix,
    grid_size=(20, 20),
)
# assignment[i] is the 0-based grid position for input element i
arranged_colors = colors[assignment]
print(f"Objective: {obj_initial:.4f} → {obj_final:.4f}")
```

---

### 5. Optional: activate the environment in your shell

If you prefer to call `python` directly without prefixing every command with
`uv run`:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Then:
```bash
python test_isomatch.py
```

Deactivate when done:
```bash
deactivate
```

---

### 6. Add / remove packages

```bash
uv add <package>       # add a new dependency and update pyproject.toml
uv remove <package>    # remove a dependency
uv sync                # re-sync the environment after manual edits to pyproject.toml
```

---

## Alternative: plain `pip`

If you prefer not to use `uv`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy scipy scikit-image matplotlib
python test_isomatch.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | All matrix / array operations |
| `scipy` | `pdist`, `squareform`, `eigs`, `gaussian_filter`, `ndimage.label` |
| `scikit-image` | Otsu threshold (`threshold_otsu`) in bounding-box step |
| `matplotlib` | Visualisation in `test_isomatch.py` (optional) |

---

## Implementation Notes

- All indexing is **0-based** (converted from MATLAB's 1-based).
- `scipy.spatial.distance.pdist` / `squareform` replace MATLAB's `pdist` / `squareform`.
- `scipy.sparse.linalg.eigs(..., which='LR')` replaces MATLAB's `eigs(..., 'LR')`.
- The image-processing step in `calculate_bounding_box_rectangle` replaces
  MATLAB's `fspecial` / `graythresh` / `im2bw` / `bwboundaries` with
  `scipy.ndimage.gaussian_filter`, `skimage.filters.threshold_otsu`, and
  `scipy.ndimage.label`.
