# DualPerspective Python Package

Python interface for [DualPerspective.jl](https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl), a Julia package for solving large-scale KL divergence problems.

## Installation

```bash
pip install DualPerspective
```

The package will automatically:
1. Install Julia if not already installed (via juliacall)
2. Install the DualPerspective.jl Julia package
3. Perform precompilation to ensure fast performance from the first run

## Usage

```python
import numpy as np
from DualPerspective import DPModel, solve, regularize

# Generate sample data
np.random.seed(42)
n = 200  # dimension of solution
m = 100  # number of measurements
x0 = np.random.rand(n)
x0 = np.pi * x0 / np.sum(x0)  # Rescale to sum to pi
A = np.random.rand(m, n)
b = A @ x0

# Create and solve the problem
model = DPModel(A, b)
regularize(model, 1e-4)  # Optional: set regularization parameter
solution = solve(model)

print(f"Sum of solution: {np.sum(solution):.6f} (should be ≈ {np.pi:.6f})")
print(f"Optimal solution shape: {solution.shape}")
```

## Reinstalling the DualPerspective.jl package

To reinstall or update the DualPerspective.jl package, you can use the following command:

```bash
pip install --force-reinstall DualPerspective
```

## Performance Considerations

This Python interface uses Julia's precompilation features to ensure good performance from the first run. The first import of the package may take slightly longer as it sets up the Julia environment, but subsequent operations should be fast.

## Features

- Easy-to-use Python interface for DualPerspective.jl
- Automatic installation of Julia dependencies
- Support for large-scale KL divergence problems
- Integration with NumPy arrays
- Precompilation for consistent performance

## Requirements

- Python 3.7+
- NumPy
- juliacall

## License

This project is licensed under the MIT License. 