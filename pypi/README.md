# DualPerspective Python Package

Python interface for [DualPerspective.jl](https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl), a Julia package for solving large-scale KL divergence problems.

## Installation

```bash
pip install DualPerspective
```

The package will automatically:
1. Install Julia if not already installed (via juliacall)
2. Install the DualPerspective.jl Julia package from the official Julia registry
3. Perform precompilation to ensure fast performance from the first run

## Usage

```python
import numpy as np
from DualPerspective import DPModel, solve, regularize

# Generate sample data
np.random.seed(42)
n = 200  # dimension of solution
m = 100  # number of measurements
x0 = np.pi * (tmp := np.random.rand(n)) / np.sum(tmp)
A = np.random.rand(m, n)
b = A @ x0  # measurements

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

## Advanced Usage

### Reinstalling the Julia Package

If you need to reinstall the Julia package, you can simply reinstall the Python package:

```bash
pip install --force-reinstall DualPerspective
```

This will ensure you have the latest version from the Julia registry.

## Development

### Building and Publishing the Package

Follow these steps to build and publish new versions of the package to PyPI:

#### Prerequisites

Make sure you have the necessary tools:

```bash
pip install --upgrade build twine
```

#### Building the Package

Build both wheel and source distributions:

```bash
python -m build
```

This creates distribution files in the `dist/` directory.

#### Testing Locally

Before publishing, test the package locally:

```bash
# Install in development mode
pip install -e .

# Or install the built wheel
pip install dist/DualPerspective-0.1.1-py3-none-any.whl
```

#### Publishing to TestPyPI (Optional)

To test the publishing process:

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ DualPerspective
```

#### Publishing to PyPI

Once tested and ready:

```bash
twine upload dist/*
```

#### Updating for New Releases

1. Update the version number in `pyproject.toml`
2. Make code changes
3. Rebuild: `python -m build`
4. Upload: `twine upload dist/*`