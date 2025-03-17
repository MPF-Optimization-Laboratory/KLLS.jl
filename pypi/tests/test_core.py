import numpy as np
import pytest
from DualPerspective import DPModel, solve, regularize
from juliacall import Main as jl

# Set random seeds for reproducibility
np.random.seed(42)

m = 10
n = 5

def test_dp_model_creation():
    """Test basic model creation."""
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    model = DPModel(A, b)
    assert model.A.shape == (m, n)
    assert model.b.shape == (m,)

def test_dp_model_with_optional_args():
    """Test model creation with optional arguments."""
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    q = np.random.rand(n)
    C_temp = np.random.rand(n, n)
    C = C_temp.T @ C_temp  # Create positive definite matrix
    c = np.random.rand(n)
    λ = 0.1
    
    model = DPModel(A, b, q=q, C=C, c=c, λ=λ)
    assert model.A.shape == (m, n)
    assert model.b.shape == (m,)

# def test_scale():
#     """Test scaling functionality."""
#     A = np.random.rand(10, 5)
#     b = np.random.rand(10)
#     model = DPModel(A, b)
#     scale_factor = 2.0
#     scale(model, scale_factor)
#     # Note: We can't directly test the internal state, but we can verify it doesn't raise an error

def test_regularize():
    """Test regularization functionality."""
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    model = DPModel(A, b)
    λ = 0.1
    regularize(model, λ)
    # Note: We can't directly test the internal state, but we can verify it doesn't raise an error

def test_solve():
    """Test solving functionality."""
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    model = DPModel(A, b)
    solution = solve(model)
    assert solution.shape == (n,)
    assert not np.any(np.isnan(solution)) 