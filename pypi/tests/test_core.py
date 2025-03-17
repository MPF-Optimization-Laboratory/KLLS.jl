import numpy as np
import pytest
from DualPerspective import DPModel, solve, regularize

def test_dp_model_creation():
    """Test basic model creation."""
    A = np.random.rand(10, 5)
    b = np.random.rand(10)
    model = DPModel(A, b)
    assert model.A.shape == (10, 5)
    assert model.b.shape == (10,)

def test_dp_model_with_optional_args():
    """Test model creation with optional arguments."""
    A = np.random.rand(10, 5)
    b = np.random.rand(10)
    q = np.random.rand(5)
    C = np.random.rand(5, 5)
    c = np.random.rand(5)
    λ = 0.1
    
    model = DPModel(A, b, q=q, C=C, c=c, λ=λ)
    assert model.A.shape == (10, 5)
    assert model.b.shape == (10,)

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
    A = np.random.rand(10, 5)
    b = np.random.rand(10)
    model = DPModel(A, b)
    λ = 0.1
    regularize(model, λ)
    # Note: We can't directly test the internal state, but we can verify it doesn't raise an error

def test_solve():
    """Test solving functionality."""
    A = np.random.rand(10, 5)
    b = np.random.rand(10)
    model = DPModel(A, b)
    solution = solve(model)
    assert solution.shape == (5,)
    assert not np.any(np.isnan(solution)) 