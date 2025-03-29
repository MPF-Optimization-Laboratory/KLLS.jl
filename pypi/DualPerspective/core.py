import numpy as np
from juliacall import Main as jl
import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Useful for development: environment variable to install the package from the local directory.
USE_LOCAL = os.environ.get('DUALPERSPECTIVE_USE_LOCAL', '').lower() in ('true', '1', 'yes')

# def _reinstall_dualperspective():
#     """Reinstall DualPerspective.jl from the repository."""
#     jl.seval(f"""
#         import Pkg
#         Pkg.rm("DualPerspective")
#         Pkg.add("DualPerspective")
#         Pkg.instantiate()
#         """)

def _initialize_julia():
    """Initialize Julia and load DualPerspective."""
    try:
        if USE_LOCAL:
            # Use local Julia project
            jl.seval(f"""
                import Pkg
                Pkg.develop("{ROOT_DIR}")
                """)
        else:
            # Use registry version
            jl.seval("""
                import Pkg
                if !haskey(Pkg.project().dependencies, "DualPerspective")
                    Pkg.add("DualPerspective")
                end
                """)

        jl.seval("""
            using DualPerspective
            solve = DualPerspective.solve!
            scale = DualPerspective.scale!
            regularize = DualPerspective.regularize!
            """)
                
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Julia or install DualPerspective: {str(e)}")

# Initialize on module import
_initialize_julia()

class DPModel:
    """Python wrapper for DualPerspective.jl's DPModel."""
    
    def __init__(self, A, b, q=None, C=None, c=None, λ=None):
        """
        Initialize a DPModel.
        
        Args:
            A: Matrix of shape (m, n)
            b: Vector of length m
            q: Optional prior vector of length n
            C: Optional covariance matrix of shape (n, n)
            c: Optional vector for linear term
            λ: Optional regularization parameter
        """
        # Convert numpy arrays to Julia arrays
        A_jl = jl.convert(jl.Matrix, A)
        b_jl = jl.convert(jl.Vector, b)
        
        kwargs = {}
        if q is not None:
            kwargs['q'] = jl.convert(jl.Vector, q)
        if C is not None:
            kwargs['C'] = jl.convert(jl.Matrix, C)
        if c is not None:
            kwargs['c'] = jl.convert(jl.Vector, c)
        if λ is not None:
            kwargs['λ'] = λ
            
        self.model = jl.DPModel(A_jl, b_jl, **kwargs)

    @property
    def A(self):
        return np.array(self.model.A)

    @property
    def b(self):
        return np.array(self.model.b)

    @classmethod
    def from_julia_model(cls, julia_model):
        """
        Create a DPModel directly from a Julia DPModel object.
        
        Args:
            julia_model: A Julia DPModel object
            
        Returns:
            DPModel: A Python wrapper for the Julia model
        """
        instance = cls.__new__(cls)
        instance.model = julia_model
        return instance

def solve(model, verbose=False, logging=0):
    """
    Solve the DualPerspective problem using SequentialSolve algorithm.
    
    Args:
        model: DualPerspectiveModel instance
        verbose: Whether to print root-finding progress information
        logging: Whether to print DualPerspective logging information

    Returns:
        numpy array containing the solution
    """
    s_model = jl.SequentialSolve()
    result = jl.solve(model.model, s_model, zverbose=verbose, logging=logging)
    return np.array(result.solution)

def scale(model, scale_factor):
    """
    Scale the problem.
    
    Args:
        model: KLLSModel instance
        scale_factor: Scaling factor
    """
    jl.scale(model.model, scale_factor)

def regularize(model, λ):
    """
    Set the regularization parameter.
    
    Args:
        model: DualPerspectiveModel instance
        λ: Regularization parameter
    """
    jl.regularize(model.model, λ)

def rand_dp_model(m, n, λ=1e-3):
    """
    Create a random DPModel with dimensions m x n.
    
    Args:
        m: Number of rows
        n: Number of columns
        λ: Regularization parameter (default: 1e-3)
        
    Returns:
        A DPModel instance with random data
    """
    julia_model = jl.randDPModel(m, n, λ=λ)
    return DPModel.from_julia_model(julia_model)
