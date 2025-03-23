import numpy as np
from juliacall import Main as jl
import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PYPI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Useful for development: environment variable to install the package from the local directory.
USE_LOCAL = os.environ.get('DUALPERSPECTIVE_USE_LOCAL', '').lower() in ('true', '1', 'yes')

# Read repository URL from pyproject.toml
def _get_repo_url():
    """Get the repository URL from pyproject.toml."""
    pyproject_path = os.path.join(PYPI_DIR, "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    return pyproject_data["project"]["urls"].get("Repository")

GITHUB_URL = _get_repo_url()

def _reinstall_dualperspective():
    """Reinstall DualPerspective.jl from the repository."""
    jl.seval(f"""
        import Pkg
        Pkg.add(url="{GITHUB_URL}")
        Pkg.resolve()
        """)

def _initialize_julia():
    """Initialize Julia and load DualPerspective from either local directory or GitHub."""
    try:
        if USE_LOCAL:
            # Use local Julia project
            jl.seval(f"""
                import Pkg
                Pkg.activate("{ROOT_DIR}")
                """)
        else:
            # Use GitHub version
            jl.seval(f"""
                import Pkg
                if !haskey(Pkg.project().dependencies, "DualPerspective")
                    Pkg.add(url="{GITHUB_URL}")
                    Pkg.resolve()
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

def version():
    """
    Return the current version of DualPerspective package.
    
    Returns:
        str: Version string of the DualPerspective package
    """
    return str(jl.DualPerspective.version())

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
        self.A = jl.convert(jl.Matrix, A)
        self.b = jl.convert(jl.Vector, b)
        
        kwargs = {}
        if q is not None:
            kwargs['q'] = jl.convert(jl.Vector, q)
        if C is not None:
            kwargs['C'] = jl.convert(jl.Matrix, C)
        if c is not None:
            kwargs['c'] = jl.convert(jl.Vector, c)
        if λ is not None:
            kwargs['λ'] = λ
            
        self.model = jl.DPModel(self.A, self.b, **kwargs)

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

def rand_dp_model(m, n):
    """
    Create a random DPModel with dimensions m x n.
    
    Args:
        m: Number of rows
        n: Number of columns
        
    Returns:
        A DPModel instance with random data
    """
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    return DPModel(A, b)
