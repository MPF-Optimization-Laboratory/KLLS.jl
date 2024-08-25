import numpy as np
import matplotlib.pyplot as plt

#######################################################################
# The next block of code installs the Julia solver and dependencies.
#
# IMPORTANT: requires the `juliacall` Python package, eg, from the
# command line, run `pip install juliacall`.
#######################################################################
from juliacall import Main as jl
jl.seval("""
         import Pkg
         Pkg.add(url=\"git@github.com:MPF-Optimization-Laboratory/KLLS.jl.git\")
         """)

#######################################################################
# `using KLLS` is equivalent to `import KLLS` in Python.
#
# Python doesn't allow the `!` character in variable or method names.
# But in Julia the `!` character is often used to indicate that a
# method modifies its arguments. The next block of code renames some
# of the methods in the Julia package to avoid this conflict.
#######################################################################
jl.seval("""
         using KLLS
         solve = KLLS.solve!
         scale = KLLS.scale!
         regularize = KLLS.regularize!
         lse = KLLS.obj!
         """)
#######################################################################

# Generate some data
n = 100
m = 200
x0 = np.random.rand(n)
A = np.random.rand(m, n)
b = A @ x0

# Create an instance of the KLLSData struct.
kl = jl.KLLSModel(A, b)

# Optionally, install a prior `q` on the solution via
# data = jl.KLLSModel(A, b, q=q)

# Set regularization and scaling parameters other than these defaults:
# - λ = √ϵ, where ϵ is the machine epsilon.
# - scale == sum(x0) = 1 
jl.regularize(kl, 1e-4)  # set new regularization parameter
jl.scale(kl, np.sum(x0)) # reset solution scale

# Solve the problem.
p = jl.solve(kl, logging=1) # set logging = 0 to turn off logging

# The solution is in p[0]
x = p.solution

# Plot the solution
plt.plot(x)
plt.show()

# Evaluate the log-sum-exp function and its gradient
def logexp(x):
    return jl.lse(kl.lse, x)
def gradlogexp(x):
    # The gradient is computed as a by-product of the objective call `lse` above. If the gradient needs to be computed separately and there's no guarantee that `logexp` has been called at `x`, uncomment this line.
    logexp(x)
    return jl.KLLS.grad(kl.lse)

logexp(x) # log-sum-exp at x
gradlogexp(x) # gradient of log-sum-exp at x