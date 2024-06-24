import numpy as np
import matplotlib.pyplot as plt

#######################################################################
# The next block of code installs the Julia solver and dependencies.
#
# IMPORTANT: requires the `juliacall` Python package, eg, from the
# command line, run `pip install juliacall`.
#
# Because Python doesn't allow `!` in variable names, we need to
# rename the methods `solve!`, `scale!`, and `regularize!` to
# `solve`, `scale`, and `regularize`, respectively.
#######################################################################
from juliacall import Main as jl
jl.seval("""
         import Pkg
         Pkg.add(url=\"git@github.com:MPF-Optimization-Laboratory/KLLS.jl.git\")
         using KLLS
         solve = KLLS.solve!
         scale = KLLS.scale!
         regularize = KLLS.regularize!
         """)
#######################################################################

# Generate some data
n = 100
m = 200
x0 = np.random.rand(n)
Anp = np.random.rand(m, n)
bnp = Anp @ x0

# Convert data to Julia
A = jl.convert(jl.Matrix, Anp)
b = jl.convert(jl.Vector, bnp)

# Create an instance of the KLLSData struct.
data = jl.KLLSModel(A, b)

# Optionally, install a prior `q` on the solution via
# data = jl.KLLSModel(A, b, q=q)

# Set regularization and scaling parameters other than these defaults:
# - λ = √ϵ, where ϵ is the machine epsilon.
# - scale == sum(x0) = 1 
jl.regularize(data, 1e-4)  # set new regularization parameter
jl.scale(data, np.sum(x0)) # reset solution scale

# Solve the problem.
p = jl.solve(data, logging=1) # set logging = 0 to turn off logging

# The solution is in p[0]
x = p.solution

# Plot the solution
plt.plot(x)
plt.show()
