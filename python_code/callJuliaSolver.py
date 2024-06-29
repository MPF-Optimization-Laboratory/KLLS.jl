import numpy as np
import matplotlib.pyplot as plt

#######################################################################
# IMPORTANT: requires the `juliacall` Python package, eg, from the
# command line, run `pip install juliacall`.
#######################################################################
from juliacall import Main as jl

#######################################################################
# The next block of code installs the Julia solver and dependencies.
# Only needed the first time you run this script or when the KLLS solver
# needs to be updated. This line can be commented out after that.
#######################################################################
jl.seval("""
         import Pkg
         Pkg.add(url=\"git@github.com:MPF-Optimization-Laboratory/KLLS.jl.git\")
         """
         )

#######################################################################
# Because Python doesn't allow `!` in variable names, we need to
# rename the methods `solve!`, `scale!`, and `regularize!` to
# `solve`, `scale`, and `regularize`, respectively.
#######################################################################
jl.seval("""
         using KLLS
         solve = KLLS.solve!
         scale = KLLS.scale!
         regularize = KLLS.regularize!
         """
         )

# Generate some data
n = 100
m = 200
x0 = np.random.rand(n)
Anp = np.random.rand(m, n)
bnp = Anp @ x0

# Convert data to Julia
A = jl.convert(jl.Matrix, Anp)
b = jl.convert(jl.Vector, bnp)
q = jl.convert(jl.Vector, np.ones(n) / n) # optional prior on the solution

# Create an instance of the KLLSData struct.
# Default prior is uniform, ie, q = ones(n) / n
data = jl.KLLSModel(A, b, q=q) 

# Set regularization and scaling parameters other than these defaults:
# - λ = √ϵ, where ϵ is the machine epsilon.
# - scale == sum(x0) = 1 
jl.regularize(data, 1e-4)  # set new regularization parameter
jl.scale(data, np.sum(x0)) # reset solution scale

# Solve the problem.
p = jl.solve(data, logging=1) # set logging = 0 to turn off logging

# Test that the solve was successful
assert p.status == jl.Symbol("optimal")

# Print the solution status
print(p)

# Extract the solution 
x = p.solution

# Plot the solution
plt.plot(x)
plt.show()
