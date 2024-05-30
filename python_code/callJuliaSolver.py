import numpy as np
import matplotlib.pyplot as plt

# This installs the Julia solver and dependencies
# IMPORTANT: requires the `juliacall` Python package, eg,
# `pip install juliacall`
# from the command line
from juliacall import Main as jl
jl.seval("""
         import Pkg
         Pkg.add(url=\"git@github.com:MPF-Optimization-Laboratory/KLLS.jl.git\")
         using KLLS
         newtoncg = KLLS.solve!
         """)

# Generate some data
n = 100
m = 200
x0 = np.random.rand(n)
x0 /= np.sum(x0)
Anp = np.random.rand(m, n)
bnp = Anp @ x0

# Convert data to Julia
A = jl.convert(jl.Matrix, Anp)
b = jl.convert(jl.Vector, bnp)

# Create an instance of the KLLSData struct and set the regularization parameter
data = jl.KLLSData(A, b)
data.λ = 1e-4

# Solve the problem. Solution is in p[0]
p = jl.newtoncg(data, trace=True)

# The solution is in p[0]
x = p[0]

# Plot the solution
plt.plot(x)
plt.show()

# To see the solver output log, add `trace=True` to the `newtoncg` call
p = jl.newtoncg(data, trace=True)
print(p[3])
