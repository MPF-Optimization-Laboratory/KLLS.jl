import numpy as np
from juliacall import Main as jl

matrix = np.random.randn(3, 4)
x = np.random.dirichlet(np.ones(4), size=1)[0]
rhs = np.dot(matrix, x)

jl.seval("""
         import Pkg
         Pkg.add(url=\"github://\")
         using KLLS
         newtoncg = KLLS.solve!
         """)

A = jl.convert(jl.Matrix, matrix)
b = jl.convert(jl.Vector, rhs)

data = jl.KLLSData(A, b)
p = jl.newtoncg(data)

