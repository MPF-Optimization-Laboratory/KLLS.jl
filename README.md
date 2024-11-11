# KLLS

This package introduces a new algorithm to solve KL regularized least squares problems, that are of form

$$ \min_{x\in \mathcal{C}} \frac{1}{2\lambda} \|Ax-b\|^2 + \langle c, x \rangle + KL(x\mid q) $$

Where $\mathcal{C}$ is either the simplex or the non-negative orthant. **TODO: Explain how it works? add a link to the paper once it is done**

## Usage

### Installation

**TODO: For this part to work, we need add the package to the julia pacakge registry**

To install this package, run:

```julia
import Pkg; Pkg.install("KLLS")
```

## Examples

**TODO: Complete this part with examples of the models that end up in the final implementation**

To solve a simple optimal transport problem

```julia
using KLLS, LinearAlgebra, Distances

μsupport = νsupport = range(-2, 2; length=100)
C = pairwise(SqEuclidean(), μsupport', νsupport'; dims=2)           # Cost matrix
μ = normalize!(exp.(-μsupport .^ 2 ./ 0.5^2), 1)                    # Start distribution
ν = normalize!(νsupport .^ 2 .* exp.(-νsupport .^ 2 ./ 0.5^2), 1)   # Target distribution

ϵ = 0.01*median(C2)                 # Entropy regularization constant
ot = KLLS.OTModel(μ, ν, C2, ϵ)      # Model initialization
solution = solve!(ot, trace=true)   # Solution to the OT problem          
```

## Developing

### Getting Started

To get started with developing the package, first pull the Github repo

```shell
git clone https://github.com/MPF-Optimization-Laboratory/KLLS.jl.git
```

Or better, [create fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) to develop!

### Usage

To be able to use the library in the julia REPL, navigate to the library root folder, and run

```bash
julia --project=. -i -e 'using Pkg; Pkg.activate(".")'
```

This will start the julia REPL, and add the repo to the julia `LOAD_PATH`. To start using, run

```julia
using KLLS
```

And start running code.

#### Using in a notebook

To use this package in a notebook while developing, add it manually to your julia `LOAD_PATH` by
```julia
push!(LOAD_PATH, "/path/to/the/library")
```

### Adding dependencies

To add new dependecies, start the REPL as above, then run

```julia
Pkg.add("Package name")
```

### Testing

Each pull requested will be tested against the current tests, and new tests are highly encouraged for any new piece of code. To run the tests, run

```bash
julia --project=. -i -e 'using Pkg; Pkg.test()'
```

From the root directory.
