### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 7b6f4670-551d-11ef-158f-b1467b0e39c7
using NLPModels, LinearAlgebra, JSOSolvers, UnPack, DataFrames, Roots, UnicodePlots

# ╔═╡ 7ac4655c-55a8-4f0e-a37d-c15f9d308590
using PlutoLinks: @revise

# ╔═╡ 88e93545-52e8-4fe7-9101-6151c8da4bc4
@revise using MyPackage

# ╔═╡ e3741fad-e818-4a94-a5fa-5dddc997fb84
md"# Self-Scaling Method"

# ╔═╡ 4f94bbcb-00f1-4df8-940a-899142f4f8fb
# ╠═╡ disabled = true
#=╠═╡
using PlutoLinks: @revise
  ╠═╡ =#

# ╔═╡ c3472e68-45e0-4f80-92f5-49b0e6b895cb
dir = joinpath(homedir(), "Documents", "Projects", "Software", "KLLS.jl")

# ╔═╡ 5e6efdbd-336a-43b2-ae65-c200af4e4013
begin
    using Pkg
    Pkg.add("TestEnv")
    Pkg.activate(dir)
    using TestEnv
    TestEnv.activate()
    Pkg.add("PlutoLinks")
end

# ╔═╡ c4452ada-3927-42df-aaad-423e0565374d
md"#### Define test problem"

# ╔═╡ 522118fb-723e-4892-af9e-794d5f11ff13
kl = let
	m, n = 2, 3
	A = randn(m, n)
	b = randn(m)
	KLLSModel(A, b)
end;

# ╔═╡ d0851366-ca37-4216-ba44-70cbd8a1ce48
md"""
#### Implement self-scaling model
"""

# ╔═╡ 5a0eda36-2f9a-4dbe-a5d3-60498abfb973
md"""
#### Residual

Optimality conditions:

$F(y, \tau) =
  \begin{bmatrix}
     τ A∇\log\sum\exp(A^Ty) + λCy - b
   \\\log\sum\exp(A^T y) - \log\tau - 1
  \end{bmatrix}$
"""

# ╔═╡ 5338d560-f4a6-47f3-98fc-708f4116ff63
begin
	struct SSModel{T, S, K<:KLLSModel{T}} <: AbstractNLSModel{T,S}
		kl::K
		meta::NLPModelMeta{T,S}
		nls_meta::NLSMeta{T,S}
		counters::NLSCounters
	end

	function SSModel(kl::KLLSModel{T}) where T
		m = kl.meta.nvar
		y0 = kl.meta.x0
		meta = NLPModelMeta(
			m+1,
			x0 = vcat(y0, one(T)),
			name = "Scaled Simplex Model"
		)
		nls_meta = NLSMeta{T, Vector{T}}(m+1, m+1)
		return SSModel(kl, meta, nls_meta, NLSCounters())
	end

	function Base.show(io::IO, ss::SSModel)
		println(io, "Self-scaled model")
		show(io, ss.kl)
	end
end

# ╔═╡ f071610f-f753-49ae-af99-03fc38f68a6b
ss = SSModel(kl)

# ╔═╡ 114dd764-41d1-43d0-bebe-d81556fa0950
residual(ss, [zeros(2); 1.0])

# ╔═╡ 5a7166ff-9106-42d7-b690-cd68fb901869
KLLS.dGrad!(ss.kl, zeros(2), zeros(2))

# ╔═╡ f7e3cecc-9cb8-44dd-bcf4-ed40088cd33b
function NLPModels.residual!(
	ss::SSModel,
	yt::AbstractVector,
	r::AbstractVector)
  	
	increment!(ss, :neval_residual)
	@unpack A, w, lse = ss.kl
	m = kl.meta.nvar

	y = yt[1:m]
	τ = yt[end]
	
	mul!(w, A', y)
    logsumexp = KLLS.obj!(lse, w)
	
	r[1:m] .= KLLS.dGrad!(ss.kl, y, r[1:m])	
	r[end] = logsumexp - log(τ) - 1
	
	return r
end

# ╔═╡ f240e90f-2e3e-4971-b1cb-2b3692e16704
md"## Minimal NLS Model Test"

# ╔═╡ c0d3e590-7a05-4414-865a-738520eb2092
md"Model definition"

# ╔═╡ 4e4761d4-d9fa-42cc-8608-224526fe4049
begin
	 struct SimpleNLSModel{T, S} <: AbstractNLSModel{T, S}
		meta::NLPModelMeta{T, S}
  		nls_meta::NLSMeta{T, S}
  		counters::NLSCounters
	end

	function SimpleNLSModel(::Type{T}, m, n) where T
  		meta = NLPModelMeta(
    		n,
    		x0 = ones(T, n),
    		name = "Simple NLS Model",
  		)
  		nls_meta = NLSMeta{T, Vector{T}}(m, n)
		return SimpleNLSModel(meta, nls_meta, NLSCounters())
	end
end;

# ╔═╡ 36fe5782-ae84-4342-9960-7f56a42f6806
begin
  m = 3
  n = 2
  nls = SimpleNLSModel(Float64, m, n)
end;

# ╔═╡ d3b6118a-e15c-4825-a600-fae940fe94c6
let
	x = ones(n)
	r = residual(nls, x)
    g = grad(nls, x)
	Jg = jprod_residual(nls, x, g)
	Jtr = jtprod_residual(nls, x, r)
	@assert all(g .== Jtr)

	J = jac_op_residual(nls, x)
	@assert all(J'*r .== Jtr)
	@assert all(Jg .== Jg)
end

# ╔═╡ f43f81c7-38d7-4269-8015-9acc655d557a
md"#### Solve NLS problem using Trunk"

# ╔═╡ d289f31a-99b2-4a0a-a933-10d0998d8a77
stats = trunk(nls, verbose=0, subsolver_verbose=0)

# ╔═╡ cb7cdb4d-aa2f-40b3-a2ba-7c6adf2c1fe8
md"""
#### Residual
Objective is squared residual: $f(x) = \Vert r(x) \Vert^2$.
"""

# ╔═╡ 75de8697-05ea-42d4-8e02-8d5583103ad4
function NLPModels.residual!(
	nls::SimpleNLSModel,
	x::AbstractVector,
	Fx::AbstractVector)
  increment!(nls, :neval_residual)
  Fx .= [1 - x[1]
         10 * (x[2] - x[1]^2)
         x[1]*x[2]
        ]
  return Fx
end

# ╔═╡ 1b1d5e1b-e9e4-4aa8-9120-78cd0e4a495e
begin
	x = randn(n)
	r = zeros(m)

	# Residual: r(x)
	rx1 = residual(nls, x)
	rx2 = residual!(nls, x, similar(r))
	@assert all(rx1 .== rx2)

	# Objective: f(x) = 1/2<r(x), r(x)>
	# Gradient:  g(x) = J'(x) r(x)
	fx1 = dot(rx1,rx1)/2
	fx2, gx2 = objgrad!(nls, x, similar(x))
	fx3, gx3 = objgrad!(nls, x, similar(x), rx1, recompute=false) # reuses `rx1` 
	fx4 = obj(nls, x)
	gx1 = grad(nls, x)
	@assert fx1 == fx2 == fx3 == fx4
	@assert all(gx2 .== gx3)
end

# ╔═╡ e4389304-1236-4cf9-a9b2-2fed46cb3295
md"""
#### Jacobian vector product

Compute product $J(x) w$ without forming Jacobian $J(x)$.
"""

# ╔═╡ ed004af6-bdbe-4981-980f-6fb0c5634188
function jacobian(x)
	J = [-1         0
         -20x[1]   10
         x[2]    x[1] ]
end;

# ╔═╡ d2916755-7ab7-4ee8-b2c1-65a4f7a06c36
function NLPModels.jprod_residual!(
  nls::SimpleNLSModel,
  x::AbstractVector,
  w::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nls, :neval_jtprod_residual)
  Jv .= jacobian(x)*w
  return Jv
end

# ╔═╡ d58954cc-1e59-4d5a-9672-4ae9ecfc094a
md"""
#### Jacobian-transpose vector product

Compute product $J'(x) v$ without forming Jacobian $J'(x)$.
"""

# ╔═╡ 68c1e0a9-2bb1-4f80-b455-8c09ce2531c9
function NLPModels.jtprod_residual!(
  nls::SimpleNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nls, :neval_jtprod_residual)
  Jtv .= jacobian(x)'*v
  return Jtv
end

# ╔═╡ Cell order:
# ╟─e3741fad-e818-4a94-a5fa-5dddc997fb84
# ╠═7b6f4670-551d-11ef-158f-b1467b0e39c7
# ╠═4f94bbcb-00f1-4df8-940a-899142f4f8fb
# ╠═5e6efdbd-336a-43b2-ae65-c200af4e4013
# ╠═7ac4655c-55a8-4f0e-a37d-c15f9d308590
# ╠═c3472e68-45e0-4f80-92f5-49b0e6b895cb
# ╠═88e93545-52e8-4fe7-9101-6151c8da4bc4
# ╟─c4452ada-3927-42df-aaad-423e0565374d
# ╠═522118fb-723e-4892-af9e-794d5f11ff13
# ╠═f071610f-f753-49ae-af99-03fc38f68a6b
# ╠═114dd764-41d1-43d0-bebe-d81556fa0950
# ╠═5a7166ff-9106-42d7-b690-cd68fb901869
# ╟─d0851366-ca37-4216-ba44-70cbd8a1ce48
# ╟─5a0eda36-2f9a-4dbe-a5d3-60498abfb973
# ╠═f7e3cecc-9cb8-44dd-bcf4-ed40088cd33b
# ╠═5338d560-f4a6-47f3-98fc-708f4116ff63
# ╟─f240e90f-2e3e-4971-b1cb-2b3692e16704
# ╟─c0d3e590-7a05-4414-865a-738520eb2092
# ╠═4e4761d4-d9fa-42cc-8608-224526fe4049
# ╠═36fe5782-ae84-4342-9960-7f56a42f6806
# ╠═1b1d5e1b-e9e4-4aa8-9120-78cd0e4a495e
# ╠═d3b6118a-e15c-4825-a600-fae940fe94c6
# ╟─f43f81c7-38d7-4269-8015-9acc655d557a
# ╠═d289f31a-99b2-4a0a-a933-10d0998d8a77
# ╟─cb7cdb4d-aa2f-40b3-a2ba-7c6adf2c1fe8
# ╠═75de8697-05ea-42d4-8e02-8d5583103ad4
# ╟─e4389304-1236-4cf9-a9b2-2fed46cb3295
# ╠═ed004af6-bdbe-4981-980f-6fb0c5634188
# ╠═d2916755-7ab7-4ee8-b2c1-65a4f7a06c36
# ╟─d58954cc-1e59-4d5a-9672-4ae9ecfc094a
# ╠═68c1e0a9-2bb1-4f80-b455-8c09ce2531c9
