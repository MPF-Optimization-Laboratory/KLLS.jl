"""
    Structure for Constrained Least Squares model to solve:

        min_{x in Constraint} (1/2)|| Ax - b ||^2

    With fields:

        - `A` is the matrix of constraints
        - `b` is the right-hand side
        - `λ` is the regularization parameter
        - `C` is a positive definite scaling matrix
        - `w` is an n-buffer for the Hessian-vector product 
        - `reg` is the regularizer function object
        - `bNrm` is the norm of the right-hand side
        - `name` is the name of the problem
        - `Constraint` is the constraint set, must be in enum Constraints
"""
@enum Constraints simplex oneBall infBall

struct CLSModel{T<:AbstractFloat, M<:AbstractMatrix{T}, CT, V<:AbstractVector{T}} <: AbstractNLPModel{T, V}
    A::M
    b::V
    λ::T
    C::CT
    w::V
    reg
    bNrm::T
    scale::T
    name::String
    meta::NLPModelMeta{T, V}
    counters::Counters
    Constraint::Constraints
end

function CLSModel(A::AbstractMatrix{T}, b::AbstractVector{T}; λ::T=√eps(T), C=I, name::String="", Constraint::Constraints=simplex) where T<:AbstractFloat
    n = size(A, 2)
    w = zeros(T, n)
    reg = nothing
    cls = CLSModel{T, typeof(A), typeof(C), typeof(b)}(
        A,
        b,
        λ,
        C,
        w,
        reg,
        norm(b),
        one(T),
        name,
        NLPModelMeta(size(A, 1), name="CLS Model"),
        Counters(),
        Constraint
    )
    set_constraint!(cls, Constraint)
    return cls
end

function Base.show(io::IO, cls::CLSModel)
    println(io, "Constrained least squares" * (cls.name == "" ? "" : ": " * cls.name))
    println(io, @sprintf("   m = %5d  bNrm = %7.1e", size(cls.A, 1), cls.bNrm))
    println(io, @sprintf("   n = %5d  λ    = %7.1e", size(cls.A, 2), cls.λ))
    println(io, @sprintf("       %5s  τ    = %7.1e", " ", cls.scale))
end

function regularize!(cls::CLSModel{T}, λ::T) where T
    cls.λ = λ
    return cls
end

function scale!(cls::CLSModel{T}, scale::T) where T
    cls.scale = scale
    return cls
end

function reset!(cls::CLSModel)
    for f in fieldnames(Counters)
        setfield!(cls.counters, f, 0)
    end
    return cls
end

function set_constraint!(cls::CLSModel, set::Constraints)
    cls.Constraint = set
    n = size(cls.A, 2)
    T = eltype(cls.A)
    if set == simplex
        cls.reg = LogSumExp(zeros(T, n))
    elseif set == oneBall
        cls.reg = LogSumCosh(zeros(T, n), zeros(T, n), zeros(T, n))
    elseif set == infBall
        cls.reg = SumLogCosh(zeros(T, n))
    else
        error("Unknown constraint set")
    end
    return cls
end