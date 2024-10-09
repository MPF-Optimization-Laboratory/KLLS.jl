"""
    An instance of this class will be used by the CLSModel when solving the system
    
    An inheriter of this abstract class must implement:

"""
abstract type Regularizer{T<:AbstractFloat, V<:AbstractVector{T}} end
