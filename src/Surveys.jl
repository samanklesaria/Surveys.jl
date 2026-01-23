module Surveys
using Statistics, StatsBase

struct FiniteSample{W<:AbstractWeights,T<:Real}
    N::T
    weights::W
end

struct SRSNoReplace <: AbstractWeights{Int,Int,Vector{Int}}
end

struct NoReplaceProbabilityWeights{T<:Real} <: AbstractWeights{T,T,Vector{T}}
    probs::Vector{T}
    joint_probs::Matrix{T}
end

FiniteSample(N::Int) = FiniteSample(N, SRSNoReplace())

# Sampling with replacement
# TODO: if we have N, we should rake the weights.

Base.sum(x::AbstractVector, w::FiniteSample{<:ProbabilityWeights}) = sum(x, w.weights)
function Statistics.var(::typeof(sum), x::AbstractVector, w::FiniteSample{<:ProbabilityWeights})
    var(x, w.weights; corrected=true) / length(x)
end

function Statistics.mean(::typeof(sum), x::AbstractVector, w::FiniteSample{<:ProbabilityWeights})
    var(sum, x, w) / w.N
end

# Simple random sampling without replacement

Base.sum(x::AbstractVector, w::FiniteSample{Nothing}) = w.N * mean(x)

function Statistics.var(::typeof(mean), x::AbstractVector, w::FiniteSample{Nothing})
    (1 / length(x) - 1 / w.N) * var(x; corrected=true)
end

function Statistics.var(::typeof(sum), x::AbstractVector, w::FiniteSample{Nothing})
    w.N^2 * mean(sum, x, w)
end

# General sampling without replacement

Base.sum(x::AbstractVector, w::FiniteSample{<:NoReplaceProbabilityWeights}) = sum(x, ProbabilityWeights(w.weights.probs))

function Statistics.var(::typeof(sum), x::AbstractVector, w::FiniteSample{NoReplaceProbabilityWeights})
    Δ = (1 .- (w.weights.probs .* w.weights.probs')) ./ w.weights.joint_probs
    Δ[diagind(Δ)] .= 1 .- w.weights.probs
    y = x ./ w.weights.probs
    y' * (Δ * y)
end

# Next up: stratified designs.
# We need the size of each strata, and what strata each observation belongs to.
# This becomes easier with DataFrames
# TODO

# Sampling Designs to Support:
# Independent sampling without replacement but unequal probabilities and known N
# Independent sampling without replacement but unequal probabilities without known N
# Stratified (of another design)
# Cluster sampling (clusters of one form, within of another)
# Given replicate weights.
# Linearization



end # module Surveys
