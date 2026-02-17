module Surveys
using Statistics, StatsBase, DataFrames, StatsAPI, HypothesisTests, StatsModels,
    DiffResults, ForwardDiff
include("docboilerplate.jl")

export SampleSum, π_sum, pwr_sum, π_lm

"Population estimate and its variance."
struct SampleSum
    "Point estimate (e.g., estimated total, mean, or ratio)"
    sum::Float64
    "Variance of the estimate"
    var::Float64
end

Base.:+(a::SampleSum, b::SampleSum) = SampleSum(a.sum + b.sum, a.var + b.var)

function StatsAPI.confint(a::SampleSum, args...; kwargs...)
    confint(OneSampleZTest(a.sum, sqrt(a.var)), args...; kwargs...)
end

"""
Multivariate population estimates. These are usually collapsed into a
`SampleSum` using `sum(f::Function, xs)` for stratified studies or `π_sum(f::Function, xs)`
for cluster studies.
"""
struct SampleSums
    "Estimates of population totals for each variable"
    sums_M::Vector{Float64}
    "Sample x variable matrix of observations"
    samples_nM::Matrix{Float64}
    "Population size"
    N::Int
end

# π Estimation

π_sum(xs::AbstractMatrix{<:Real}, N::Int) =
    SampleSums(N * vec(mean(xs; dims=1)), xs, N)

function Base.sum(f::Function, xs::Vector{SampleSums})
    x0_M = sum(x.sums_M for x in xs)
    result = DiffResults.GradientResult(x0_M)
    ForwardDiff.gradient!(result, f, x0_M)
    ∇f_M = DiffResults.gradient(result)
    result_var = sum(xs; init=0.0) do x
        π_sum(x.samples_nM * ∇f_M, x.N).var
    end
    SampleSum(DiffResults.value(result), result_var)
end

function π_sum(f::Function, xs::Vector{SampleSums}, N::Int)
    x0_M = N * mean(x.sums_M for x in xs)
    result = DiffResults.GradientResult(x0_M)
    ForwardDiff.gradient!(result, f, x0_M)
    ∇f_M = DiffResults.gradient(result)
    us = [π_sum(x.samples_nM * ∇f_M, x.N) for x in xs]
    v = var(u.sum for u in us; corrected=true)
    SampleSum(DiffResults.value(result),
        N^2 * (1 / length(us) - 1 / N) * v + N * mean(u.var for u in us))
end

"""
Compute the Horvitz-Thompson estimate of a total for simple
random samples `xs` without replacement from a population of size `N`.

See also
R's `survey::svytotal()` with `svydesign(id=~1, fpc=~fpc)`
"""
function π_sum(xs::AbstractVector{<:Real}, N::Int)
    m, v = mean_and_var(xs; corrected=true)
    SampleSum(N * m, N^2 * (1 / length(xs) - 1 / N) * v)
end

"""
Combine cluster-level estimates for multi-stage sampling designs.

This is used for cluster sampling or multi-stage designs where you have already computed
estimates within each sampled cluster. The variance properly accounts for clustering.

# Examples
```julia
# Two-stage sampling: compute within-cluster totals, then aggregate
cluster_totals = [π_sum(cluster1_data, n1), π_sum(cluster2_data, n2)]
overall_total = π_sum(cluster_totals, N_clusters)
```

# See also
R's `survey::svydesign()` with `id=~cluster`
"""
function π_sum(xs::AbstractVector{SampleSum}, N::Int)
    m, v = mean_and_var((x.sum for x in xs); corrected=true)
    SampleSum(N * m, N^2 * (1 / length(xs) - 1 / N) * v + N * mean(x.var for x in xs))
end

"""
Horvitz-Thompson estimator for a sample `xs` from a population with size `N`
with arbitrary inclusion probabilities `probs` and pairwise inclusion
probabilities `joint_probs`.
"""
function π_sum(xs::AbstractVector{<:Real}, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int)
    Δ = 1 .- (probs .* probs') ./ joint_probs
    y = xs ./ probs
    SampleSum(sum(y), y' * (Δ * y))
end

"""
Combine cluster estimates with unequal probability sampling.
"""
function π_sum(xs::AbstractVector{SampleSum}, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int)
    Δ = 1 .- (probs .* probs') ./ joint_probs
    y = [x.sum for x in xs] ./ probs
    SampleSum(sum(y), y' * (Δ * y), sum([x.var for x in xs] ./ probs))
end

"""
Estimate a nonlinear function of totals using Taylor series linearization with unequal probability sampling.
"""
function π_sum(f::Function, xs::Matrix{<:Real}, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int)
    x0 = vec(sum(xs ./ reshape(probs, (:, 1)); dims=1))
    result = DiffResults.GradientResult(x0)
    ForwardDiff.gradient!(result, f, x0)
    ∇f = DiffResults.gradient(result)
    u = (xs * ∇f) ./ probs
    Δ = 1 .- (probs .* probs') ./ joint_probs
    SampleSum(DiffResults.value(result), u' * (Δ * u))
end

"""
Estimate a nonlinear function of totals using Taylor series linearization for simple random sampling.
"""
function π_sum(f::Function, xs::Matrix{<:Real}, N::Int)
    x0 = N * vec(mean(xs; dims=1))
    result = DiffResults.GradientResult(x0)
    ForwardDiff.gradient!(result, f, x0)
    ∇f = DiffResults.gradient(result)
    u = xs * ∇f
    SampleSum(DiffResults.value(result), N^2 * (1 / size(xs, 1) - 1 / N) * var(u; corrected=true))
end

"""
Fit a linear regression model accounting for survey design (simple random sampling).

# See also
R's `survey::svyglm()`
"""
function π_lm(f::FormulaTerm, df, N::Int)
    y, X = modelcols(f, df)
    XX = X' * X
    β = XX \ (X'y)
    V = Diagonal((y - X * β) .^ 2)
    n = length(y)
    SampleSum(β, (1 - n / N) * (n / (n - 1)) * (XX \ (XX \ X_A_Xt(V, X))'))
end

"""
Fit a weighted linear regression model with unequal probability sampling.
"""
function π_lm(f::FormulaTerm, df, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int)
    y, X = modelcols(f, df)
    Λ = Diagonal(1 ./ probs)
    XX = Xt_A_X(Λ, X)
    β = XX \ (X' * Λ * y)
    R = Diagonal(y - X * β)
    Δ = 1 .- (probs .* probs') ./ joint_probs
    V = X_A_Xt(Δ, X' * Λ * R)
    SampleSum(β, XX \ (XX \ V)')
end

function pps_weight(xs::Vector{<:Real})
    # TODO
end

# P Estimation (Probability Weighted Ratio)

"""
Probability-weighted ratio estimator (Hansen-Hurwitz estimator) for sampling with replacement.
"""
function pwr_sum(xs::AbstractVector{<:Real}, probs::Vector{<:Real}, N::Int)
    y = xs ./ probs
    SampleSum(mean(y), var(y; corrected=true) / length(xs))
end

"""
Combine cluster estimates using probability-weighted ratio estimation.
"""
function pwr_sum(xs::AbstractVector{SampleSum}, probs::AbstractVector{<:Real}, N::Int)
    y = [x.sum for x in xs] ./ probs
    SampleSum(mean(y), var(y; corrected=true) / N)
end


end # module Surveys
