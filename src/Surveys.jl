module Surveys
using Statistics, StatsBase, DataFrames, StatsAPI, HypothesisTests, StatsModels
using DiffResults, ForwardDiff
import ForwardDiff: gradient!
import DiffResults: GradientResult

export SampleSum, π_sum, pwr_sum, apply_π_sum, π_lm

"""
    SampleSum

A structure containing a point estimate and its variance for survey statistics.

# Fields
- `sum::Float64`: The point estimate (e.g., estimated total, mean, or ratio)
- `var::Float64`: The variance of the estimate

# Examples
```julia
# Create a sample sum
est = SampleSum(1000.0, 25.0)

# Combine estimates from two strata
total = SampleSum(500.0, 10.0) + SampleSum(500.0, 15.0)

# Get confidence interval
confint(est)
```
"""
struct SampleSum
    sum::Float64
    var::Float64
end

Base.:+(a::SampleSum, b::SampleSum) = SampleSum(a.sum + b.sum, a.var + b.var)

function StatsAPI.confint(a::SampleSum, args...; kwargs...)
    confint(OneSampleZTest(a.sum, sqrt(a.var)), args...; kwargs...)
end

# π Estimation

"""
    π_sum(xs::AbstractVector{<:Real}, N::Int) -> SampleSum

Compute the Horvitz-Thompson estimate of a population total for simple random sampling without replacement.

# Arguments
- `xs`: Vector of observations from the sample
- `N`: Population size (finite population correction)

# Returns
A `SampleSum` containing the estimated total and its variance.

# Details
Computes the estimator T̂ = N * mean(xs) with variance:
Var(T̂) = N² × (1/n - 1/N) × s²

where n is the sample size and s² is the sample variance.

# Examples
```julia
# Sample 100 observations from population of 1000
sample_data = [12.5, 15.3, 10.2, ...]  # length 100
total_est = π_sum(sample_data, 1000)
```

# See also
R's `survey::svytotal()` with `svydesign(id=~1, fpc=~fpc)`
"""
π_sum(xs::AbstractVector{<:Real}, N::Int) = SampleSum(
    N * mean(xs),
    N^2 * (1 / length(xs) - 1 / N) * var(xs; corrected=true))

"""
    π_sum(xs::AbstractVector{SampleSum}, N::Int) -> SampleSum

Combine cluster-level estimates for multi-stage sampling designs.

# Arguments
- `xs`: Vector of `SampleSum` objects, one per sampled cluster
- `N`: Total number of clusters in the population

# Returns
A `SampleSum` with the combined estimate and variance accounting for both between-cluster
and within-cluster variation.

# Details
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
    π_sum(xs::AbstractVector{<:Real}, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int) -> SampleSum

Horvitz-Thompson estimator with arbitrary inclusion probabilities.

# Arguments
- `xs`: Vector of observations from the sample
- `probs`: Vector of first-order inclusion probabilities (πᵢ)
- `joint_probs`: Matrix of second-order inclusion probabilities (πᵢⱼ)
- `N`: Population size

# Returns
A `SampleSum` with the estimated total and its variance using the full Horvitz-Thompson variance.

# Details
Uses the unbiased variance estimator:
Var(T̂) = Σᵢ Σⱼ (πᵢⱼ - πᵢπⱼ) / πᵢⱼ × (yᵢ/πᵢ) × (yⱼ/πⱼ)

This is appropriate for unequal probability sampling without replacement.

# Examples
```julia
# Probability proportional to size sampling
π_sum(sample_values, inclusion_probs, joint_inclusion_probs, N)
```
"""
function π_sum(xs::AbstractVector{<:Real}, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int)
    Δ = 1 .- (probs .* probs') ./ joint_probs
    y = xs ./ probs
    SampleSum(sum(y), y' * (Δ * y))
end

"""
    π_sum(xs::AbstractVector{SampleSum}, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int) -> SampleSum

Combine cluster estimates with unequal probability sampling.

# Arguments
- `xs`: Vector of `SampleSum` objects from sampled clusters
- `probs`: Vector of cluster selection probabilities
- `joint_probs`: Matrix of joint selection probabilities
- `N`: Total number of clusters in population

# Returns
A `SampleSum` accounting for unequal probability cluster sampling with within-cluster variation.
"""
function π_sum(xs::AbstractVector{SampleSum}, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int)
    Δ = 1 .- (probs .* probs') ./ joint_probs
    y = [x.sum for x in xs] ./ probs
    SampleSum(sum(y), y' * (Δ * y), sum([x.var for x in xs] ./ probs))
end

"""
    apply_π_sum(f::Function, xs::Matrix{<:Real}, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int) -> SampleSum

Estimate a nonlinear function of totals using Taylor series linearization with unequal probability sampling.

# Arguments
- `f`: Function to apply to the vector of population totals
- `xs`: Matrix where each column represents a variable
- `probs`: Vector of inclusion probabilities
- `joint_probs`: Matrix of joint inclusion probabilities
- `N`: Population size

# Returns
A `SampleSum` with the estimated value of f and its variance via the delta method.

# Details
Uses automatic differentiation to compute the gradient of f, then applies the delta method:
Var(f(T̂)) ≈ ∇f' × Var(T̂) × ∇f

# Examples
```julia
# Ratio estimator with unequal probabilities
apply_π_sum(x -> x[1]/x[2], [numerator denominator], probs, joint_probs, N)
```
"""
function apply_π_sum(f::Function, xs::Matrix{<:Real}, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int)
    x0 = vec(sum(xs ./ reshape(probs, (:, 1)); dims=1))
    result = GradientResult(x0)
    gradient!(result, f, x0)
    ∇f = DiffResults.gradient(result)
    u = (xs * ∇f) ./ probs
    Δ = 1 .- (probs .* probs') ./ joint_probs
    SampleSum(DiffResults.value(result), u' * (Δ * u))
end

"""
    apply_π_sum(f::Function, xs::Matrix{<:Real}, N::Int) -> SampleSum

Estimate a nonlinear function of totals using Taylor series linearization for simple random sampling.

# Arguments
- `f`: Function to apply to the vector of population totals
- `xs`: Matrix where each column represents a variable (rows are observations)
- `N`: Population size

# Returns
A `SampleSum` with the estimated value of f and its variance computed via the delta method.

# Details
Uses automatic differentiation (ForwardDiff.jl) to compute gradients, avoiding manual derivation.
The variance is approximated using first-order Taylor expansion.

# Examples
```julia
# Estimate ratio of two variables
data = [api_stu enroll]  # n×2 matrix
ratio = apply_π_sum(x -> x[1]/x[2], data, N)

# Coefficient of variation
cv = apply_π_sum(x -> sqrt(x[2] - x[1]^2/N)/x[1], [totals totals_sq], N)
```

# See also
R's `survey::svyratio()`, `survey::svycontrast()`
"""
function apply_π_sum(f::Function, xs::Matrix{<:Real}, N::Int)
    x0 = N * vec(mean(xs; dims=1))
    result = GradientResult(x0)
    gradient!(result, f, x0)
    ∇f = DiffResults.gradient(result)
    u = xs * ∇f
    SampleSum(DiffResults.value(result), N^2 * (1 / size(xs, 1) - 1 / N) * var(u; corrected=true))
end

"""
    π_lm(f::FormulaTerm, df, N::Int) -> SampleSum

Fit a linear regression model accounting for survey design (simple random sampling).

# Arguments
- `f`: Formula specifying the model (e.g., `@formula(y ~ x1 + x2)`)
- `df`: DataFrame containing the variables
- `N`: Population size

# Returns
A `SampleSum` containing regression coefficients and their covariance matrix.

# Details
Computes design-based variance estimates for regression coefficients using the finite
population correction. Uses heteroskedasticity-robust variance estimation.

# Examples
```julia
using StatsModels
model = π_lm(@formula(income ~ age + education), survey_data, N)
```

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
    π_lm(f::FormulaTerm, df, probs::AbstractVector{<:Real}, N::Int) -> SampleSum

Fit a weighted linear regression model with unequal probability sampling.

# Arguments
- `f`: Formula specifying the model
- `df`: DataFrame containing the variables
- `probs`: Vector of inclusion probabilities
- `N`: Population size

# Returns
A `SampleSum` with regression coefficients and design-based covariance matrix.

# Details
Fits a weighted least squares model where weights are inverse inclusion probabilities,
with variance accounting for the sampling design.

# Examples
```julia
model = π_lm(@formula(y ~ x), data, inclusion_probs, N)
```
"""
function π_lm(f::FormulaTerm, df, probs::AbstractVector{<:Real}, N::Int)
    y, X = modelcols(f, df)
    Λ = Diagonal(1 ./ probs)
    XX = Xt_A_X(Λ, X)
    β = XX \ (X' * Λ * y)
    R = Diagonal(y - X * β)
    Δ = 1 .- (probs .* probs') ./ joint_probs
    V = X_A_Xt(Δ, X' * Λ * R)
    SampleSum(β, XX \ (XX \ V)')
end

# TODO: derive fixed size design for this.
# Also: what about Taylor linearization with clustering? Or stratification?

function pps_weight(xs::Vector{<:Real})
    # TODO
end

# P Estimation (Probability Weighted Ratio)

"""
    pwr_sum(xs::AbstractVector{<:Real}, probs::Vector{<:Real}, N::Int) -> SampleSum

Probability-weighted ratio estimator (Hansen-Hurwitz estimator) for sampling with replacement.

# Arguments
- `xs`: Vector of observations from the sample
- `probs`: Vector of selection probabilities
- `N`: Population size

# Returns
A `SampleSum` with the weighted estimate and its variance.

# Details
Computes the estimator: T̂ = mean(xᵢ/pᵢ) with variance Var(T̂) = var(xᵢ/pᵢ)/n

This is appropriate for sampling with replacement or when joint inclusion probabilities
are unknown. It is generally less efficient than the Horvitz-Thompson estimator but
simpler to compute.

# Examples
```julia
# Sampling with replacement with known selection probabilities
pwr_sum(sample_data, selection_probs, N)
```

# See also
`π_sum` for sampling without replacement
"""
function pwr_sum(xs::AbstractVector{<:Real}, probs::Vector{<:Real}, N::Int)
    y = xs ./ probs
    SampleSum(mean(y), var(y; corrected=true) / length(xs))
end

"""
    pwr_sum(xs::AbstractVector{SampleSum}, probs::AbstractVector{<:Real}, N::Int) -> SampleSum

Combine cluster estimates using probability-weighted ratio estimation.

# Arguments
- `xs`: Vector of `SampleSum` objects from sampled clusters
- `probs`: Vector of cluster selection probabilities
- `N`: Population size

# Returns
A `SampleSum` combining cluster-level estimates with variance.

# Details
Used for multi-stage designs where clusters are sampled with replacement or when
joint inclusion probabilities are unavailable.
"""
function pwr_sum(xs::AbstractVector{SampleSum}, probs::AbstractVector{<:Real}, N::Int)
    y = [x.sum for x in xs] ./ probs
    SampleSum(mean(y), var(y; corrected=true) / N)
end


end # module Surveys
