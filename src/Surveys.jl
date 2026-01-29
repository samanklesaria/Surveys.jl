module Surveys
using Statistics, StatsBase, DataFrames, StatsAPI, HypothesisTests, StatsModels
using DiffResults, ForwardDiff
import ForwardDiff: gradient!
import DiffResults: GradientResult

export SampleSum, π_sum, pwr_sum, apply_π_sum

struct SampleSum
    sum::Float64
    var::Float64
end

Base.:+(a::SampleSum, b::SampleSum) = SampleSum(a.sum + b.sum, a.var + b.var)

function StatsAPI.confint(a::SampleSum, args...; kwargs...)
    confint(OneSampleZTest(a.sum, sqrt(a.var)), args...; kwargs...)
end

# π Estimattion

π_sum(xs::AbstractVector{<:Real}, N::Int) = SampleSum(
    N * mean(xs),
    N^2 * (1 / length(xs) - 1 / N) * var(xs; corrected=true))

function π_sum(xs::AbstractVector{SampleSum}, N::Int)
    m, v = mean_and_var((x.sum for x in xs); corrected=true)
    SampleSum(N * m, N^2 * (1 / length(xs) - 1 / N) * v + N * mean(x.var for x in xs))
end

function π_sum(xs::AbstractVector{<:Real}, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int)
    Δ = 1 .- (probs .* probs') ./ joint_probs
    y = xs ./ probs
    SampleSum(sum(y), y' * (Δ * y))
end

function π_sum(xs::AbstractVector{SampleSum}, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int)
    Δ = 1 .- (probs .* probs') ./ joint_probs
    y = [x.sum for x in xs] ./ probs
    SampleSum(sum(y), y' * (Δ * y), sum([x.var for x in xs] ./ probs))
end

function apply_π_sum(f::Function, xs::Matrix{<:Real}, probs::AbstractVector{<:Real}, joint_probs::Matrix, N::Int)
    x0 = vec(sum(xs ./ reshape(probs, (:, 1)); dims=1))
    result = GradientResult(x0)
    gradient!(result, f, x0)
    ∇f = DiffResults.gradient(result)
    u = (xs * ∇f) ./ probs
    Δ = 1 .- (probs .* probs') ./ joint_probs
    SampleSum(DiffResults.value(result), u' * (Δ * u))
end

function apply_π_sum(f::Function, xs::Matrix{<:Real}, N::Int)
    x0 = N * vec(mean(xs; dims=1))
    result = GradientResult(x0)
    gradient!(result, f, x0)
    ∇f = DiffResults.gradient(result)
    u = xs * ∇f
    SampleSum(DiffResults.value(result), N^2 * (1 / size(xs, 1) - 1 / N) * var(u; corrected=true))
end

function π_lm(f::FormulaTerm, df, N::Int)
    y, X = modelcols(f, df)
    XX = X' * X
    β = XX \ (X'y)
    V = Diagonal((y - X * β) .^ 2)
    n = length(y)
    SampleSum(β, (1 - n / N) * (n / (n - 1)) * (XX \ (XX \ X_A_Xt(V, X))'))
end

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

# P Estimation

function pwr_sum(xs::AbstractVector{<:Real}, probs::Vector{<:Real}, N::Int)
    y = xs ./ probs
    SampleSum(mean(y), var(y; corrected=true) / length(xs))
end

function pwr_sum(xs::AbstractVector{SampleSum}, probs::AbstractVector{<:Real}, N::Int)
    y = [x.sum for x in xs] ./ probs
    SampleSum(mean(y), var(y; corrected=true) / N)
end

# What about the variance of predictions from a linear model?


# TODO:
# Docs
# Make sure the stuff so far works
# What about Taylor for p-estimation?
# GLM coefficients
# subpopulation stuff
# PPS
# self weighted

# PPS example:
# @chain df begin
# 	@combine(:total_income=si_sum(:income, pps_weight(:height)..., N))
# end

# Stratified example:
# @chain df begin
# 	@groupby(:strata)
# 	@combine(:t = si_sum(:income, N[:strata[1]]))
# 	@combine(sum(:t))
# end

# Two Stage Cluster Sampling
# @chain df begin
#   @groupby(cluster)
# 	@combine(:total= si_sum(:xs, N2s[:cluster[1]]))
# 	@select(result=si_sum(:total, N1))
# end

# One Stage Cluster Sampling
# @chain df begin
#     @groupby(cluster)
# 	@combine(:total= sum(:xs))
# 	@select(result=sum(:total, N1))
# end


end # module Surveys
