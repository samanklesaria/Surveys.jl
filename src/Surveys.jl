module Surveys
using Statistics, StatsBase, DataFrames

# General Utilities
# TODO: perhaps these are unnecessary. Easier just to use DataFramesMeta manually.

function stratified(f, df::DataFrame, cols, Ns::Dict, strata::Symbol)
    sum(combine(groupby(df, strata), ([strata; cols] => ((s, c...)->f(c..., Ns[s])) => :val)[!, :val]))
end

unstratified(f, df::DataFrame, cols, N) = f(df[!, cols]..., N)

# SI Designs

Base.sum(xs::Vector, N::Int) = N * mean(xs)

function sum_var(xs::Vector, N::Int)
    N^2 * (1 / length(x) - 1 / N) * var(xs; corrected=true)
end

# SIR Designs

Base.sum(xs::Vector, probs::Vector, N::Int) = sum(xs ./ probs) # also works for non-replacement
# Note: same as in StatsBase.

function sum_var(xs::Vector, probs::Vector, N::Int)
    N^2 * var(xs, ProbabilityWeights(1./ probs); corrected=true) / length(xs)
end

# General sampling without replacement

sum_var(strata_info::Tuple{Int, Matrix}) = (x, probs)-> begin
    (joint_probs, N) = strata_info
    Δ = (1 .- (probs .* probs')) ./ joint_probs
    Δ[diagind(Δ)] .= 1 .- probs
    y = x ./ probs
    y' * (Δ * y)
end

# Two Stage Cluster Sampling

# @chain df begin
#   @groupby(cluster)
# 	@combine(:total= sum(:xs, N2s[:cluster[1]]))
# 	@select(result=sum(:total, N1))
# end

# One Stage Cluster Sampling
# @chain df begin
#     @groupby(cluster)
# 	@combine(:total= sum(:xs))
# 	@select(result=sum(:total, N1))
# end

# This same pattern (above) can be used to get overall variance, just using `sum_var` instead of sum.

# What about CIs? Maybe have a type that captures both variance and point estimate? Could use Normal from Distributions.
# What about sampling clusters?
# When we're sampling whole groups, at least for the total, just sum the clusters first, and use the existing functions above.




# When two stage sampling with replacement, we can just multiply the probabilities.
# to get a pwr estimate.
# NOT quite- that doesn't account for a lack of independence.

# TODO: other functions.

# Sampling Designs to Support:
# Cluster sampling (clusters of one form, within of another)
# Given replicate weights.
# Linearization



end # module Surveys
