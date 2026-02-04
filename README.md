# Surveys.jl

A Julia package for design-based inference in survey sampling. This package provides functionality comparable to R's `survey` package, with native Julia performance and integration with DataFrames.jl.

## Overview

`Surveys.jl` implements methods for analyzing data from complex survey designs, including:

- Simple random sampling (SRS)
- Stratified sampling
- One-stage cluster sampling
- Two-stage cluster sampling
- Taylor series variance estimation

The main exported type is `SampleSum`, which stores both an estimate and its variance.

## Simple Random Sampling

For simple random sampling without replacement, use `π_sum` with the finite population correction (FPC).

```julia
using Surveys, DataFramesMeta
result = @combine(apisrs, :total = π_sum(:enroll, N))
```

The `π_sum` function computes the Horvitz-Thompson estimator for the total and its variance, accounting for the finite population correction.

## Stratified Sampling

For stratified sampling, compute subtotals within each stratum, then combine them.

```julia
strat_result = @chain apistrat begin
    @groupby(:stype)
    @combine(:subtotal = π_sum(:enroll, Int(:fpc[1])))
    @combine(:total = sum(:subtotal))
end
```

The `SampleSum` type supports addition, so stratified estimates can be combined by summing the `subtotal` column.

## One-Stage Cluster Sampling

For one-stage cluster sampling, first aggregate within clusters, then use `π_sum` on the cluster totals.

```julia
gdf = groupby(cal_crime, :county)
@chain gdf begin
    @combine(:subtotal = sum(:Burglary))
    @combine(:total = π_sum(:subtotal, N_counties))
end
```

## Two-Stage Cluster Sampling

For two-stage sampling, apply `π_sum` twice: once within primary sampling units (PSUs), then across PSUs.

```julia
@chain df begin
    @groupby(:county)
    @combine(:subtotal = π_sum(:Burglary, county_sizes[first(:county)]))
    @combine(:total = π_sum(:subtotal, N_counties))
end
```

The variance calculation properly accounts for both stages of sampling through the nested application of `π_sum`.

## Ratio Estimation

For ratio estimation or other nonlinear functions of totals, use `π_sum` with a `Function` argument for Taylor series linearization.

```julia
# Estimate ratio of api.stu to enroll
ratio_result = @combine(apisrs, :total = 
    π_sum((a -> a[1] / a[2]), [:api_stu :enroll], Int(:fpc[1])))
```

The `π_sum` function uses automatic differentiation to compute the Taylor series approximation to the variance of nonlinear estimators.

## Linearization with Stratification and Clustering

When `π_sum` is passed a Matrix instead of a Vector, it creates a `SampleSums` object instead. This can be passed to `sum` or `π_sum` to get clustered or stratified Taylor series variance estimates.

```julia
@chain apistrat begin
    @groupby(:stype)
    @combine(:subtotal=π_sum([:api_stu :enroll],  Int(:fpc[1])))
    @combine(:total=sum(a->a[1] / a[2], :subtotal))
end
```
